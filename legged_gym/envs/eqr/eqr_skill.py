# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from time import time
from warnings import WarningMessage
import numpy as np
import os
import re
from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
from legged_gym.utils.isaacgym_utils import compute_meshes_normals, Point, get_euler_xyz, get_contact_normals

import torch
from torch import Tensor
from typing import Tuple, Dict

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float
from legged_gym.utils.helpers import class_to_dict
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg
from PIL import Image as im
from PIL import ImageDraw
from tqdm import tqdm
from legged_gym.envs.base.legged_robot import LeggedRobot

class EQRSkill(LeggedRobot):
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = False
        self.init_done = False
        self._parse_cfg(self.cfg)
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)
        self.task_name = 'eqr_skill'
        self.rew_cnt = 0

    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[:, 3:7]
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state)
        self.rigid_body_pos = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[..., 0:3]
        self.rpy = get_euler_xyz(self.base_quat)  # xyzw
        self.old_rpy = self.rpy.clone()
        self.old_pos = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.old_pos[:] = self.root_states[:, :3]
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1,3)  # shape: num_envs, num_bodies, xyz axis
        self.rigid_body_state = gymtorch.wrap_tensor(rigid_body_state).view(self.num_envs, -1, 13)
        # print(self.rigid_body_state.shape)  # 4*5 + 1
        # print(self.feet_indices)
        # print("num_envs = ", self.num_envs)

        self.feet_pos = self.rigid_body_state[:, self.feet_indices, 0:3]
        # self.hip_pos = self.rigid_body_state[:, self.feet_indices-3, 0:3]

        self.sensor_forces = gymtorch.wrap_tensor(sensor_tensor).view(self.num_envs, 4, 6)[..., :3]

        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis

        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        self.skill_commands = torch.zeros(self.num_envs, self.cfg.skill_commands.num_skill_commands, dtype=torch.float,
                                    device=self.device, requires_grad=False)
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], device=self.device, requires_grad=False,) # TODO change this
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.back_last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool,device=self.device, requires_grad=False)
        self.contact = torch.zeros_like(self.last_contacts)
        self.contact_filt = torch.zeros_like(self.last_contacts)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.world_lin_vel = self.root_states[:, 7:10]
        self.world_lin_acc = torch.zeros_like(self.world_lin_vel)
        self.base_lin_acc = torch.zeros_like(self.world_lin_acc)
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.imu_G_offset = to_torch([0., 0., 9.8], device=self.device).repeat((self.num_envs, 1))

        self.is_jumping = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.is_descending = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.jump_target_height = torch.ones(self.num_envs, dtype=torch.float, device=self.device)
        self.jump_target_height = self.jump_target_height*self.cfg.params.jump_height_goal["jump_height_goal"]

        self.feet_link_indices = [i for i, name in enumerate(self.rigid_body_names) if
                        re.match(self.cfg.params.feet_name_reward["feet_name"], name)]
        # print(feet_indices)
        # print("Rigid body pos shape:", self.rigid_body_pos.shape)
        self.feet_indices_tensor = torch.tensor(self.feet_link_indices, dtype=torch.long, device=self.rigid_body_pos.device)
        # feet_indices_tensor = torch.tensor(feet_indices, dtype=torch.long, device=self.rigid_body_pos.device)
        self.foot_positions = self.rigid_body_pos[:, self.feet_indices_tensor, :]

        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
            self.heights = torch.zeros((self.num_envs, 187),
                                       device=self.device,
                                       dtype=torch.float,
                                       requires_grad=False)
        self.measured_heights = 0
        #self.friction_coeffs = None
        self.push_forces = torch.zeros((self.num_envs, self.num_bodies, 3),
                                       device=self.device,
                                       dtype=torch.float,
                                       requires_grad=False)
        self.push_torques = torch.zeros((self.num_envs, self.num_bodies, 3),
                                        device=self.device,
                                        dtype=torch.float,
                                        requires_grad=False)

        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)
        self.base_init_quat = torch.tensor([1., 0., 0., 0.], device=self.device)
    #
    def compute_observations(self):
        """ Computes observations
        """
        if self.cfg.env.num_observations==48:
            phase = torch.pi * self.episode_length_buf[:, None] * self.dt / 2
            self.obs_buf = torch.cat((  #self.base_lin_vel * self.obs_scales.lin_vel,
                                        #self.skill_commands,
                                         #self.commands[:, :3] * self.commands_scale,
                                        #self.projected_gravity,
                                        self.rpy * self.obs_scales.orientation,
                                        self.base_ang_vel  * self.obs_scales.ang_vel,
                                        (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                        self.dof_vel * self.obs_scales.dof_vel,
                                        self.actions,
                                        torch.sin(phase),
                                        torch.cos(phase),
                                        torch.sin(phase / 2),
                                        torch.cos(phase / 2),
                                        torch.sin(phase / 4),
                                        torch.cos(phase / 4),
                                        ),dim=-1)
        else:
            self.obs_buf = torch.cat((  # self.base_lin_vel * self.obs_scales.lin_vel,
                # self.skill_commands,
                self.commands[:, :3] * self.commands_scale,
                # self.projected_gravity,
                self.rpy * self.obs_scales.orientation,
                self.base_ang_vel * self.obs_scales.ang_vel,
                (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                self.dof_vel * self.obs_scales.dof_vel,
                self.actions,
            ), dim=-1)
        # add perceptive inputs if not blind
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
            self.heights=heights
            #self.obs_buf = torch.cat((self.obs_buf, heights), dim=-1)
        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec
        #print(self.commands)
    #
    def compute_privileged_observations(self):
        """ Computes privileged observations
        """
        contact_states = torch.norm(self.sensor_forces, dim=2) > 1.
        #print(contact_states)
        # contact_states = torch.norm(self.sensor_forces[:, :, :2], dim=2) > 1. # todo
        # contact_forces = self.sensor_forces.flatten(1)
        # contact_normals = self.contact_normal
        if self.friction_coeffs is not None:
            friction_coefficients = self.friction_coeffs.squeeze(-1).repeat(1, 4).to(self.device)
        else:
            friction_coefficients = torch.tensor(self.cfg.terrain.static_friction).repeat(self.num_envs, 4).to(self.device)

        # thigh_and_shank_contact = torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1
        external_forces_and_torques = torch.cat((self.push_forces[:, 0, :], self.push_torques[:, 0, :]), dim=-1)
        # # airtime = self.feet_air_time
        # # self.privileged_obs_buf = torch.cat(
        # #     (contact_states * self.priv_obs_scales.contact_state,
        # #      contact_forces * self.priv_obs_scales.contact_force,
        # #      contact_normals * self.priv_obs_scales.contact_normal,
        # #      friction_coefficients * self.priv_obs_scales.friction,
        # #      thigh_and_shank_contact * self.priv_obs_scales.thigh_and_shank_contact_state,
        # #      external_forces_and_torques * self.priv_obs_scales.external_wrench,
        # #      airtime * self.priv_obs_scales.airtime),
        # #     dim=-1)

        self.privileged_obs_buf = torch.cat((
            #self.heights.squeeze(-1),
            self.root_states[:, 2].unsqueeze(-1),#base_height
            self.root_states[:, 7:10] * self.obs_scales.lin_vel,
            #self.base_ang_vel * self.obs_scales.ang_vel,
             contact_states * self.priv_obs_scales.contact_state,
             friction_coefficients * self.priv_obs_scales.friction,
             #external_forces_and_torques * self.priv_obs_scales.external_wrench,

             (self.mass_payloads - 6) * self.priv_obs_scales.mass_payload,  # payload, 1
             self.com_displacements * self.priv_obs_scales.com_displacement,  # com_displacements, 3
             (self.motor_strengths - 1) * self.priv_obs_scales.motor_strength,  # motor strength, 12
             (self.Kp_factors - 1) * self.priv_obs_scales.kp_factor,  # Kp factor, 12
             (self.Kd_factors - 1) * self.priv_obs_scales.kd_factor,  # Kd factor, 12
        ), dim=1)
        # print(self.privileged_obs_buf.shape)


    # ------------ reward functions----------------
    def _reward_handstand_feet_height_exp(self):
        feet_indices = [i for i, name in enumerate(self.rigid_body_names) if
                        re.match(self.cfg.params.feet_name_reward["feet_name"], name)]
        # print(feet_indices)
        # print("Rigid body pos shape:", self.rigid_body_pos.shape)
        feet_indices_tensor = torch.tensor(feet_indices, dtype=torch.long, device=self.rigid_body_pos.device)
        # feet_indices_tensor = torch.tensor(feet_indices, dtype=torch.long, device=self.rigid_body_pos.device)
        foot_pos = self.rigid_body_pos[:, feet_indices_tensor, :]
        feet_height = foot_pos[..., 2]
        # print(feet_height)
        target_height = self.cfg.params.handstand_feet_height_exp["target_height"]
        std = self.cfg.params.handstand_feet_height_exp["std"]
        feet_height_error = torch.sum((feet_height - target_height) ** 2, dim=1)
        # print(torch.exp(-feet_height_error / (std**2)))
        return torch.exp(-feet_height_error / (std ** 2))
        # return 0

    def _reward_handstand_feet_on_air(self):
        """
        脚部在空奖励：
        1. 使用 self.contact_forces 判断足部是否接触地面（通过预先设置的阈值）。
        2. 如果所有足部都没有接触地面，则奖励1，否则奖励为0（或取平均）。
        """
        feet_indices = [i for i, name in enumerate(self.rigid_body_names) if
                        re.match(self.cfg.params.feet_name_reward["feet_name"], name)]
        # print(feet_indices)
        feet_indices_tensor = torch.tensor(feet_indices, dtype=torch.long, device=self.rigid_body_pos.device)
        # contact_forces: shape = (num_envs, num_bodies, 3)
        contact = torch.norm(self.contact_forces[:, feet_indices_tensor, :], dim=-1) > 1.0
        # 如果所有足部均未接触地面，reward = 1；也可以使用 mean 得到部分奖励
        reward = (~contact).float().prod(dim=1)
        # print(reward)
        return reward
        # return 0

    def _reward_handstand_feet_air_time(self):
        """
        计算手倒立时足部空中时间奖励
        """
        threshold = self.cfg.params.handstand_feet_air_time["threshold"]

        # 获取 "R.*_foot" 索引
        feet_indices = [i for i, name in enumerate(self.rigid_body_names) if
                        re.match(self.cfg.params.feet_name_reward["feet_name"], name)]
        feet_indices_tensor = torch.tensor(feet_indices, dtype=torch.long, device=self.device)

        # 计算当前接触状态
        contact = self.contact_forces[:, feet_indices_tensor, 2] > 1.0  # (batch_size, num_feet)
        if not hasattr(self, "last_contacts") or self.back_last_contacts.shape != contact.shape:
            self.back_last_contacts = torch.zeros_like(contact, dtype=torch.bool, device=contact.device)

        if not hasattr(self, "feet_air_time") or self.feet_air_time.shape != contact.shape:
            self.feet_air_time = torch.zeros_like(contact, dtype=torch.float, device=contact.device)
        contact_filt = torch.logical_or(contact, self.back_last_contacts)
        self.back_last_contacts = contact
        first_contact = (self.feet_air_time > 0.0) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - threshold) * first_contact, dim=1)
        # rew_airTime*=torch.norm(self.commands[:,:2],dim =1)>0.1
        self.feet_air_time *= ~contact_filt

        # print(rew_airTime)
        return rew_airTime

    def _reward_handstand_orientation_l2(self):
        """
        姿态奖励：
        1. 使用 self.projected_gravity（机器人基座坐标系下的重力投影）来评估姿态。
        2. 目标重力方向通过配置传入（例如 [1, 0, 0] 表示目标为竖直向上）。
        3. 对比当前和目标重力方向的 L2 距离，偏差越大惩罚越大。
        """
        target_gravity = torch.tensor(
            self.cfg.params.handstand_orientation_l2["target_gravity"],
            device=self.device
        )

        return torch.sum((self.projected_gravity - target_gravity) ** 2, dim=1)

    def _reward_hipy_angle_threshold(self):
        # 获取名称匹配 hipy_name 的关节索引
        hipy_indices = [i for i, name in enumerate(self.dof_names) if
                        re.match(self.cfg.params.hip_name_reward["hipy_name"], name)]

        # 转为 tensor，放到关节数据所在的设备上
        hipy_indices_tensor = torch.tensor(hipy_indices, dtype=torch.long, device=self.dof_pos.device)

        # 提取这些关节当前角度（shape: [num_envs, num_hipy_joints]）
        hipy_angles = self.dof_pos[:, hipy_indices_tensor]

        # 检查所有匹配关节的角度是否都 > -1（按环境维度求 and）
        is_valid = (hipy_angles > -1.0).all(dim=1)

        # 返回 float 类型的奖励值（1.0 或 0.0）
        return is_valid.float()

    def _reward_tracking_lin_vel_skill(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.root_states[:, 7:9]), dim=1)
        return torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma)

    def _reward_tracking_ang_vel_skill(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self.commands[:, 2] - self.root_states[:, 12])
        return torch.exp(-ang_vel_error / self.cfg.rewards.tracking_sigma)

    def _reward_both_feet_air(self):
        """
        检测双脚是否同时离地，如果是则给予 -1 的惩罚
        返回:
            penalty (torch.Tensor): 形状为 (batch_size,) 的惩罚值
        """
        # 获取双脚的索引（假设和原方法一致）
        feet_indices = [i for i, name in enumerate(self.rigid_body_names) if
                        re.match(self.cfg.params.feet_name_reward["feet_name"], name)]
        feet_indices_tensor = torch.tensor(feet_indices, dtype=torch.long, device=self.device)

        # 计算当前接触状态（True=触地，False=离地）
        contact = self.contact_forces[:, feet_indices_tensor, 2] > 1.0  # (batch_size, num_feet)

        # 检测是否双脚同时离地（contact = [False, False]）
        both_feet_air = torch.all(~contact, dim=1)  # (batch_size,)

        # 如果双脚同时离地，返回 -1，否则返回 0
        penalty = 1.0 * both_feet_air.float()  # (batch_size,)
        return penalty

    # ------------ backflip reward----------------

    def _reward_ang_vel_y(self):
        # current_time = self.episode_length_buf * self.dt
        # ang_vel = -self.base_ang_vel[:, 1].clamp(max=7.2, min=-7.2)
        # return ang_vel * torch.logical_and(current_time > 0.5, current_time < 1.0)

        # 计算1.8秒周期内的时间
        cycle_time = (self.episode_length_buf * self.dt) % 2.5
        ang_vel = -self.base_ang_vel[:, 1].clamp(max=7.2, min=-7.2)
        # 调整为新的空翻阶段时间(0.9-1.4)
        return ang_vel * torch.logical_and(cycle_time > 0.9, cycle_time < 1.4)

    def _reward_ang_vel_z(self):
        return torch.abs(self.base_ang_vel[:, 2])

    def _reward_lin_vel_z(self):
        # current_time = self.episode_length_buf * self.dt
        # lin_vel = self.base_lin_vel[:, 2].clamp(max=3)
        # return lin_vel * torch.logical_and(current_time > 0.5, current_time < 0.75)
        cycle_time = (self.episode_length_buf * self.dt) % 2.5
        lin_vel = self.base_lin_vel[:, 2].clamp(max=3)
        # 调整为新的起跳初期时间(0.9-1.15)
        return lin_vel * torch.logical_and(cycle_time > 0.9, cycle_time < 1.15)

    def _reward_orientation_control(self):
        # Penalize non flat base orientation
        # current_time = self.episode_length_buf * self.dt
        # phase = (current_time - 0.5).clamp(min=0, max=0.5)

        cycle_time = (self.episode_length_buf * self.dt) % 2.5
        # 调整相位计算，基于新的空翻时间段
        phase = (cycle_time - 0.9).clamp(min=0, max=0.5)

        # 使用优化后的四元数生成函数
        quat_pitch = quat_from_angle_axis(angle=4 * phase * torch.pi,axis=torch.tensor([0, 1, 0], device=self.device, dtype=torch.float))

        # 使用优化后的四元数乘法
        desired_base_quat = quat_mul(quat_pitch,self.base_init_quat.reshape(1, -1).repeat(self.num_envs, 1))

        # 使用优化后的逆旋转函数
        desired_projected_gravity = quat_rotate_inverse(desired_base_quat,self.gravity_vec)

        # 计算方向误差（保持不变）
        orientation_diff = torch.sum(torch.square(self.projected_gravity - desired_projected_gravity),dim=1)

        return orientation_diff

    def _reward_feet_height_before_backflip(self):
        # current_time = self.episode_length_buf * self.dt
        # foot_height = (self.foot_positions[:, :, 2]).view(self.num_envs, -1) - 0.02
        # return foot_height.clamp(min=0).sum(dim=1) * (current_time < 0.5)

        cycle_time = (self.episode_length_buf * self.dt) % 2.5
        foot_height = (self.foot_positions[:, :, 2]).view(self.num_envs, -1) - 0.02
        # 调整为新的准备阶段时间(0-0.9)
        return foot_height.clamp(min=0).sum(dim=1) * (cycle_time < 0.9)

    def _reward_height_control(self):
        # Penalize non flat base orientation
        # current_time = self.episode_length_buf * self.dt
        # target_height = 0.36
        # height_diff = torch.square(target_height - self.root_states[:, 2]) * torch.logical_or(current_time < 0.4, current_time > 1.4)
        # return height_diff
        cycle_time = (self.episode_length_buf * self.dt) % 2.5
        target_height = 0.36
        # 调整为新的准备阶段(0-0.8)和空翻后期(1.4-1.8)
        height_diff = torch.square(target_height - self.root_states[:, 2]) * torch.logical_or(
            cycle_time < 0.8,  # 准备阶段
            cycle_time > 1.8  # 空翻后期
        )
        return height_diff

    def _reward_actions_symmetry(self):
        actions_diff = torch.square(self.actions[:, 0] + self.actions[:, 3])
        actions_diff += torch.square(self.actions[:, 1:3] - self.actions[:, 4:6]).sum(dim=-1)
        actions_diff += torch.square(self.actions[:, 6] + self.actions[:, 9])
        actions_diff += torch.square(self.actions[:, 7:9] - self.actions[:, 10:12]).sum(dim=-1)
        return actions_diff

    def _reward_gravity_y(self):
        return torch.square(self.projected_gravity[:, 1])

    def _reward_feet_distance(self):
        current_time = self.episode_length_buf * self.dt
        cur_footsteps_translated = self.foot_positions - self.root_states[:,:3].unsqueeze(1)
        footsteps_in_body_frame = torch.zeros(self.num_envs, 4, 3, device=self.device)
        for i in range(4):
            footsteps_in_body_frame[:, i, :] = quat_apply(quat_conjugate(self.base_quat),
                                                             cur_footsteps_translated[:, i, :])

        stance_width = 0.3 * torch.zeros([self.num_envs, 1, ], device=self.device)
        desired_ys = torch.cat([stance_width / 2, -stance_width / 2, stance_width / 2, -stance_width / 2], dim=1)
        stance_diff = torch.square(desired_ys - footsteps_in_body_frame[:, :, 1]).sum(dim=1)

        return stance_diff

