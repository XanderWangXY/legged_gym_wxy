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
from legged_gym.utils.isaacgym_utils import compute_meshes_normals, Point, get_euler_xyz, get_contact_normals,quat_rotate_inverse

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

class Lite3SWC(LeggedRobot):
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
        self.task_name = 'lite3swc'
        self.rew_cnt = 0

        # 阶段缓冲：每个 env 有 5 个阶段位
        self.stage_buf = torch.zeros((self.num_envs, 5), device=self.device)
        self.stage_buf[:, 0] = 1.0  # 所有环境初始处于 stage 0（站立）

        # 辅助变量：翻转状态检测
        self.is_half_turn_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.is_one_turn_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # 控制指令时间记录：记录每个 env 指令下发的时间点（单位：秒）
        self.cmd_time_buf = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        # 控制信号张量：3维 one-hot 向量 [等待跳跃, 起跳初期, 起跳完成]
        self.skill_commands = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)
        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_quat[:] = self.root_states[:, 3:7]
        self.rpy[:] = get_euler_xyz(self.base_quat)

        # update info just for terrian moveup/movedown
        if self.count % 3 == 0:  # every 3 step, means  3*12ms = 36ms
            self.episode_v_integral += torch.norm(self.root_states[:, :3] - self.old_pos, dim=-1)
            dyaw = self.rpy[:, 2] - self.old_rpy[:, 2]

            self.episode_w_integral += torch.where(
                torch.abs(dyaw) > torch.pi / 2,
                dyaw + torch.pow(-1.0,
                                 torch.less(self.rpy[:, 2], torch.pi / 2).long() + 1) * torch.pi * 2, dyaw)
            self.old_pos[:] = self.root_states[:, :3]
            self.old_rpy[:] = self.rpy

        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.world_lin_acc[:] = (self.root_states[:, 7:10] - self.world_lin_vel) / self.dt
        self.world_lin_vel[:] = self.root_states[:, 7:10]
        self.base_lin_acc[:] = quat_rotate_inverse(self.base_quat, self.world_lin_acc + self.imu_G_offset)

        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        self.feet_pos[:] = self.rigid_body_state[:, self.feet_indices, 0:3]
        self.feet_pos[:, :, :2] /= self.cfg.terrain.horizontal_scale
        self.feet_pos[:, :, :2] += self.cfg.terrain.border_size / self.cfg.terrain.horizontal_scale
        if self.cfg.terrain.mesh_type == 'trimesh' and self.cfg.env.num_privileged_obs is not None:
            self.feet_pos[:, :, 0] = torch.clip(self.feet_pos[:, :, 0], min=0., max=self.height_samples.shape[0] - 2.)
            self.feet_pos[:, :, 1] = torch.clip(self.feet_pos[:, :, 1], min=0., max=self.height_samples.shape[1] - 2.)

            if self.cfg.terrain.dummy_normal is False:
                self.contact_normal[:] = get_contact_normals(self.feet_pos, self.mesh_normals, self.sensor_forces)

        # self.cpg_phase_information = self.pmtg.update_observation()

        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        self.contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        self.contact_filt = torch.logical_or(self.contact, self.last_contacts)
        self.last_contacts = self.contact

        self._post_physics_step_callback()
        self._update_stage_buf()
        self._update_skill_commands()

        # compute observations, rewards, resets, ...
        self.check_termination()

        self.is_stage_stand = self.stage_buf[:, 0] == 1.0
        self.is_stage_down = self.stage_buf[:, 1] == 1.0
        self.is_stage_jump = self.stage_buf[:, 2] == 1.0
        self.is_stage_flip = self.stage_buf[:, 3] == 1.0
        self.is_stage_land = self.stage_buf[:, 4] == 1.0

        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        self.compute_observations()  # in some cases a simulation step might be required to refresh some obs (for example body positions)
        if self.num_privileged_obs is not None:
            self.compute_privileged_observations()
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()


    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length == 0):
            self.update_command_curriculum(env_ids)

        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)
        self.episode_v_integral[env_ids].zero_()
        self.episode_w_integral[env_ids].zero_()
        # self.pmtg.reset(env_ids)
        self.base_quat[env_ids] = self.root_states[env_ids, 3:7]
        self.rpy[env_ids] = get_euler_xyz(self.root_states[env_ids, 3:7])
        self.base_lin_vel[env_ids] = quat_rotate_inverse(
            self.base_quat[env_ids], self.root_states[env_ids, 7:10])
        self.base_ang_vel[env_ids] = quat_rotate_inverse(
            self.base_quat[env_ids], self.root_states[env_ids, 10:13])
        self.old_pos[env_ids] = self.root_states[env_ids, :3]
        self.old_rpy[env_ids] = self.rpy[env_ids]

        self._resample_commands(env_ids)

        # reset buffers
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(
                self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        # log additional curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

        self.stage_buf[env_ids] = 0
        self.stage_buf[env_ids, 0] = 1.0  # 重置为 stage 0
        self.is_half_turn_buf[env_ids] = 0
        self.is_one_turn_buf[env_ids] = 0
        self.cmd_time_buf[env_ids] = 0.0
        self.skill_commands[env_ids] = torch.tensor([1.0, 0.0, 0.0], device=self.device).unsqueeze(0)

    def compute_observations(self):
        """ Computes observations
        """
        self.obs_buf = torch.cat((  #self.base_lin_vel * self.obs_scales.lin_vel,
                                    self.commands[:, :3] * self.commands_scale,
                                    #self.projected_gravity,
                                    self.rpy * self.obs_scales.orientation,
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions,
                                    self.skill_commands,
                                    ),dim=-1)
        # add perceptive inputs if not blind
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
            self.heights=heights
            #self.obs_buf = torch.cat((self.obs_buf, heights), dim=-1)
        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec
        #print(self.commands)

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
            #self.base_ang_vel * self.obs_scales.ang_vel,
            self.root_states[:, 2].unsqueeze(-1),  # base_height
             contact_states * self.priv_obs_scales.contact_state,
             friction_coefficients * self.priv_obs_scales.friction,
             #external_forces_and_torques * self.priv_obs_scales.external_wrench,

             (self.mass_payloads - 6) * self.priv_obs_scales.mass_payload,  # payload, 1
             self.com_displacements * self.priv_obs_scales.com_displacement,  # com_displacements, 3
             (self.motor_strengths - 1) * self.priv_obs_scales.motor_strength,  # motor strength, 12
             (self.Kp_factors - 1) * self.priv_obs_scales.kp_factor,  # Kp factor, 12
             (self.Kd_factors - 1) * self.priv_obs_scales.kd_factor,  # Kd factor, 12
            self.base_lin_vel * self.obs_scales.lin_vel,
        ), dim=1)
        # print(self.privileged_obs_buf.shape)

    def _update_stage_buf(self):
        com_height = self.root_states[:, 2]
        body_z = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        contact = self.contact_filt  # shape: [num_envs, 4]

        # 翻转检测
        self.is_half_turn_buf = torch.logical_or(
            self.is_half_turn_buf,
            torch.logical_and(body_z[:, 0] < 0, body_z[:, 2] < 0)
        ).long()

        self.is_one_turn_buf = torch.logical_or(
            self.is_one_turn_buf,
            torch.logical_and(self.is_half_turn_buf.bool(), torch.logical_and(body_z[:, 0] >= 0, body_z[:, 2] >= 0))
        ).long()

        # stage 3 -> 4（落地检测 + 半圈翻转完成）
        from3_to4 = torch.logical_and(
            self.stage_buf[:, 3] == 1.0,
            torch.logical_and(contact.any(dim=1), self.is_half_turn_buf == 1)
        ).float()
        self.stage_buf[:, 3] = (1.0 - from3_to4) * self.stage_buf[:, 3]
        self.stage_buf[:, 4] = from3_to4 + (1.0 - from3_to4) * self.stage_buf[:, 4]

        # stage 2 -> 3（空中检测）
        from2_to3 = torch.logical_and(
            self.stage_buf[:, 2] == 1.0,
            contact.sum(dim=1) < 0.1
        ).float()
        self.stage_buf[:, 2] = (1.0 - from2_to3) * self.stage_buf[:, 2]
        self.stage_buf[:, 3] = from2_to3 + (1.0 - from2_to3) * self.stage_buf[:, 3]

        # stage 1 -> 2（准备起跳检测）
        from1_to2 = torch.logical_and(
            self.stage_buf[:, 1] == 1.0,
            torch.logical_and(com_height <= 0.25, contact.sum(dim=1) >= 3.5)
        ).float()
        self.stage_buf[:, 1] = (1.0 - from1_to2) * self.stage_buf[:, 1]
        self.stage_buf[:, 2] = from1_to2 + (1.0 - from1_to2) * self.stage_buf[:, 2]

        # stage 0 -> 1（站立开始）
        current_time = self.episode_length_buf * self.dt
        from0_to1 = torch.logical_and(
            self.stage_buf[:, 0] == 1.0,
            torch.logical_and(current_time > 0.5, com_height >= 0.3)
        ).float()
        self.stage_buf[:, 0] = (1.0 - from0_to1) * self.stage_buf[:, 0]
        self.stage_buf[:, 1] = from0_to1 + (1.0 - from0_to1) * self.stage_buf[:, 1]

    def _update_skill_commands(self):
        # 当前时间（秒）
        current_time = self.episode_length_buf * self.dt

        # 尚未收到指令的环境（初始阶段）
        masks0 = (self.cmd_time_buf == 0).float()

        # 收到指令且处于 [0, 0.2] 秒的执行初期
        masks1 = (1.0 - masks0) * (current_time < self.cmd_time_buf + 0.2).float()

        # 收到指令且超过 0.2 秒（稳定执行阶段）
        masks2 = (1.0 - masks0) * (1.0 - masks1)

        # 组合为 one-hot 风格的控制信号
        self.skill_commands[:, 0] = masks0
        self.skill_commands[:, 1] = masks1
        self.skill_commands[:, 2] = masks2

    # ------------ reward functions----------------
    def _reward_com_height(self):
        com_height = self.root_states[:, 2]
        reward = torch.zeros(self.num_envs, device=self.device)
        reward += self.is_stage_stand * (-torch.abs(com_height - 0.35))
        reward += self.is_stage_down * (-torch.abs(com_height - 0.20))
        reward += self.is_stage_jump * (com_height <= 1.0) * com_height
        reward += self.is_stage_flip * (com_height <= 1.0) * com_height
        reward += self.is_stage_land * (-torch.abs(com_height - 0.35))
        return reward

    def _reward_body_balance(self):
        body_z = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        reward = torch.zeros(self.num_envs, device=self.device)
        reward += self.is_stage_stand * (-torch.arccos(torch.clamp(body_z[:, 2], -1.0, 1.0)))
        reward += self.is_stage_down * (-torch.arccos(torch.clamp(body_z[:, 2], -1.0, 1.0)))
        reward += self.is_stage_jump * (-torch.abs(torch.arccos(torch.clamp(body_z[:, 1], -1.0, 1.0)) - np.pi / 2))
        reward += self.is_stage_flip * (-torch.abs(torch.arccos(torch.clamp(body_z[:, 1], -1.0, 1.0)) - np.pi / 2))
        reward += self.is_stage_land * (-torch.arccos(torch.clamp(body_z[:, 2], -1.0, 1.0)))
        return reward

    def _reward_velocity_penalty(self):
        lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        vel_penalty = torch.square(lin_vel[:, 0]) + torch.square(lin_vel[:, 1]) + torch.square(ang_vel[:, 2])
        ang_vel_y = ang_vel[:, 1]

        reward = torch.zeros(self.num_envs, device=self.device)
        reward += self.is_stage_stand * (-vel_penalty)
        reward += self.is_stage_down * (-vel_penalty)
        reward += self.is_stage_jump * (1.0 - self.is_one_turn_buf) * (-ang_vel_y) * 2.0
        reward += self.is_stage_flip * (1.0 - self.is_one_turn_buf) * (-ang_vel_y) * 2.0
        reward += self.is_stage_land * (-vel_penalty)
        return reward

    def _reward_energy_penalty(self):
        return -torch.square(self.torques).mean(dim=-1)

    def _reward_style_consistency(self):
        return -torch.square(self.dof_pos - self.default_dof_pos).mean(dim=-1)

    # ------------ cost functions----------------
    def _reward_foot_contact(self):
        foot_contact_forces = self.contact_forces[:, self.feet_indices, :]
        calf_contact_forces = self.contact_forces[:, self.calf_indices, :]
        foot_contact = ((torch.norm(foot_contact_forces, dim=2) > 10.0) |
                        (torch.norm(calf_contact_forces, dim=2) > 10.0)).float()

        reward = torch.zeros(self.num_envs, device=self.device)
        reward += self.is_stage_stand * 0.25
        reward += self.is_stage_down * 0.25
        reward += self.is_stage_jump * (1.0 - (foot_contact[:, 2] + foot_contact[:, 3]) / 2.0)
        reward += self.is_stage_flip * 0.25
        reward += self.is_stage_land * 0.25
        return reward

    def _reward_undesired_contact(self):
        contact = self.contact_forces
        term_contact = torch.any(torch.norm(contact[:, self.termination_contact_indices, :], dim=-1) > 1.0, dim=-1)
        undesired_contact = torch.any(torch.norm(contact[:, self.penalised_contact_indices, :], dim=-1) > 1.0, dim=-1)

        flag = (term_contact | undesired_contact).float()
        reward = torch.zeros(self.num_envs, device=self.device)
        reward += self.is_stage_stand * flag
        reward += self.is_stage_down * flag
        reward += self.is_stage_jump * flag
        reward += self.is_stage_flip * undesired_contact.float()
        reward += self.is_stage_land * undesired_contact.float()
        return reward

    def _reward_joint_position_limit(self):
        out_of_bounds = (self.dof_pos < self.dof_pos_limits[:, 0]) | (self.dof_pos > self.dof_pos_limits[:, 1])
        return out_of_bounds.float().mean(dim=-1)

    def _reward_joint_velocity_limit(self):
        over_speed = torch.abs(self.dof_vel) > self.dof_vel_limits
        return over_speed.float().mean(dim=-1)

    def _reward_joint_torque_limit(self):
        over_torque = torch.abs(self.torques) > self.torque_limits
        return over_torque.float().mean(dim=-1)
