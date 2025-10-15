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

class Go2(LeggedRobot):
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
        self.task_name = 'go2'
        self.rew_cnt = 0

    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        for s in range(len(props)):
            props[s].friction = self.cfg.terrain.static_friction
            random_foot_restitution = self.cfg.asset.restitution_mean + torch_rand_float(
                self.cfg.asset.restitution_offset_range[0],
                self.cfg.asset.restitution_offset_range[1], (1, 1),
                device=self.device)
            if 'go2' in self.task_name:
                feet_list = [4, 8, 14, 18]
            else:
                raise Exception("")
            if s in feet_list:
                props[s].restitution = random_foot_restitution
                props[s].compliance = self.cfg.asset.compliance

        if self.cfg.domain_rand.randomize_friction:
            if env_id==0:
                # prepare friction randomization
                friction_range = self.cfg.domain_rand.friction_range
                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets,1), device='cpu')
                self.friction_coeffs = friction_buckets[bucket_ids]

            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]
        return props

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

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        self.compute_observations()  # in some cases a simulation step might be required to refresh some obs (for example body positions)
        if self.num_privileged_obs is not None:
            self.compute_privileged_observations()
        self.last_last_actions[:] = self.last_actions[:]
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
        self.last_last_actions[env_ids] = 0.
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
        self.last_last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
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

    def _reward_joint_power(self):
        return torch.sum((torch.abs(self.dof_vel)*torch.abs(self.torques)),dim=1)

    def _reward_smoothness(self):
        rew = torch.sum(torch.square(self.actions - 2 * self.last_actions + self.last_last_actions), dim=1)
        return rew