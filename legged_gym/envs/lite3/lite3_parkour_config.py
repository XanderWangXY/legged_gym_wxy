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

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class Lite3ParkourCfg( LeggedRobotCfg ):
    class env(LeggedRobotCfg.env):
        num_observations = 45#235-187
        n_scan = 187
        num_privileged_obs = 187+36+3+1+3+4+4 # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise
        num_observation_history = 50
        num_envs = 4096

    class depth:
        use_camera = False
        camera_num_envs = 192
        camera_terrain_num_rows = 10
        camera_terrain_num_cols = 20

        position = [0.27, 0, 0.03]  # front camera
        angle = [-5, 5]  # positive pitch down

        update_interval = 5  # 5 works without retraining, 8 worse

        original = (106, 60)
        resized = (87, 58)
        horizontal_fov = 87
        buffer_len = 2

        near_clip = 0
        far_clip = 2
        dis_noise = 0.0

        scale = 1
        invert = True

    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.36]  # x,y,z [m]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            'FL_HipX_joint': 0.0,  # [rad]
            'HL_HipX_joint': 0.0,  # [rad]
            'FR_HipX_joint': -0.0,  # [rad]
            'HR_HipX_joint': -0.0,  # [rad]

            'FL_HipY_joint': -1.,  # [rad]
            'HL_HipY_joint': -1.,  # [rad]
            'FR_HipY_joint': -1.,  # [rad]
            'HR_HipY_joint': -1.,  # [rad]

            'FL_Knee_joint': 1.8,  # [rad]
            'HL_Knee_joint': 1.8,  # [rad]
            'FR_Knee_joint': 1.8,  # [rad]
            'HR_Knee_joint': 1.8,  # [rad]
        }

    class terrain( LeggedRobotCfg.terrain ):
        mesh_type = 'trimesh' # "heightfield" # none, plane, heightfield or trimesh
        # rough terrain only:
        measure_heights = True
        num_rows= 10 # number of terrain rows (levels)
        num_cols = 20 # number of terrain cols (types)
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete,梅花桩，沟壑，墙]
        terrain_proportions = [0., 0.1, 0.1, 0.1, 0.1, 0., 0.3, 0.3]
        # trimesh only:
        slope_treshold = 0.75 # slopes above this threshold will be corrected to vertical surfaces

    class commands:
        curriculum = False
        max_curriculum = 1.
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 6. # time before command are changed[s]
        heading_command = True # if true: compute ang vel command from heading error

        lin_vel_clip = 0.2
        ang_vel_clip = 0.4
        # Easy ranges
        class ranges:
            lin_vel_x = [0., 1.5] # min max [m/s]
            lin_vel_y = [0.0, 0.0]   # min max [m/s]
            ang_vel_yaw = [0, 0]    # min max [rad/s]
            heading = [0, 0]

        # Easy ranges
        class max_ranges:
            lin_vel_x = [0.3, 1.2] # min max [m/s]
            lin_vel_y = [-0.3, 0.3]#[0.15, 0.6]   # min max [m/s]
            ang_vel_yaw = [-0, 0]    # min max [rad/s]
            heading = [-1.6, 1.6]

        class crclm_incremnt:
            lin_vel_x = 0.1 # min max [m/s]
            lin_vel_y = 0.1  # min max [m/s]
            ang_vel_yaw = 0.1    # min max [rad/s]
            heading = 0.5

        waypoint_delta = 0.7

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.1, 1.25]
        randomize_base_mass = True
        added_mass_range = [-1., 3.]
        randomize_com_offset = True
        com_offset_range = [[-0.05, 0.01], [-0.03, 0.03], [-0.03, 0.03]]
        randomize_motor_strength = True
        motor_strength_range = [0.8, 1.2]
        randomize_Kp_factor = True
        Kp_factor_range = [0.8, 1.2]
        randomize_Kd_factor = True
        Kd_factor_range = [0.8, 1.2]

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'joint': 20.}  # [N*m/rad]
        damping = {'joint': 0.7}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/lite3/urdf/Lite3.urdf'
        name = "Lite3"
        foot_name = "FOOT"
        # shoulder_name = "shoulder"
        # penalize_contacts_on = ["THIGH", "shoulder", "SHANK"]
        penalize_contacts_on = ["THIGH", "SHANK"]
        # terminate_after_contacts_on = ["TORSO", "shoulder"]
        terminate_after_contacts_on = ["TORSO"]
        self_collisions = 1  # 1 to disable, 0 to enable...bitwise filter
        restitution_mean = 0.5
        restitution_offset_range = [-0.1, 0.1]
        compliance = 0.5
  
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.25
        class scales( LeggedRobotCfg.rewards.scales ):
            termination = -0.0
            tracking_lin_vel = 1.5
            tracking_ang_vel = 0.5
            dof_vel = -0.
            base_height = -0.
            feet_air_time = 0.0
            stand_still = -0.

            # regularization rewards
            lin_vel_z = -1.0
            ang_vel_xy = -0.05
            orientation = -1.
            dof_acc = -2.5e-7
            joint_power = -2e-5
            collision = -10.
            action_rate = -0.01
            smoothness = -0.01

            delta_torques = 0.#-1.0e-7
            torques = 0.#-0.00001
            hip_pos = 0.#-0.5
            dof_error = 0.#-0.04
            feet_stumble = 0.#-1
            feet_edge = 0.#-1

    class student:
        student = False
        num_envs = 192

class Lite3ParkourCfgPPO( LeggedRobotCfgPPO ):
    class policy(LeggedRobotCfgPPO.policy):
        terrain_hidden_dims = [512, 256, 128]
        terrain_input_dims = 187
        terrain_latent_dims = 36
        encoder_latent_dims = 12
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
        student = False
        dagger_beta = 1.0
        num_mini_batches = 4  # mini batch size = num_envs*nsteps / nminibatches

    class depth_encoder:
        if_depth = Lite3ParkourCfg.depth.use_camera
        depth_shape = Lite3ParkourCfg.depth.resized
        buffer_len = Lite3ParkourCfg.depth.buffer_len
        hidden_dims = 512
        learning_rate = 1.e-3
        num_steps_per_env = Lite3ParkourCfg.depth.update_interval * 24

    class student:
        num_mini_batches = 6  # mini batch size = num_envs*nsteps / nminibatches
        num_steps_per_env = 120
        num_learning_epochs = 1

    class runner( LeggedRobotCfgPPO.runner ):
        max_iterations = 15000  # number of policy updates
        run_name = ''
        experiment_name = 'parkour_lite3'
        description = 'test'
        num_steps_per_env = 24

  