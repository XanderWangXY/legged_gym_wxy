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
import glob

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

MOTION_FILES = glob.glob('/home/ehr/wxy/legged_gym_wxy/datasets/mocap_motions_go2/*')


class Go2AMPCfg( LeggedRobotCfg ):

    class env(LeggedRobotCfg.env):
        num_observations = 45#235-187
        num_privileged_obs = 36+3+1+3+4+4  # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise
        num_observation_history = 50
        num_envs = 4096
        reference_state_initialization = False
        reference_state_initialization_prob = 0.85
        amp_motion_files = MOTION_FILES

    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.4]  # x,y,z [m]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.1,   # [rad]
            'RL_hip_joint': 0.1,   # [rad]
            'FR_hip_joint': -0.1 ,  # [rad]
            'RR_hip_joint': -0.1,   # [rad]

            'FL_thigh_joint': 0.8,     # [rad]
            'RL_thigh_joint': 1.,   # [rad]
            'FR_thigh_joint': 0.8,     # [rad]
            'RR_thigh_joint': 1.,   # [rad]

            'FL_calf_joint': -1.5,   # [rad]
            'RL_calf_joint': -1.5,    # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RR_calf_joint': -1.5,    # [rad]
        }

    class terrain( LeggedRobotCfg.terrain ):
        mesh_type = 'plane' # "heightfield" # none, plane, heightfield or trimesh
        # rough terrain only:
        measure_heights = False
        num_rows= 10 # number of terrain rows (levels)
        num_cols = 20 # number of terrain cols (types)
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete, stepping stones, wave]
        terrain_proportions = [0.0, 0.2, 0.3, 0.25, 0.25, 0., 0.0, 0.]
        # trimesh only:
        slope_treshold = 0.75 # slopes above this threshold will be corrected to vertical surfaces

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.1, 1.25]
        randomize_base_mass = True
        added_mass_range = [-1., 3.]
        randomize_com_offset = True
        com_offset_range = [[-0.04, 0.05], [-0.02, 0.02], [-0.03, 0.03]]
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
        damping = {'joint': 0.5}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/urdf/go2.urdf'
        name = "go2"
        foot_name = "foot"
        # shoulder_name = "shoulder"
        # penalize_contacts_on = ["THIGH", "shoulder", "SHANK"]
        penalize_contacts_on = ["thigh", "calf"]
        # terminate_after_contacts_on = ["TORSO", "shoulder"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 1  # 1 to disable, 0 to enable...bitwise filter
        restitution_mean = 0.5
        restitution_offset_range = [-0.1, 0.1]
        compliance = 0.5

    class params:  # 参数单独放在params类中
        feet_name_reward={
            "feet_name" : ".*foot"
        }

    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.25
        class scales( LeggedRobotCfg.rewards.scales ):
            termination = 0.0
            tracking_lin_vel = 1.5 * 1. / (.005 * 6)
            tracking_ang_vel = 0.5 * 1. / (.005 * 6)
            lin_vel_z = 0.0
            ang_vel_xy = 0.0
            orientation = 0.0
            torques = 0.#-1e-4* 1. / (.005 * 6)
            dof_vel = 0.0
            dof_acc = 0.#-2.5e-7* 1. / (.005 * 6)
            dof_pos_limits = 0.#-10.0* 1. / (.005 * 6)
            base_height = 0.0
            feet_air_time =  0.#1.0* 1. / (.005 * 6)
            collision = 0.#-0.01* 1. / (.005 * 6)
            feet_stumble = 0.0
            action_rate = 0.#-0.01* 1. / (.005 * 6)
            stand_still = 0.0
            dof_pos_limits = 0.0

    class student:
        student = False
        num_envs = 192

    class commands:
        curriculum = False
        max_curriculum = 1.
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10. # time before command are changed[s]
        heading_command = False # if true: compute ang vel command from heading error
        class ranges:
            lin_vel_x = [-1.0, 2.0] # min max [m/s]
            lin_vel_y = [-0.5, 0.5]   # min max [m/s]
            ang_vel_yaw = [-1.57, 1.57]    # min max [rad/s]
            heading = [-3.14, 3.14]

        class viewer(LeggedRobotCfg.viewer):
            debug_viz = True

class Go2AMPRoughCfg( Go2AMPCfg ):

    class env(Go2AMPCfg.env):
        num_observations = 45#235-187
        num_privileged_obs = 36+3+1+3+4+4+187  # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise
        num_observation_history = 50
        num_envs = 4096
        reference_state_initialization = False
        reference_state_initialization_prob = 0.85
        amp_motion_files = MOTION_FILES

    class terrain(Go2AMPCfg.terrain):
        mesh_type = 'trimesh'  # "heightfield" # none, plane, heightfield or trimesh
        # rough terrain only:
        measure_heights = True
        num_rows = 10  # number of terrain rows (levels)
        num_cols = 20  # number of terrain cols (types)
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete, stepping stones, wave]
        terrain_proportions = [0.0, 0.2, 0.3, 0.25, 0.25, 0., 0.0, 0.]
        # trimesh only:
        slope_treshold = 0.75  # slopes above this threshold will be corrected to vertical surfaces

    class commands(Go2AMPCfg.commands):
        curriculum = False
        max_curriculum = 1.
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10. # time before command are changed[s]
        heading_command = True # if true: compute ang vel command from heading error
        class ranges:
            lin_vel_x = [-1.0, 2.0] # min max [m/s]
            lin_vel_y = [-1.0, 1.0]   # min max [m/s]
            ang_vel_yaw = [-1, 1]    # min max [rad/s]
            heading = [-3.14, 3.14]


class Go2AMPCfgPPO( LeggedRobotCfgPPO ):
    runner_class_name = 'AMPOnPolicyRunner'
    class policy(LeggedRobotCfgPPO.policy):
        terrain_hidden_dims = None
        terrain_input_dims = 0
        terrain_latent_dims = 0
        encoder_latent_dims = 36
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
        student = False
        dagger_beta = 1.0
        amp_replay_buffer_size = 1000000
        num_learning_epochs = 5
        num_mini_batches = 4  # mini batch size = num_envs*nsteps / nminibatches

    class student:
        num_mini_batches = 1  # mini batch size = num_envs*nsteps / nminibatches
        num_steps_per_env = 120
        num_learning_epochs = 1

    class runner( LeggedRobotCfgPPO.runner ):
        max_iterations = 30000  # number of policy updatesf
        run_name = ''
        description = 'test'
        num_steps_per_env = 24

        experiment_name = 'go2_amp'
        algorithm_class_name = 'AMPPPO'
        policy_class_name = 'ActorCritic'

        amp_reward_coef = 2.0
        amp_motion_files = MOTION_FILES
        amp_num_preload_transitions = 2000000
        amp_task_reward_lerp = 0.3
        amp_discr_hidden_dims = [1024, 512]

        min_normalized_std = [0.05, 0.02, 0.05] * 4


  