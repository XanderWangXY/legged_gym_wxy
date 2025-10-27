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

class Go2RoughCfg( LeggedRobotCfg ):
    class env(LeggedRobotCfg.env):
        num_observations = 45#235-187
        num_privileged_obs = 187+36+3+1+3+4+4 # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise
        num_observation_history = 50
        num_envs = 4096
        task_name = 'go2'

    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.4]  # x,y,z [m]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.1,  # [rad]
            'RL_hip_joint': 0.1,  # [rad]
            'FR_hip_joint': -0.1,  # [rad]
            'RR_hip_joint': -0.1,  # [rad]

            'FL_thigh_joint': 0.8,  # [rad]
            'RL_thigh_joint': 1.,  # [rad]
            'FR_thigh_joint': 0.8,  # [rad]
            'RR_thigh_joint': 1.,  # [rad]

            'FL_calf_joint': -1.5,  # [rad]
            'RL_calf_joint': -1.5,  # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RR_calf_joint': -1.5,  # [rad]
        }

    class terrain( LeggedRobotCfg.terrain ):
        mesh_type = 'trimesh' # "heightfield" # none, plane, heightfield or trimesh
        # rough terrain only:
        measure_heights = True
        num_rows= 10 # number of terrain rows (levels)
        num_cols = 20 # number of terrain cols (types)
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]
        # trimesh only:
        slope_treshold = 0.75 # slopes above this threshold will be corrected to vertical surfaces

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.1, 1.25]
        randomize_base_mass = True
        added_mass_range = [-1., 3.]
        randomize_com_offset = True
        com_offset_range = [[-0.03, 0.08], [-0.02, 0.02], [-0.02, 0.04]]
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
        hip_joint_name = "hip"
        # shoulder_name = "shoulder"
        # penalize_contacts_on = ["THIGH", "shoulder", "SHANK"]
        penalize_contacts_on = ["thigh", "calf"]
        # terminate_after_contacts_on = ["TORSO", "shoulder"]
        terminate_after_contacts_on = ["base", "Head"]
        self_collisions = 1  # 1 to disable, 0 to enable...bitwise filter
        restitution_mean = 0.2
        restitution_offset_range = [-0.2, 0.2]
        compliance = 0.5
        # armature = 0.001
        # compliance_offset_range = [-0.1, 0.1]
  
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.25
        class scales( LeggedRobotCfg.rewards.scales ):
            torques = -0.0002
            dof_pos_limits = -10.0
            stand_still = -0.0#5
            joint_pos_penalty = -0.5
            smoothness = -0.002
            # smoothness = -0.001
            #PIE_rew
            # termination = -0.0
            # tracking_lin_vel = 1.5
            # tracking_ang_vel = 0.5
            # lin_vel_z = -1.0
            # ang_vel_xy = -0.05
            # orientation = -1.
            # torques = -0.0000
            # dof_vel = -0.
            # dof_acc = -2.5e-7
            # base_height = -0.
            # feet_air_time = 0.0
            # collision = -10.
            # feet_stumble = -0.0
            # action_rate = -0.01
            # joint_power = -2.0e-5
            # smoothness = -0.01

    class student:
        student = False
        num_envs = 256

class Go2RoughCfgPPO( LeggedRobotCfgPPO ):
    class policy(LeggedRobotCfgPPO.policy):
        terrain_hidden_dims = None
        terrain_input_dims = 0
        terrain_latent_dims = 0
        encoder_latent_dims = 36

    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
        student = False
        dagger_beta = 1.0
        num_mini_batches = 4  # mini batch size = num_envs*nsteps / nminibatches

    class student:
        num_mini_batches = 1  # mini batch size = num_envs*nsteps / nminibatches
        num_steps_per_env = 120
        num_learning_epochs = 1

    class runner( LeggedRobotCfgPPO.runner ):
        max_iterations = 20000  # number of policy updates
        run_name = ''
        experiment_name = 'rough_go2'
        description = 'test'
        num_steps_per_env = 24

    # class policy(LeggedRobotCfgPPO.policy):
    #     terrain_hidden_dims = [512, 256, 128]
    #     terrain_input_dims = 132
    #     terrain_latent_dims = 36
    #     encoder_latent_dims = 12

class Go2RoughCfgDVAEPPO( LeggedRobotCfgPPO ):
    runner_class_name = 'OnPolicyRunner_DVAE'
    class policy( LeggedRobotCfgPPO.policy ):
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        vh_encoder_dims=[256, 128]
        vh_decoder_dims=[128, 256]
        oh_out_dim=32
        vh_in_dim = 187 + 3
        critic_in_dim = oh_out_dim+36+3+1+4+4+45

    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
        student = False
        # dagger_beta = 1.0
        num_mini_batches = 4  # mini batch size = num_envs*nsteps / nminibatches

    class student:
        num_mini_batches = 1  # mini batch size = num_envs*nsteps / nminibatches
        num_steps_per_env = 48
        num_learning_epochs = 1

    class runner( LeggedRobotCfgPPO.runner ):
        policy_class_name = 'ActorCritic_DVAE'
        algorithm_class_name = 'PPO_DVAE'
        max_iterations = 50000  # number of policy updates
        run_name = ''
        experiment_name = 'go2_DVAE'
        description = 'test'
        num_steps_per_env = 24

class Go2RoughDWAQCfg( Go2RoughCfg ):
    class env(Go2RoughCfg.env):
        num_observation_history = 10
        task_name = 'go2dwaq'
    class rewards(Go2RoughCfg.rewards):
        class scales(Go2RoughCfg.rewards.scales):
            smoothness = -0.0

class Go2RoughCfgDWAQPPO( LeggedRobotCfgPPO ):
    runner_class_name = 'OnPolicyRunner_DWAQ'
    class policy( LeggedRobotCfgPPO.policy ):
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        vh_encoder_dims=[256, 128]
        vh_decoder_dims=[128, 256]
        oh_out_dim=32
        vh_in_dim = 187 + 3
        critic_in_dim = oh_out_dim+36+3+1+4+4+45

    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
        student = False
        # dagger_beta = 1.0
        num_mini_batches = 4  # mini batch size = num_envs*nsteps / nminibatches

    class student:
        num_mini_batches = 1  # mini batch size = num_envs*nsteps / nminibatches
        num_steps_per_env = 48
        num_learning_epochs = 1

    class runner( LeggedRobotCfgPPO.runner ):
        policy_class_name = 'ActorCritic_DWAQ'
        algorithm_class_name = 'PPO_DWAQ'
        max_iterations = 20000  # number of policy updates
        run_name = ''
        experiment_name = 'go2_DWAQ'
        description = 'test'
        num_steps_per_env = 24