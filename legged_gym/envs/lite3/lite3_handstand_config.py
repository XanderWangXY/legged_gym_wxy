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

class Lite3FootStandCfg( LeggedRobotCfg ):
    class env(LeggedRobotCfg.env):
        num_observations = 45#235-187
        num_privileged_obs = 36+3+1+3+4+4+1 # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise
        num_observation_history = 50
        num_envs = 4096

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
        mesh_type = 'plane' # "heightfield" # none, plane, heightfield or trimesh
        # rough terrain only:
        measure_heights = False
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
        added_mass_range = [-1., 5.]
        randomize_com_offset = True
        com_offset_range = [[-0.05, 0.10], [-0.03, 0.03], [-0.10, 0.10]]
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
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
        restitution_mean = 0.5
        restitution_offset_range = [-0.1, 0.1]
        compliance = 0.5
  
    class rewards( LeggedRobotCfg.rewards ):
        only_positive_rewards = False  # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25  # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 5.  # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1.
        soft_torque_limit = 29.
        base_height_target = 0.42
        max_contact_force = 100.
        class scales( LeggedRobotCfg.rewards.scales ):
            termination = -0.0
            tracking_lin_vel_skill = 3.  # 20.0
            tracking_ang_vel_skill = 1.5  # 6.66
            tracking_lin_vel = 0.0
            tracking_ang_vel = 0.
            lin_vel_z = -0.0
            ang_vel_xy = -0.0
            orientation = -0.0
            torques = -0.0002
            torque_limits = -1
            dof_vel = -0.
            dof_acc = -2.5e-7
            base_height = 0.#-0.5
            feet_air_time = 0.0
            collision = -2.
            feet_stumble = -0.0
            action_rate = -0.03
            stand_still = -0.
            handstand_feet_height_exp = 10.0
            handstand_feet_on_air = 1.0
            handstand_feet_air_time = 1.0
            handstand_orientation_l2 = -2.5
            hipy_angle_threshold = 0.5
            #both_feet_air = -1.0

    class commands:
        curriculum = False
        max_curriculum = 1.
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10. # time before command are changed[s]
        heading_command = True # if true: compute ang vel command from heading error
        class ranges:
            lin_vel_x = [-0.8, 0.8] # min max [m/s]
            lin_vel_y = [-0.8, 0.8]   # min max [m/s]
            ang_vel_yaw = [-1, 1]    # min max [rad/s]
            heading = [-3.14, 3.14]

    class params:  # 参数单独放在params类中
        handstand_feet_height_exp = {
            "target_height": 0.73,
            "std": 0.5
        }
        handstand_orientation_l2 = {
            "target_gravity": [-0.999, 0., -0.035]
        }
        handstand_feet_air_time = {
            "threshold": 5.0
        }
        feet_name_reward={
            "feet_name" : "F.*_FOOT"
        }
        hip_name_reward={
            "hipy_name" : "F.*_HipY_joint"
        }
        jump_height_goal={
            "jump_height_goal":0.7,
            "std": 0.25
        }
        epsilon_h = {
            "epsilon_h": 0.08
        }


    class student:
        student = False
        num_envs = 192

    class skill_commands:
        num_skill_commands = 3

class Lite3HandStandCfg( Lite3FootStandCfg ):
    class rewards(Lite3FootStandCfg.rewards):
        base_height_target = 0.42
        class scales( Lite3FootStandCfg.rewards.scales ):
            handstand_feet_height_exp = 10.0
            action_rate = -0.05
            hipy_angle_threshold = 0.

    class params(Lite3FootStandCfg.params):
        handstand_feet_height_exp = {
            "target_height": 0.70,
            "std": 0.5
        }
        handstand_orientation_l2 = {
            "target_gravity": [0.999, 0., -0.035]
        }
        feet_name_reward = {
            "feet_name": "H.*_FOOT"
        }