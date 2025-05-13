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

class EqrRoughCfg( LeggedRobotCfg ):
    class env(LeggedRobotCfg.env):
        num_observations = 45#235-187
        num_privileged_obs = 187+36+3+1+3+4+4 # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise
        num_observation_history = 50
        num_envs = 4096

    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.40]  # x,y,z [m]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
             "fl_leg_1": 0.0,
            "fr_leg_1": 0.0,
            "hl_leg_1": 0.0,
            "hr_leg_1": 0.0,
            "fl_leg_2": -0.5,
            "fr_leg_2": -0.5,
            "hl_leg_2": -0.5,
            "hr_leg_2": -0.5,
            "fl_leg_3": 1.1,
            "fr_leg_3": 1.1,
            "hl_leg_3": 1.1,
            "hr_leg_3": 1.1,
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
        com_offset_range = [[-0.05, 0.05], [-0.05, 0.05], [-0.05, 0.05]]
        randomize_motor_strength = True
        motor_strength_range = [0.8, 1.2]
        randomize_Kp_factor = True
        Kp_factor_range = [0.8, 1.2]
        randomize_Kd_factor = True
        Kd_factor_range = [0.8, 1.2]

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'joint': 30.}  # [N*m/rad]
        damping = {'joint': 1.0}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/eqr1_description/urdf/eqr1_pin.urdf'
        name = "eqr1"
        foot_name = "leg_lee"
        # shoulder_name = "shoulder"
        # penalize_contacts_on = ["THIGH", "shoulder", "SHANK"]
        penalize_contacts_on = ["leg_l2", "leg_l3"]
        # terminate_after_contacts_on = ["TORSO", "shoulder"]
        terminate_after_contacts_on = ["base_link"]
        self_collisions = 1  # 1 to disable, 0 to enable...bitwise filter
        restitution_mean = 0.5
        restitution_offset_range = [-0.1, 0.1]
        compliance = 0.5
        flip_visual_attachments = False  # Some .obj meshes must be flipped from y-up to z-up
  
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.25
        class scales( LeggedRobotCfg.rewards.scales ):
            torques = -0.0002
            dof_pos_limits = -10.0
            stand_still = -0.0#5

    class student:
        student = False
        num_envs = 512

class EqrRoughCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
        student = False
        dagger_beta = 1.0
        num_mini_batches = 4  # mini batch size = num_envs*nsteps / nminibatches

    class student:
        num_mini_batches = 4  # mini batch size = num_envs*nsteps / nminibatches
        num_steps_per_env = 120
        num_learning_epochs = 1

    class runner( LeggedRobotCfgPPO.runner ):
        max_iterations = 15000  # number of policy updates
        run_name = ''
        experiment_name = 'rough_eqr'
        description = 'test'
        num_steps_per_env = 24

  