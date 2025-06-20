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

from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from legged_gym.envs.a1.a1_config import A1RoughCfg, A1RoughCfgPPO
from .base.legged_robot import LeggedRobot
from .anymal_c.anymal import Anymal
from .anymal_c.mixed_terrains.anymal_c_rough_config import AnymalCRoughCfg, AnymalCRoughCfgPPO
from .anymal_c.flat.anymal_c_flat_config import AnymalCFlatCfg, AnymalCFlatCfgPPO
from .anymal_b.anymal_b_config import AnymalBRoughCfg, AnymalBRoughCfgPPO
from .cassie.cassie import Cassie
from .cassie.cassie_config import CassieRoughCfg, CassieRoughCfgPPO
from .a1.a1_config import A1RoughCfg, A1RoughCfgPPO
from .lite3.lite3_config import Lite3RoughCfg,Lite3RoughCfgPPO
from .lite3.lite3_amp_config import Lite3AMPCfg,Lite3AMPCfgPPO
from .lite3.lite3_amp import Lite3AMP
from .lite3.lite3_handstand_config import Lite3HandStandCfg, Lite3FootStandCfg
from .lite3.lite3_skill import Lite3Skill
from .lite3.lite3_skill_config import Lite3SkillCfg, Lite3SkillCfgPPO
from .lite3.lite3_parkour import Lite3Parkour
from .lite3.lite3_parkour_config import Lite3ParkourCfg, Lite3ParkourCfgPPO
from .lite3.lite3_dreamwaq import Lite3DreamWaQ
from .lite3.lite3_dreamwaq_config import Lite3DreamWaQCfg, Lite3DreamWaQCfgPPO
from .lite3.lite3_pie import Lite3PIE
from .lite3.lite3_pie_config import Lite3PIECfg,Lite3PIECfgPPO
from .lite3.lite3_swc import Lite3SWC
from .lite3.lite3_swc_config import Lite3SWCCfg,Lite3SWCCfgPPO

from .eqr.eqr import Eqr
from .eqr.eqr_skill import EQRSkill
from .eqr.eqr_config import EqrRoughCfg, EqrRoughCfgPPO
from .eqr.eqr_handstand_config import EqrFootStandCfg, EqrHandStandCfg, EqrSkillCfgPPO
from .eqr.eqr_amp import EqrAMP
from .eqr.eqr_amp_config import EqrAMPCfg, EqrAMPCfgPPO

import os

from legged_gym.utils.task_registry import task_registry

task_registry.register( "anymal_c_rough", Anymal, AnymalCRoughCfg(), AnymalCRoughCfgPPO() )
task_registry.register( "anymal_c_flat", Anymal, AnymalCFlatCfg(), AnymalCFlatCfgPPO() )
task_registry.register( "anymal_b", Anymal, AnymalBRoughCfg(), AnymalBRoughCfgPPO() )
task_registry.register( "a1", LeggedRobot, A1RoughCfg(), A1RoughCfgPPO() )
task_registry.register( "cassie", Cassie, CassieRoughCfg(), CassieRoughCfgPPO() )
task_registry.register( "lite3", LeggedRobot, Lite3RoughCfg(), Lite3RoughCfgPPO() )
task_registry.register( "lite3amp", Lite3AMP, Lite3AMPCfg(), Lite3AMPCfgPPO() )
task_registry.register( "lite3skill", Lite3Skill, Lite3SkillCfg(), Lite3SkillCfgPPO() )
task_registry.register( "lite3handstand", Lite3Skill, Lite3HandStandCfg(), Lite3SkillCfgPPO() )
task_registry.register( "lite3footstand", Lite3Skill, Lite3FootStandCfg(), Lite3SkillCfgPPO() )
task_registry.register( "lite3parkour", Lite3Parkour, Lite3ParkourCfg(), Lite3ParkourCfgPPO() )
task_registry.register( "lite3dreamwaq", Lite3DreamWaQ, Lite3DreamWaQCfg(), Lite3DreamWaQCfgPPO() )
task_registry.register( "lite3pie", Lite3PIE, Lite3PIECfg(), Lite3PIECfgPPO() )
task_registry.register( "lite3swc", Lite3SWC, Lite3SWCCfg(), Lite3SWCCfgPPO() )

task_registry.register( "eqr", Eqr, EqrRoughCfg(), EqrRoughCfgPPO() )
task_registry.register( "eqrhandstand", EQRSkill, EqrHandStandCfg(), EqrSkillCfgPPO() )
task_registry.register( "eqrfootstand", EQRSkill, EqrFootStandCfg(), EqrSkillCfgPPO() )
task_registry.register( "eqramp", EqrAMP, EqrAMPCfg(), EqrAMPCfgPPO() )