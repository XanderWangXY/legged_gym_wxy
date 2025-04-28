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

import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.modules import rnn


class RNNAdaptationModule(nn.Module):
    def __init__(self, obs_dim, latent_dim, hidden_size=256, num_layers=2):
        """
        RNN适应网络，用于从观察历史中提取环境特征

        Args:
            obs_dim: 观察空间维度
            latent_dim: 输出潜变量维度
            hidden_size: RNN隐藏层大小
            num_layers: RNN层数
        """
        super(RNNAdaptationModule, self).__init__()

        self.obs_dim = obs_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.latent_dim = latent_dim

        # RNN核心
        self.rnn = nn.GRU(
            input_size=obs_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        # 潜变量映射层
        self.latent_projection = nn.Sequential(
           # nn.Linear(hidden_size, hidden_size),
           # nn.Tanh(),
            nn.Linear(hidden_size, latent_dim)
        )

    def init_hidden(self, batch_size, device):
        """初始化隐藏状态"""
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)

    def forward(self, current_obs, hidden_state=None):
        """
        前向传播函数

        Args:
            current_obs: 当前观察值, shape [batch_size, obs_dim]
            hidden_state: 之前的隐藏状态, shape [num_layers, batch_size, hidden_size]
                         如果为None则初始化为零

        Returns:
            latent: 提取的潜变量, shape [batch_size, latent_dim]
            new_hidden: 更新后的隐藏状态, shape [num_layers, batch_size, hidden_size]
        """
        batch_size = current_obs.shape[0]
        device = current_obs.device

        # 确保输入形状正确
        if len(current_obs.shape) == 2:
            # 添加时间维度 [batch, obs_dim] -> [batch, 1, obs_dim]
            current_obs = current_obs.unsqueeze(1)

        # 如果没有提供隐藏状态，初始化为零
        if hidden_state is None:
            hidden_state = self.init_hidden(batch_size, device)

        # 通过RNN处理当前观察
        _, new_hidden = self.rnn(current_obs, hidden_state)

        # 从最后一层隐藏状态生成潜变量
        latent = self.latent_projection(new_hidden[-1])

        return latent, new_hidden

    def process_sequence(self, obs_sequence):
        """
        处理完整观察序列并返回最终潜变量（用于训练时）

        Args:
            obs_sequence: 观察序列, shape [batch_size, seq_len, obs_dim]

        Returns:
            latent: 提取的潜变量, shape [batch_size, latent_dim]
            hidden: 最终隐藏状态
        """
        batch_size = obs_sequence.shape[0]
        device = obs_sequence.device

        # 初始化隐藏状态
        hidden = self.init_hidden(batch_size, device)

        # 通过RNN处理序列
        _, hidden = self.rnn(obs_sequence, hidden)

        # 从最后一层隐藏状态生成潜变量
        latent = self.latent_projection(hidden[-1])

        return latent, hidden

class StateHistoryEncoder(nn.Module):
    def __init__(self, activation_fn, input_size, tsteps, output_size, tanh_encoder_output=False):
        # self.device = device
        super(StateHistoryEncoder, self).__init__()
        self.activation_fn = activation_fn
        self.tsteps = tsteps

        channel_size = 10
        # last_activation = nn.ELU()

        self.encoder = nn.Sequential(
                nn.Linear(input_size, 3 * channel_size), self.activation_fn,
                )

        if tsteps == 50:
            self.conv_layers = nn.Sequential(
                    nn.Conv1d(in_channels = 3 * channel_size, out_channels = 2 * channel_size, kernel_size = 8, stride = 4), self.activation_fn,
                    nn.Conv1d(in_channels = 2 * channel_size, out_channels = channel_size, kernel_size = 5, stride = 1), self.activation_fn,
                    nn.Conv1d(in_channels = channel_size, out_channels = channel_size, kernel_size = 5, stride = 1), self.activation_fn, nn.Flatten())
        elif tsteps == 10:
            self.conv_layers = nn.Sequential(
                nn.Conv1d(in_channels = 3 * channel_size, out_channels = 2 * channel_size, kernel_size = 4, stride = 2), self.activation_fn,
                nn.Conv1d(in_channels = 2 * channel_size, out_channels = channel_size, kernel_size = 2, stride = 1), self.activation_fn,
                nn.Flatten())
        elif tsteps == 20:
            self.conv_layers = nn.Sequential(
                nn.Conv1d(in_channels = 3 * channel_size, out_channels = 2 * channel_size, kernel_size = 6, stride = 2), self.activation_fn,
                nn.Conv1d(in_channels = 2 * channel_size, out_channels = channel_size, kernel_size = 4, stride = 2), self.activation_fn,
                nn.Flatten())
        else:
            raise(ValueError("tsteps must be 10, 20 or 50"))

        self.linear_output = nn.Sequential(
                nn.Linear(channel_size * 3, output_size)#, self.activation_fn
                )

    def forward(self, obs):
        # nd * T * n_proprio
        if obs.dim()==3:
            obs = obs.flatten(0, 1)
        nd = obs.shape[0]
        T = self.tsteps
        # print("obs device", obs.device)
        # print("encoder device", next(self.encoder.parameters()).device)
        projection = self.encoder(obs.reshape([nd * T, -1])) # do projection for n_proprio -> 32
        output = self.conv_layers(projection.reshape([nd, T, -1]).permute((0, 2, 1)))
        output = self.linear_output(output)
        return output

class ActorCritic(nn.Module):
    is_recurrent = False
    def __init__(self,  num_actor_obs,
                        num_critic_obs,
                        num_obs_history,
                        num_actions,
                        num_privileged_obs=231,
                        depth_encoder_cfg=None,
                        if_depth=None,
                        actor_hidden_dims=[256, 256, 256],
                        critic_hidden_dims=[256, 256, 256],
                        encoder_hidden_dims=[512, 256],
                        adaptation_hidden_dims=[256, 32],
                        terrain_hidden_dims=None,
                        activation='elu',
                        init_noise_std=1.0,
                        terrain_input_dims=187,
                        terrain_latent_dims=36,
                        encoder_latent_dims=36,
                        adaptation_rnn_hidden_size=256,  # 新增
                        adaptation_rnn_num_layers=1,  # 新增
                        parkour = False,
                        **kwargs):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(ActorCritic, self).__init__()

        activation = get_activation(activation)
        
        env_encoder_input = num_privileged_obs
        adaptation_output = encoder_latent_dims
        self.terrain_hidden_dims = terrain_hidden_dims
        self.terrain_latent_dims = terrain_latent_dims
        self.terrain_input_dims = terrain_input_dims
        self.parkour = parkour
        
        if terrain_hidden_dims is not None:
            terrain_encoder_layers = []
            terrain_encoder_layers.append(nn.Linear(terrain_input_dims, terrain_hidden_dims[0]))
            terrain_encoder_layers.append(activation)
            for l in range(len(terrain_hidden_dims)):
                if l == len(terrain_hidden_dims) - 1:
                    terrain_encoder_layers.append(nn.Linear(terrain_hidden_dims[l], terrain_latent_dims))
                else:
                    terrain_encoder_layers.append(nn.Linear(terrain_hidden_dims[l], terrain_hidden_dims[l + 1]))
                    terrain_encoder_layers.append(activation)
            self.terrain_encoder = nn.Sequential(*terrain_encoder_layers)
            self.add_module(f"encoder", self.terrain_encoder)

            env_encoder_input = num_privileged_obs - terrain_input_dims
            adaptation_output = encoder_latent_dims + terrain_latent_dims
        if if_depth is not None and if_depth:
            adaptation_output = encoder_latent_dims
        if self.parkour:
            adaptation_output = encoder_latent_dims
        # Env factor encoder
        env_encoder_layers = []
        env_encoder_layers.append(nn.Linear(env_encoder_input, encoder_hidden_dims[0]))
        env_encoder_layers.append(activation)
        for l in range(len(encoder_hidden_dims)):
            if l == len(encoder_hidden_dims) - 1:
                env_encoder_layers.append(nn.Linear(encoder_hidden_dims[l], encoder_latent_dims))
            else:
                env_encoder_layers.append(nn.Linear(encoder_hidden_dims[l], encoder_hidden_dims[l + 1]))
                env_encoder_layers.append(activation)
        self.env_factor_encoder = nn.Sequential(*env_encoder_layers)
        self.add_module(f"encoder", self.env_factor_encoder)

        # Adaptation module
        # adaptation_module_layers = []
        # adaptation_module_layers.append(nn.Linear(num_obs_history, adaptation_hidden_dims[0]))
        # adaptation_module_layers.append(activation)
        # for l in range(len(adaptation_hidden_dims)):
        #     if l == len(adaptation_hidden_dims) - 1:
        #         adaptation_module_layers.append(nn.Linear(adaptation_hidden_dims[l], encoder_latent_dims))
        #     else:
        #         adaptation_module_layers.append(nn.Linear(adaptation_hidden_dims[l], adaptation_hidden_dims[l + 1]))
        #         adaptation_module_layers.append(activation)
        # self.adaptation_module = nn.Sequential(*adaptation_module_layers)
        # self.add_module(f"adaptation_module", self.adaptation_module)

        # self.adaptation_module = RNNAdaptationModule(
        #     obs_dim=num_actor_obs,  # 单个观察的维度
        #     latent_dim=encoder_latent_dims,
        #     #hidden_size=adaptation_hidden_dims[0] if adaptation_hidden_dims else 128,
        #     num_layers=2,
        # )
        # self.add_module(f"adaptation_module", self.adaptation_module)

        self.adaptation_module = StateHistoryEncoder(
            activation_fn=activation,
            input_size=num_actor_obs,  # 输入维度为单个观察的维度
            tsteps=int(num_obs_history/num_actor_obs),  # 可以根据需要设置时间步长
            output_size=adaptation_output,  # 输出维度与encoder_latent_dims一致
            tanh_encoder_output=False
        )
        self.add_module(f"adaptation_module", self.adaptation_module)

        latent_dim = int(torch.tensor(encoder_latent_dims))
        if terrain_hidden_dims is not None:
            latent_dim = int(torch.tensor(encoder_latent_dims + terrain_latent_dims))

        mlp_input_dim_a = num_actor_obs + latent_dim
        mlp_input_dim_c = num_critic_obs + latent_dim

        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        print(f"Environment Factor Encoder: {self.env_factor_encoder}")
        print(f"Adaptation Module: {self.adaptation_module}")
        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")
        if terrain_hidden_dims is not None:
            print(f"Terrain Encoder: {self.terrain_encoder}")
        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False
        
        # seems that we get better performance without init
        # self.init_memory_weights(self.memory_a, 0.001, 0.)
        # self.init_memory_weights(self.memory_c, 0.001, 0.)

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]


    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError
    
    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations, privileged_observations):
        if self.terrain_hidden_dims is not None:
            terrain_latent = self.terrain_encoder(privileged_observations[:,:self.terrain_input_dims])
            env_latent = self.env_factor_encoder(privileged_observations[:,self.terrain_input_dims:])
            latent = torch.cat((terrain_latent, env_latent), dim=-1)
        else:
            latent = self.env_factor_encoder(privileged_observations)
        mean = self.actor(torch.cat((observations, latent), dim=-1))
        self.distribution = Normal(mean, mean * 0. + self.std)

    def act(self, observations, privileged_observations, **kwargs):
        self.update_distribution(observations, privileged_observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_expert(self, observations, privileged_observations, policy_info={}):
        if self.terrain_hidden_dims is not None:
            terrain_latent = self.terrain_encoder(privileged_observations[:,:self.terrain_input_dims])
            env_latent = self.env_factor_encoder(privileged_observations[:,self.terrain_input_dims:])
            latent = torch.cat((terrain_latent, env_latent), dim=-1)
        else:
            latent = self.env_factor_encoder(privileged_observations)
        actions_mean = self.actor(torch.cat((observations, latent), dim=-1))
        policy_info["latents"] = latent.detach().cpu().numpy()
        return actions_mean, latent

    def act_inference(self, observations, observation_history, privileged_observations=None, policy_info={}):
        if privileged_observations is not None:
            if self.terrain_hidden_dims is not None:
                terrain_latent = self.terrain_encoder(privileged_observations[:, :self.terrain_input_dims])
                env_latent = self.env_factor_encoder(privileged_observations[:, self.terrain_input_dims:])
                latent = torch.cat((terrain_latent, env_latent), dim=-1)
            else:
                latent = self.env_factor_encoder(privileged_observations)
            policy_info["gt_latents"] = latent.detach().cpu().numpy()

        latent = self.adaptation_module(observation_history)
        actions_mean = self.actor(torch.cat((observations, latent), dim=-1))
        policy_info["latents"] = latent.detach().cpu().numpy()
        return actions_mean, latent

    def act_student_inference(self, observations, observation_history, privileged_observations=None, policy_info={}):
        if privileged_observations is not None:
            if self.terrain_hidden_dims is not None:
                terrain_latent = self.terrain_encoder(privileged_observations[:, :self.terrain_input_dims])
                env_latent = self.env_factor_encoder(privileged_observations[:, self.terrain_input_dims:])
                latent = torch.cat((terrain_latent, env_latent), dim=-1)
            else:
                latent = self.env_factor_encoder(privileged_observations)
            policy_info["gt_latents"] = latent.detach().cpu().numpy()

        latent = self.student_adaptation(observation_history)
        actions_mean = self.student_actor(torch.cat((observations, latent), dim=-1))
        policy_info["latents"] = latent.detach().cpu().numpy()
        return actions_mean, latent

    def evaluate(self, critic_observations, privileged_observations, **kwargs):
        """
        Computes the value estimates of critic observations.

        Args:
            critic_observations (Tensor): The observations for the critic network.
            privileged_observations (Tensor): The privileged observations.

        Returns:
            Tensor: The value estimates.
        """
        if self.terrain_hidden_dims is not None:
            terrain_latent = self.terrain_encoder(privileged_observations[:, :self.terrain_input_dims])
            env_latent = self.env_factor_encoder(privileged_observations[:, self.terrain_input_dims:])
            latent = torch.cat((terrain_latent, env_latent), dim=-1)
        else:
            latent = self.env_factor_encoder(privileged_observations)
        value = self.critic(torch.cat((critic_observations, latent), dim=-1))
        return value

def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
