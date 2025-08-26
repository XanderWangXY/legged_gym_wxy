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
from typing import Optional

import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.modules import rnn

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

class ActorCritic_DVAE(nn.Module):
    is_recurrent = False
    def __init__(self,  num_actor_obs,
                        num_critic_obs,
                        num_obs_history,
                        num_actions,
                        num_privileged_obs=231,
                        actor_hidden_dims=[256, 256, 256],
                        critic_hidden_dims=[256, 256, 256],
                        vh_encoder_dims=[256, 128],
                        vh_decoder_dims=[128, 256],
                        #adaptation_hidden_dims=[256, 32],
                        terrain_hidden_dims=None,
                        activation='elu',
                        init_noise_std=1.0,
                        vh_in_dim=187+3,
                        oh_in_dim=225,
                        oh_out_dim=32,
                        terrain_input_dims=187,
                        terrain_latent_dims=36,
                        encoder_latent_dims=36,
                        adaptation_rnn_hidden_size=256,  # 新增
                        adaptation_rnn_num_layers=1,  # 新增
                        critic_in_dim=45+48+32,
                        **kwargs):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(ActorCritic_DVAE, self).__init__()

        activation = get_activation(activation)

        mlp_input_dim_a = num_actor_obs + oh_out_dim
        mlp_input_dim_c = critic_in_dim

        oh_in_dim = num_obs_history

        # oh_encoder
        class OHEncoder(nn.Module):
            def __init__(self, input_dim, num_obs_history, num_actor_obs, z_dim=oh_out_dim):
                super().__init__()
                self.gru = nn.GRU(input_size=input_dim, hidden_size=512, num_layers=1, batch_first=True)
                self.out_layer = nn.Linear(512, z_dim)
                self.num_obs_history = num_obs_history
                self.num_actor_obs = num_actor_obs

            def forward(self, input):
                # input_reshaped = input.reshape(len(input), self.num_obs_history, self.num_actor_obs)
                out = self.gru(input)[1]
                if len(out.shape) == 3:
                    out = out.squeeze(0)
                out = self.out_layer(out)
                return out

        self.oh_encoder = StateHistoryEncoder(activation, num_actor_obs, int(num_obs_history/num_actor_obs), oh_out_dim )

        encoder_layers = []
        encoder_layers.append(nn.Linear(vh_in_dim, vh_encoder_dims[0]))
        encoder_layers.append(activation)
        for l in range(len(vh_encoder_dims)):
            if l == len(vh_encoder_dims) - 1:
                encoder_layers.append(nn.Linear(vh_encoder_dims[l], oh_out_dim))
                # pass
            else:
                encoder_layers.append(nn.Linear(vh_encoder_dims[l], vh_encoder_dims[l + 1]))
                encoder_layers.append(activation)
        self.encoder = nn.Sequential(*encoder_layers)

        # self.encode_mean_latent = nn.Linear(vh_encoder_dims[-1], cenet_out_dim - 3)
        # self.encode_logvar_latent = nn.Linear(vh_encoder_dims[-1], cenet_out_dim - 3)
        # self.encode_mean_vel = nn.Linear(vh_encoder_dims[-1], 3)
        # self.encode_logvar_vel = nn.Linear(vh_encoder_dims[-1], 3)

        decoder_layers = []
        decoder_layers.append(nn.Linear(oh_out_dim, vh_decoder_dims[0]))
        decoder_layers.append(activation)
        for l in range(len(vh_decoder_dims)):
            if l == len(vh_decoder_dims) - 1:
                decoder_layers.append(nn.Linear(vh_decoder_dims[l], vh_in_dim))
            else:
                decoder_layers.append(nn.Linear(vh_decoder_dims[l], vh_decoder_dims[l + 1]))
                decoder_layers.append(activation)
        self.decoder = nn.Sequential(*decoder_layers)

        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for layer_index in range(len(actor_hidden_dims)):
            if layer_index == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], actor_hidden_dims[layer_index + 1]))
                actor_layers.append(activation)
        self.actor_ = nn.Sequential(*actor_layers)

        class Actor(nn.Module):
            def __init__(self, oh_encoder, vh_decoder, actor):
                super().__init__()
                self.oh_encoder = oh_encoder
                self.vh_decoder = vh_decoder
                self.actor = actor

            def forward(self, obs, obs_history, forward_vae: bool = False):
                z = self.oh_encoder(obs_history)

                vae_out: Optional[torch.Tensor] = None
                if forward_vae:
                    vae_out = self.vh_decoder(z)

                actor_in = torch.cat((obs, z), dim=-1)
                act = self.actor(actor_in)
                return act, z, vae_out

        self.actor = Actor(self.oh_encoder, self.decoder, self.actor_)

        # self.actor = nn.Sequential(*([self.oh_encoder] + actor_layers))

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for layer_index in range(len(critic_hidden_dims)):
            if layer_index == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], critic_hidden_dims[layer_index + 1]))
                critic_layers.append(activation)
        self.critic_ = nn.Sequential(*critic_layers)

        class Critic(nn.Module):
            def __init__(self, vh_encoder, vh_decoder, critic):
                super().__init__()
                self.vh_encoder = vh_encoder
                self.vh_decoder = vh_decoder
                self.critic = critic

            def forward(self, obs, forward_vae: bool = False):
                obs_vh = torch.cat((obs[:, 45:45+187], obs[:, -3:]), dim=-1)#terrain:187,base_vel:3
                z = self.vh_encoder(obs_vh)

                vae_out: Optional[torch.Tensor] = None
                if forward_vae:
                    vae_out = self.vh_decoder(z)

                critic_in = torch.cat((obs[:, :45], obs[:, 45+187:-3], z), dim=-1)
                value = self.critic(critic_in)
                return value, z, vae_out

        self.critic = Critic(self.encoder, self.decoder, self.critic_)

        print(f"Encoder: {self.encoder}")
        print(f"Decoder: {self.decoder}")
        print(f"OH_Encoder: {self.oh_encoder}")
        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")
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

    def update_distribution(self, obs, obs_history):
        # compute mean
        mean, z, vae_out = self.actor(obs, obs_history, forward_vae=True)
        # create distribution
        self.distribution = Normal(mean, mean * 0. + self.std)
        return mean, z, vae_out

    def act(self, obs, obs_history, **kwargs):
        _, z, vae_out = self.update_distribution(obs, obs_history)
        return self.distribution.sample(), z, vae_out
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, obs, obs_history):
        actions_mean, _, _ = self.actor(obs, obs_history)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        value, z, vae_out = self.critic(critic_observations, forward_vae=True)
        return value, z, vae_out

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
