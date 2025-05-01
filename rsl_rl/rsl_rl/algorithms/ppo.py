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

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from copy import deepcopy

from rsl_rl.modules import ActorCritic
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import unpad_trajectories

def split_tensor_along_dim1(tensor, num_splits):
    """ 将 tensor 按 dim=1 均匀分成 num_splits 块，不能整除时最后一块大一点 """
    total_len = tensor.shape[1]
    chunk_size = total_len // num_splits
    remainder = total_len % num_splits

    split_sizes = [chunk_size] * num_splits
    for i in range(remainder):
        split_sizes[i] += 1

    return torch.split(tensor, split_sizes, dim=1)


class RNNAdaptationModule(nn.Module):
    def __init__(self, obs_dim, history_steps, latent_dim, hidden_size=128, num_layers=2, activation=nn.Tanh()):
        super(RNNAdaptationModule, self).__init__()

        self.obs_dim = obs_dim
        self.history_steps = history_steps
        self.latent_dim = latent_dim

        # RNN处理时序数据
        self.rnn = nn.GRU(
            input_size=obs_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        # 从RNN输出映射到潜变量
        self.latent_projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            activation,
            nn.Linear(hidden_size, latent_dim)
        )

class PPO:
    actor_critic: ActorCritic
    def __init__(self,
                 actor_critic,
                 depth_encoder = None,
                 depth_encoder_paras= None,
                 num_learning_epochs=1,
                 num_mini_batches=1,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 value_loss_coef=1.0,
                 entropy_coef=0.0,
                 learning_rate=1e-3,
                 max_grad_norm=1.0,
                 use_clipped_value_loss=True,
                 schedule="fixed",
                 desired_kl=0.01,
                 num_adaptation_module_substeps=1,
                 device='cpu',
                 student = False,
                 dagger_beta=0.9,
                 ):

        self.device = device

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None # initialized later
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        self.adaptation_optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        self.transition = RolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.num_adaptation_module_substeps = num_adaptation_module_substeps
        self.if_depth = depth_encoder != None
        if self.if_depth:
            self.depth_encoder = depth_encoder
            self.depth_encoder_optimizer = optim.Adam(self.depth_encoder.parameters(),lr=depth_encoder_paras["learning_rate"])
        if student:
            self.student_adaptation = deepcopy(actor_critic.adaptation_module)
            self.student_adaptation_optimizer = optim.Adam(self.student_adaptation.parameters())
            self.student_actor = deepcopy(actor_critic.actor)
            self.student_actor_optimizer = optim.Adam([*self.student_actor.parameters(), *self.student_adaptation.parameters()])
            if self.if_depth:
                self.depth_encoder = depth_encoder
                self.depth_encoder_optimizer = optim.Adam(self.depth_encoder.parameters(),lr=depth_encoder_paras["learning_rate"])
                self.depth_encoder_paras = depth_encoder_paras
                self.student_actor_optimizer = optim.Adam(
                    [*self.student_actor.parameters(), *self.student_adaptation.parameters(), *self.depth_encoder.parameters()], lr=depth_encoder_paras["learning_rate"])

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, privileged_obs_shape, obs_history_shape, action_shape, depth_image_shape=None):
        self.storage = RolloutStorage(num_envs, num_transitions_per_env, actor_obs_shape, privileged_obs_shape, obs_history_shape, action_shape, depth_image_shape, self.device)

    def test_mode(self):
        self.actor_critic.test()
    
    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, critic_obs, privileged_obs, obs_history, depth_image = None):
        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()
        # Compute the actions and values
        self.transition.actions = self.actor_critic.act(obs, privileged_obs).detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs, privileged_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        self.transition.privileged_observations = privileged_obs
        self.transition.observation_histories = obs_history
        if self.if_depth:
            self.transition.depth_image = depth_image
            self.depth_encoder(depth_image, obs)
            self.transition.hidden_states = tuple([self.depth_encoder.get_hidden_states()])
        return self.transition.actions
    
    def process_env_step(self, rewards, dones, infos):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        # Bootstrapping on time outs
        if 'time_outs' in infos:
            self.transition.rewards += self.gamma * torch.squeeze(self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)
    
    def compute_returns(self, last_critic_obs, last_critic_privileged_obs):
        last_values = self.actor_critic.evaluate(last_critic_obs, last_critic_privileged_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_adaptation_loss = 0
        mean_depth_loss = 0
        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        elif self.if_depth:
            generator = self.storage.depth_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for (obs_batch, critic_obs_batch, privileged_obs_batch, obs_history_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
            old_mu_batch, old_sigma_batch
             ,depth_image_batch,hidden_states_batch, depth_masks_batch,obs_recurrent_batch, priv_obs_recurrent_batch
             , hid_states_batch, masks_batch) in generator:


                self.actor_critic.act(obs_batch, privileged_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
                actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
                value_batch = self.actor_critic.evaluate(critic_obs_batch, privileged_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
                mu_batch = self.actor_critic.action_mean
                sigma_batch = self.actor_critic.action_std
                entropy_batch = self.actor_critic.entropy

                # KL
                if self.desired_kl != None and self.schedule == 'adaptive':
                    with torch.inference_mode():
                        kl = torch.sum(
                            torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                        kl_mean = torch.mean(kl)

                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                        
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = self.learning_rate


                # Surrogate loss
                ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
                surrogate = -torch.squeeze(advantages_batch) * ratio
                surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                                1.0 + self.clip_param)
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                # Value function loss
                if self.use_clipped_value_loss:
                    value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
                                                                                                    self.clip_param)
                    value_losses = (value_batch - returns_batch).pow(2)
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (returns_batch - value_batch).pow(2).mean()

                loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()

                # Adaptation module gradient step
                for epoch in range(self.num_adaptation_module_substeps):
                    adaptation_pred = self.actor_critic.adaptation_module(obs_history_batch)
                    with torch.no_grad():
                        if self.actor_critic.terrain_hidden_dims is not None:
                            terrain_latent_target = self.actor_critic.terrain_encoder(privileged_obs_batch[:,:self.actor_critic.terrain_input_dims])
                            env_latent_target = self.actor_critic.env_factor_encoder(privileged_obs_batch[:,self.actor_critic.terrain_input_dims:])
                            adaptation_target = torch.cat((terrain_latent_target,env_latent_target),dim=-1)
                            if self.if_depth:
                                adaptation_target = env_latent_target
                            if self.actor_critic.parkour:
                                adaptation_target = env_latent_target
                        else:
                            adaptation_target = self.actor_critic.env_factor_encoder(privileged_obs_batch)
                        # residual = (adaptation_target - adaptation_pred).norm(dim=1)

                    adaptation_loss = F.mse_loss(adaptation_pred, adaptation_target)

                    self.adaptation_optimizer.zero_grad()
                    adaptation_loss.backward()
                    self.adaptation_optimizer.step()

                    mean_adaptation_loss += adaptation_loss.item()

                if self.if_depth:
                    with torch.no_grad():
                        terrain_latent_target = self.actor_critic.terrain_encoder(priv_obs_recurrent_batch[:,:,:self.actor_critic.terrain_input_dims])
                    terrain_latent_target = unpad_trajectories(terrain_latent_target, depth_masks_batch)
                    depth_latent_pred = self.depth_encoder.forward(depth_image_batch, obs_recurrent_batch, depth_masks_batch, hidden_states_batch)
                    depth_loss = F.mse_loss(depth_latent_pred, terrain_latent_target)

                    self.depth_encoder_optimizer.zero_grad()
                    depth_loss.backward()
                    self.depth_encoder_optimizer.step()

                    mean_depth_loss += depth_loss.item()



        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_adaptation_loss /= (num_updates * self.num_adaptation_module_substeps)
        self.storage.clear()

        return mean_value_loss, mean_surrogate_loss, mean_adaptation_loss, mean_depth_loss

    def behavioral_cloning(self,
                           actions_student_buffer,
                           actions_teacher_buffer,
                           env_latent_buffer=None,
                           adaptation_latent_buffer=None,
                           clip_param_supervise=0.2,
                           supervise_epochs=1,
                           supervise_mini_batches=1,
                           which=None
                           ):
        """supervise训练 student_actor 和 student_adaptation"""

        batch_size = len(actions_student_buffer)
        if batch_size == 0:
            return 0, 0  # 避免空数据训练

        mini_batch_size = max(1, batch_size // supervise_mini_batches)

        for _ in range(supervise_epochs):
            # 使用 torch.split() 生成小批次数据，返回的是元组

            # 直接遍历 batch，不需要切片索引
            # #for i in range(len(actions_student_buffer)):
            #     actions_student_mini = actions_student_buffer.clone()
            #     actions_teacher_mini = actions_teacher_buffer.clone()
            #     adaptation_latent_mini = adaptation_latent_buffer.clone()
            #     env_latent_mini = env_latent_buffer.clone()

            if 'actor' in which:
                # 训练 `student_actor`（模仿专家策略）
                # actor_loss = F.mse_loss(actions_student_mini, actions_teacher_mini)
                actor_loss = (actions_teacher_buffer.detach() - actions_student_buffer).norm(p=2, dim=1).mean()
                adaptation_loss = (env_latent_buffer.detach() - adaptation_latent_buffer).norm(p=2, dim=1).mean()
                loss = actor_loss+adaptation_loss
                self.student_actor_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.student_actor.parameters(), clip_param_supervise)
                self.student_actor_optimizer.step()
            elif 'adaptation' in which:
                # 训练 `student_adaptation`（模仿环境适应模块）
                adaptation_loss = F.mse_loss(env_latent_buffer, adaptation_latent_buffer)
                actor_loss = torch.tensor(0., device=self.device)
                self.student_adaptation_optimizer.zero_grad()
                adaptation_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.student_adaptation.parameters(), clip_param_supervise)
                self.student_adaptation_optimizer.step()
            else:
                return torch.tensor(0., device=self.device), torch.tensor(0., device=self.device)

        return actor_loss.detach().item(), adaptation_loss.detach().item()

    def update_dagger(self,
                      which = None,
                      clip_param_supervise=0.2,
                      ):
        mean_actor_loss = 0
        mean_adaptation_loss = 0
        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for obs_batch, critic_obs_batch, privileged_obs_batch, obs_history_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
                old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch in generator:

                        # Student actor module gradient step
                        for epoch in range(self.num_adaptation_module_substeps):
                            latent_pred = self.student_adaptation(obs_history_batch)
                            actor_input = torch.cat((obs_batch, latent_pred), dim=-1)
                            actor_pred = self.student_actor(actor_input)
                            with torch.no_grad():
                                actor_target,latent_target = self.actor_critic.act_expert(obs_batch, privileged_obs_batch)
                                # residual = (adaptation_target - adaptation_pred).norm(dim=1)
                            if 'actor' in which:
                                actor_loss = F.mse_loss(actor_pred, actor_target)

                                self.student_actor_optimizer.zero_grad()
                                actor_loss.backward()
                                torch.nn.utils.clip_grad_norm_(self.student_actor.parameters(), clip_param_supervise)
                                self.student_actor_optimizer.step()

                                mean_actor_loss += actor_loss.item()
                            if 'adaptation' in which:
                                adaptation_loss = F.mse_loss(latent_pred, latent_target)

                                self.student_adaptation_optimizer.zero_grad()
                                adaptation_loss.backward()
                                torch.nn.utils.clip_grad_norm_(self.student_adaptation.parameters(), clip_param_supervise)
                                self.student_adaptation_optimizer.step()

                                mean_adaptation_loss += adaptation_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_actor_loss /= num_updates
        mean_adaptation_loss /= (num_updates * self.num_adaptation_module_substeps)
        self.storage.clear()

        return mean_actor_loss, mean_adaptation_loss

    def update_dagger_vision(self,
                            which = None,
                            clip_param_supervise=0.2,
                            ):
        mean_actor_loss = 0
        mean_adaptation_loss = 0
        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.depth_recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for (obs_batch, critic_obs_batch, privileged_obs_batch, obs_history_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
            old_mu_batch, old_sigma_batch
             ,depth_image_batch,hidden_states_batch
             , hid_states_batch, masks_batch) in generator:

                        # Student actor module gradient step
                        for epoch in range(self.num_adaptation_module_substeps):
                            latent_pred = self.student_adaptation(obs_history_batch)
                            depth_latent_pred = self.depth_encoder(depth_image_batch, obs_batch, masks_batch, hidden_states_batch)
                            actor_input = torch.cat((obs_batch.flatten(0, 1), depth_latent_pred.flatten(0, 1), latent_pred), dim=-1)
                            actor_pred = self.student_actor(actor_input)
                            with torch.no_grad():
                                actor_target,latent_target = self.actor_critic.act_expert(obs_batch.flatten(0, 1), privileged_obs_batch.flatten(0, 1))
                                # residual = (adaptation_target - adaptation_pred).norm(dim=1)
                            if 'actor' in which:
                                actor_loss = F.mse_loss(actor_pred, actor_target)

                                self.student_actor_optimizer.zero_grad()
                                actor_loss.backward()
                                torch.nn.utils.clip_grad_norm_(self.student_actor.parameters(), clip_param_supervise)
                                self.student_actor_optimizer.step()

                                mean_actor_loss += actor_loss.item()
                            if 'adaptation' in which:
                                adaptation_loss = F.mse_loss(latent_pred, latent_target)

                                self.student_adaptation_optimizer.zero_grad()
                                adaptation_loss.backward()
                                torch.nn.utils.clip_grad_norm_(self.student_adaptation.parameters(), clip_param_supervise)
                                self.student_adaptation_optimizer.step()

                                mean_adaptation_loss += adaptation_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_actor_loss /= num_updates
        mean_adaptation_loss /= (num_updates * self.num_adaptation_module_substeps)
        self.storage.clear()

        return mean_actor_loss, mean_adaptation_loss

    def get_student_inference_policy(self, device=None):
        self.student_adaptation.eval()
        self.student_actor.eval()
        return self.act_student_inference

    def act_student_inference(self, observations, observation_history, policy_info={}):
        latent = self.student_adaptation(observation_history)
        actions_mean = self.student_actor(torch.cat((observations, latent), dim=-1))
        policy_info["latents"] = latent.detach().cpu().numpy()
        return actions_mean, latent
