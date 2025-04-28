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

import time
import os
from collections import deque
import statistics
from copy import copy,deepcopy
from logging import critical

from torch.utils.tensorboard import SummaryWriter
import torch
import wandb
from rsl_rl.algorithms import PPO
from rsl_rl.modules import *
from rsl_rl.env import VecEnv
from datetime import datetime
import warnings

class OnPolicyRunner:

    def __init__(self,
                 env: VecEnv,
                 train_cfg,
                 log_dir=None,
                 device='cpu'):

        self.cfg=train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env
        self.all_cfg = train_cfg
        self.wandb_run_name = (
                train_cfg["runner"]["description"]
                + "_"
                + datetime.now().strftime("%b%d_%H-%M-%S")
                + "_"
                + train_cfg["runner"]["experiment_name"]
                + "_"
                + train_cfg["runner"]["run_name"]
        )
        # if self.env.num_privileged_obs is not None:
        #     num_critic_obs = self.env.num_privileged_obs
        # else:
        #     num_critic_obs = self.env.num_obs
        num_critic_obs = self.env.num_obs
        actor_critic_class = eval(self.cfg["policy_class_name"]) # ActorCritic
        if 'depth_encoder' in self.all_cfg and self.all_cfg['depth_encoder'] is not None:
            self.depth_encoder_cfg = train_cfg["depth_encoder"]
            self.if_depth = self.depth_encoder_cfg["if_depth"]
            actor_critic: ActorCritic = actor_critic_class( self.env.num_obs,
                                                            num_critic_obs,
                                                            self.env.num_obs_history,
                                                            self.env.num_actions,
                                                            self.env.num_privileged_obs,
                                                            self.depth_encoder_cfg,
                                                            self.if_depth,
                                                            **self.policy_cfg).to(self.device)
        else:
            self.if_depth = False
            actor_critic: ActorCritic = actor_critic_class(self.env.num_obs,
                                                           num_critic_obs,
                                                           self.env.num_obs_history,
                                                           self.env.num_actions,
                                                           self.env.num_privileged_obs,
                                                           **self.policy_cfg).to(self.device)

        if 'depth_encoder' in self.all_cfg and self.all_cfg['depth_encoder'] is not None:
            if self.if_depth:
                depth_backbone = DepthOnlyFCBackbone58x87(env.cfg.env.num_observations,
                                                          self.policy_cfg['terrain_latent_dims'],
                                                          self.depth_encoder_cfg["hidden_dims"],
                                                          )
                depth_encoder = RecurrentDepthBackbone(depth_backbone, env.cfg, self.depth_encoder_cfg).to(self.device)
                print(depth_encoder)
            else:
                depth_encoder = None
        else:
            self.depth_encoder_cfg = None
            depth_encoder = None

        alg_class = eval(self.cfg["algorithm_class_name"]) # PPO
        self.alg: PPO = alg_class(actor_critic, depth_encoder, self.depth_encoder_cfg, device=self.device, **self.alg_cfg)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # init storage and model
        if self.if_depth:
            self.alg.init_storage(self.env.num_envs, self.num_steps_per_env, [self.env.num_obs], [self.env.num_privileged_obs],[self.env.num_obs_history], [self.env.num_actions], list(self.depth_encoder_cfg['depth_shape']))
        else:
            self.alg.init_storage(self.env.num_envs, self.num_steps_per_env, [self.env.num_obs], [self.env.num_privileged_obs], [self.env.num_obs_history], [self.env.num_actions])
        #self.learn = self.learn if not train_cfg['algorithm']['student'] else self.learn_BC
        self.learn = self.learn if not train_cfg['algorithm']['student'] else self.dagger
        if 'depth_encoder' in self.all_cfg and self.all_cfg['depth_encoder'] is not None:
            if self.if_depth and train_cfg['algorithm']['student']:
                self.learn = self.dagger_vision
        # Log
        self.log_dir = log_dir+self.cfg['description']
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        if not train_cfg['no_wandb']:
            wandb.init(
                project=self.all_cfg["runner"]["experiment_name"],
                sync_tensorboard=True,
                name=self.wandb_run_name,
                config=self.all_cfg,
            )

        #_, _ = self.env.reset()
        _ = self.env.reset()
    
    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))
        obs_dict = self.env.get_observations()
        obs, privileged_obs, obs_history = obs_dict["obs"], obs_dict["privileged_obs"], obs_dict["obs_history"]
        obs, privileged_obs, obs_history = obs.to(self.device), privileged_obs.to(self.device), obs_history.to(self.device)
        if self.if_depth:
            depth_image = self.env.get_depth_image()
            depth_image.to(self.device)

        self.alg.actor_critic.train() # switch to train mode (for dropout for example)

        best_reward = 0
        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    critic_obs = obs
                    if self.if_depth:
                        actions = self.alg.act(obs, critic_obs, privileged_obs, obs_history, depth_image)
                    else:
                        actions = self.alg.act(obs, critic_obs, privileged_obs, obs_history)
                    obs_dict, rewards, dones, infos = self.env.step(actions)
                    obs, privileged_obs, obs_history = obs_dict["obs"], obs_dict["privileged_obs"], obs_dict["obs_history"]
                    obs, privileged_obs, obs_history, rewards, dones = obs.to(self.device), privileged_obs.to(self.device), obs_history.to(self.device), rewards.to(self.device), dones.to(self.device)
                    self.alg.process_env_step(rewards, dones, infos)
                    if self.if_depth:
                        depth_image = self.env.get_depth_image()
                        depth_image.to(self.device)
                    
                    if self.log_dir is not None:
                        # Book keeping
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(critic_obs,privileged_obs)
            
            mean_value_loss, mean_surrogate_loss, mean_adaptation_loss, mean_depth_loss = self.alg.update()
            stop = time.time()
            learn_time = stop - start
            if self.log_dir is not None:
                self.log(locals())
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
            if rewbuffer and statistics.mean(rewbuffer) > best_reward:
                best_reward = statistics.mean(rewbuffer)
                self.save(os.path.join(self.log_dir, 'model_best.pt'.format(it)))
            ep_infos.clear()
        
        self.current_learning_iteration += num_learning_iterations
        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar('Episode/' + key, value, locs['it'])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

        self.writer.add_scalar('Loss/value_function', locs['mean_value_loss'], locs['it'])
        self.writer.add_scalar('Loss/surrogate', locs['mean_surrogate_loss'], locs['it'])
        self.writer.add_scalar('Loss/mean_adaptation_loss', locs['mean_adaptation_loss'], locs['it'])
        self.writer.add_scalar('Loss/mean_depth_loss', locs['mean_depth_loss'], locs['it'])
        self.writer.add_scalar('Loss/learning_rate', self.alg.learning_rate, locs['it'])
        self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), locs['it'])
        self.writer.add_scalar('Perf/total_fps', fps, locs['it'])
        self.writer.add_scalar('Perf/collection time', locs['collection_time'], locs['it'])
        self.writer.add_scalar('Perf/learning_time', locs['learn_time'], locs['it'])
        if len(locs['rewbuffer']) > 0:
            self.writer.add_scalar('Train/mean_reward', statistics.mean(locs['rewbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_reward/time', statistics.mean(locs['rewbuffer']), self.tot_time)
            self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)

        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n""")
        print(log_string)

    def log_student(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar('Episode/' + key, value, locs['it'])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

        self.writer.add_scalar('Loss/actor_loss', locs['actor_loss'], locs['it'])
        self.writer.add_scalar('Loss/adaptation_loss', locs['adaptation_loss'], locs['it'])
        self.writer.add_scalar('Loss/learning_rate', self.alg.learning_rate, locs['it'])
        self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), locs['it'])
        self.writer.add_scalar('Perf/total_fps', fps, locs['it'])
        self.writer.add_scalar('Perf/collection time', locs['collection_time'], locs['it'])
        self.writer.add_scalar('Perf/learning_time', locs['learn_time'], locs['it'])
        if len(locs['rewbuffer']) > 0:
            self.writer.add_scalar('Train/mean_reward', statistics.mean(locs['rewbuffer']), locs['it'])
            #self.writer.add_scalar('Train/mean_reward_inference', statistics.mean(locs['rewbuffer_inference']), locs['it'])
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_reward/time', statistics.mean(locs['rewbuffer']), self.tot_time)
            self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)

        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'actor_loss:':>{pad}} {locs['actor_loss']:.4f}\n"""
                          f"""{'adaptation_loss:':>{pad}} {locs['adaptation_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'actor_loss:':>{pad}} {locs['actor_loss']:.4f}\n"""
                          f"""{'adaptation_loss:':>{pad}} {locs['adaptation_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n""")
        print(log_string)

    def learn_BC(self, num_learning_iterations, init_at_random_ep_len=False):
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        tot_iter = self.current_learning_iteration + num_learning_iterations
        self.start_learning_iteration = copy(self.current_learning_iteration)

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        obs_dict = self.env.get_observations()
        obs, privileged_obs, obs_history = obs_dict["obs"], obs_dict["privileged_obs"], obs_dict["obs_history"]
        obs, privileged_obs, obs_history = obs.to(self.device), privileged_obs.to(self.device), obs_history.to(self.device)

        infos = {}

        self.alg.student_actor.load_state_dict(self.alg.actor_critic.actor.state_dict())
        self.alg.student_adaptation.load_state_dict(self.alg.actor_critic.adaptation_module.state_dict())

        self.alg.student_actor.train()
        self.alg.student_adaptation.train()

        num_pretrain_iter = 0
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            adaptation_latent_buffer = []
            env_latent_buffer = []
            actions_teacher_buffer = []
            actions_student_buffer = []
            for i in range(self.cfg['num_steps_per_env']):
                with torch.no_grad():
                    actions_teacher, latent_teacher = self.alg.actor_critic.act_expert(obs,privileged_obs)
                    actions_teacher_buffer.append(actions_teacher)
                    env_latent_buffer.append(latent_teacher)

                latent_student = self.alg.student_adaptation(obs_history)
                actions_student = self.alg.student_actor(torch.cat((obs, latent_student), dim=-1))

                actions_student_buffer.append(actions_student)
                adaptation_latent_buffer.append(latent_student)

                obs_dict, rewards, dones, infos = self.env.step(actions_student.detach())
                obs, privileged_obs, obs_history = obs_dict["obs"], obs_dict["privileged_obs"], obs_dict["obs_history"]
                obs, privileged_obs, obs_history, rewards, dones = obs.to(self.device), privileged_obs.to(self.device), obs_history.to(self.device), rewards.to(self.device), dones.to(self.device)

                if self.log_dir is not None:
                        # Book keeping
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

            stop = time.time()
            collection_time = stop - start
            start = stop

            actions_teacher_buffer = torch.cat(actions_teacher_buffer, dim=0)
            actions_student_buffer = torch.cat(actions_student_buffer, dim=0)
            env_latent_buffer = torch.cat(env_latent_buffer, dim=0)
            adaptation_latent_buffer = torch.cat(adaptation_latent_buffer, dim=0)

            which = 'actor'
            actor_loss, adaptation_loss = self.alg.behavioral_cloning(
                actions_student_buffer,
                actions_teacher_buffer,
                env_latent_buffer,
                adaptation_latent_buffer,
                which=which
            )

            stop = time.time()
            learn_time = stop - start

            if self.log_dir is not None:
                self.log_student(locals())
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
            ep_infos.clear()

        self.current_learning_iteration += num_learning_iterations
        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))

    def get_beta(self, current_iteration, total_iterations, start_rate=0.2, end_rate=0.6, initial_beta=1.0,
                 final_beta=0.6):
        # 计算当前训练进度的百分比
        progress = current_iteration / total_iterations

        # 前 20% 的训练过程，beta 保持为 initial_beta
        if progress < start_rate:
            return initial_beta
        # 60% 之后的训练过程，beta 保持为 final_beta
        elif progress > end_rate:
            return final_beta
        # 中间的 20% 到 60% 训练过程，beta 从 initial_beta 线性下降到 final_beta
        else:
            # 计算线性下降的比例
            decay_progress = (progress - start_rate) / (end_rate - start_rate)
            return initial_beta - (initial_beta - final_beta) * decay_progress

    def dagger(self, num_learning_iterations, init_at_random_ep_len=False,beta=0.9):
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf,
                                                             high=int(self.env.max_episode_length))
        obs_dict = self.env.get_observations()
        obs, privileged_obs, obs_history = obs_dict["obs"], obs_dict["privileged_obs"], obs_dict["obs_history"]
        obs, privileged_obs, obs_history = obs.to(self.device), privileged_obs.to(self.device), obs_history.to(
            self.device)
        #self.alg.actor_critic.train()  # switch to train mode (for dropout for example)

        best_reward = 0
        ep_infos = []
        rewbuffer = deque(maxlen=100)
        # rewbuffer_inference = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        # cur_reward_inference_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations

        # self.alg.student_actor.load_state_dict(self.alg.actor_critic.actor.state_dict())
        # self.alg.student_adaptation.load_state_dict(self.alg.actor_critic.adaptation_module.state_dict())

        self.alg.student_actor.train()
        self.alg.student_adaptation.train()

        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    critic_obs = obs

                    actions_teacher, latent_teacher = self.alg.actor_critic.act_expert(obs, privileged_obs)
                    latent_student = self.alg.student_adaptation(obs_history)
                    actions_student = self.alg.student_actor(torch.cat((obs, latent_student), dim=-1))

                    # 生成一个随机掩码，决定哪些样本使用专家策略
                    beta = self.get_beta(it, tot_iter)
                    expert_mask = torch.bernoulli(torch.full((obs.shape[0], 1), beta, device=obs.device))

                    # 使用掩码进行混合
                    actions_dagger = expert_mask * actions_teacher + (1 - expert_mask) * actions_student
                    latent_dagger = expert_mask * latent_teacher + (1 - expert_mask) * latent_student

                    actions = self.alg.act(obs, critic_obs, privileged_obs, obs_history)
                    # actions, inference_actions = self.alg.act(obs, critic_obs, privileged_obs, obs_history)
                    obs_dict, rewards, dones, infos = self.env.step(actions_student.detach())
                    obs, privileged_obs, obs_history = obs_dict["obs"], obs_dict["privileged_obs"], obs_dict["obs_history"]
                    obs, privileged_obs, obs_history, rewards, dones = obs.to(self.device), privileged_obs.to(
                        self.device), obs_history.to(self.device), rewards.to(self.device), dones.to(self.device)
                    self.alg.process_env_step(rewards, dones, infos)

                    if self.log_dir is not None:
                        # Book keeping
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        cur_reward_sum += rewards
                        # cur_reward_inference_sum += rewards_inference
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(critic_obs, privileged_obs)

            # if (it <30000):
            #     which = 'adaptation'
            # else:
            #     which = 'actor'
            which = 'actor'
            actor_loss, adaptation_loss = self.alg.update_dagger(which)
            stop = time.time()
            learn_time = stop - start

            if self.log_dir is not None:
                self.log_student(locals())
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
            if rewbuffer and statistics.mean(rewbuffer) > best_reward:
                best_reward = statistics.mean(rewbuffer)
                self.save(os.path.join(self.log_dir, 'model_best.pt'.format(it)))
            ep_infos.clear()

        self.current_learning_iteration += num_learning_iterations
        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))

    def dagger_vision(self, num_learning_iterations, init_at_random_ep_len=False, beta=0.9):
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf,
                                                             high=int(self.env.max_episode_length))
        obs_dict = self.env.get_observations()
        obs, privileged_obs, obs_history = obs_dict["obs"], obs_dict["privileged_obs"], obs_dict["obs_history"]
        obs, privileged_obs, obs_history = obs.to(self.device), privileged_obs.to(self.device), obs_history.to(
            self.device)
        depth_image = self.env.get_depth_image()
        depth_image.to(self.device)
        # self.alg.actor_critic.train()  # switch to train mode (for dropout for example)

        best_reward = 0
        ep_infos = []
        rewbuffer = deque(maxlen=100)
        # rewbuffer_inference = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        # cur_reward_inference_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations

        # self.alg.student_actor.load_state_dict(self.alg.actor_critic.actor.state_dict())
        # self.alg.student_adaptation.load_state_dict(self.alg.actor_critic.adaptation_module.state_dict())

        self.alg.student_actor.train()
        self.alg.student_adaptation.train()
        self.alg.depth_encoder.train()

        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    critic_obs = obs

                    actions_teacher, latent_teacher = self.alg.actor_critic.act_expert(obs, privileged_obs)
                    latent_student = self.alg.student_adaptation(obs_history)
                    depth_latent_student = self.alg.depth_encoder(depth_image, obs)
                    actions_student = self.alg.student_actor(torch.cat((obs, depth_latent_student, latent_student), dim=-1))

                    # # 生成一个随机掩码，决定哪些样本使用专家策略
                    # beta = self.get_beta(it, tot_iter)
                    # expert_mask = torch.bernoulli(torch.full((obs.shape[0], 1), beta, device=obs.device))
                    #
                    # # 使用掩码进行混合
                    # actions_dagger = expert_mask * actions_teacher + (1 - expert_mask) * actions_student
                    # latent_dagger = expert_mask * latent_teacher + (1 - expert_mask) * latent_student

                    actions = self.alg.act(obs, critic_obs, privileged_obs, obs_history, depth_image)
                    # actions, inference_actions = self.alg.act(obs, critic_obs, privileged_obs, obs_history)
                    obs_dict, rewards, dones, infos = self.env.step(actions_student.detach())
                    obs, privileged_obs, obs_history = obs_dict["obs"], obs_dict["privileged_obs"], obs_dict[
                        "obs_history"]
                    obs, privileged_obs, obs_history, rewards, dones = obs.to(self.device), privileged_obs.to(
                        self.device), obs_history.to(self.device), rewards.to(self.device), dones.to(self.device)
                    self.alg.process_env_step(rewards, dones, infos)
                    depth_image = self.env.get_depth_image()
                    depth_image.to(self.device)

                    if self.log_dir is not None:
                        # Book keeping
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        cur_reward_sum += rewards
                        # cur_reward_inference_sum += rewards_inference
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(critic_obs, privileged_obs)

            # if (it <30000):
            #     which = 'adaptation'
            # else:
            #     which = 'actor'
            which = 'actor'
            actor_loss, adaptation_loss = self.alg.update_dagger_vision(which)
            stop = time.time()
            learn_time = stop - start

            if self.log_dir is not None:
                self.log_student(locals())
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
            if rewbuffer and statistics.mean(rewbuffer) > best_reward:
                best_reward = statistics.mean(rewbuffer)
                self.save(os.path.join(self.log_dir, 'model_best.pt'.format(it)))
            ep_infos.clear()

        self.current_learning_iteration += num_learning_iterations
        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))

    def save(self, path, infos=None):
        state_dict = {
            'model_state_dict': self.alg.actor_critic.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'iter': self.current_learning_iteration,
            'infos': infos,
        }
        if self.alg_cfg['student']:
            state_dict['student_actor_state_dict'] = self.alg.student_actor.state_dict()
            state_dict['student_adaptation_state_dict'] = self.alg.student_adaptation.state_dict()
            if self.if_depth:
                state_dict['depth_encoder_state_dict'] = self.alg.depth_encoder.state_dict()
        else:
            state_dict['student_actor_state_dict'] = self.alg.actor_critic.actor.state_dict()
            state_dict['student_adaptation_state_dict'] = self.alg.actor_critic.adaptation_module.state_dict()
            if self.if_depth:
                state_dict['depth_encoder_state_dict'] = self.alg.depth_encoder.state_dict()
        torch.save(state_dict, path)

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path)
        self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'])
        if self.alg_cfg['student']:
            if 'student_adaptation_state_dict' not in loaded_dict:
                warnings.warn("'student_adaptation_state_dict' key does not exist, not loading student_adaptation...")
            else:
                print("Saved student_adaptation detected, loading...")
                self.alg.student_adaptation.load_state_dict(loaded_dict['student_adaptation_state_dict'])
            if 'student_actor_state_dict' in loaded_dict:
                print("Saved student_actor detected, loading...")
                self.alg.student_actor.load_state_dict(loaded_dict['student_actor_state_dict'])
            else:
                print("No saved student actor, Copying actor critic actor to student_actor...")
                self.alg.student_actor.load_state_dict(self.alg.actor_critic.actor.state_dict())
        if self.if_depth:
            if 'depth_encoder' not in loaded_dict:
                warnings.warn("'depth_encoder' key does not exist, not loading depth_encoder...")
            else:
                print("Saved depth_encoder detected, loading...")
                self.alg.depth_encoder.load_state_dict(loaded_dict['depth_encoder_state_dict'])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
        self.current_learning_iteration = loaded_dict['iter']
        return loaded_dict['infos']

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference

    def get_expert_policy(self, device=None):
        self.alg.actor_critic.eval()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_expert
