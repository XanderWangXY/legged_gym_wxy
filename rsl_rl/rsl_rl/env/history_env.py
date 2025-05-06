import gym
import torch

class HistoryWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env

        self.obs_history_length = self.env.cfg.env.num_observation_history
        self.num_obs = self.env.num_obs
        self.num_envs = self.env.num_envs
        self.num_obs_history = self.obs_history_length * self.num_obs
        self.num_privileged_obs = self.num_privileged_obs
        self.device = self.env.device

        self.obs_history_buffer = torch.zeros(
            self.obs_history_length, self.num_envs, self.num_obs,
            dtype=torch.float, device=self.device
        )
        self.obs_history_ptr = 0  # 环形指针初始化

    def step(self, action):
        # 原始 step 分支
        if 'amp' in self.env.task_name:
            obs, privileged_obs, rew, dones, infos, reset_env_ids, terminal_amp_states = self.env.step(action)
        elif 'parkour' in self.env.task_name:
            obs, privileged_obs, rew, dones, infos, depth_buffer = self.env.step(action)
        else:
            obs, privileged_obs, rew, dones, infos = self.env.step(action)

        # 清零 reset env 的历史记录
        self.obs_history_buffer[:, dones] = 0.0

        # 写入当前 obs 到滑动窗口
        self.obs_history_buffer[self.obs_history_ptr] = obs
        self.obs_history_ptr = (self.obs_history_ptr + 1) % self.obs_history_length

        # 构造 obs_history: [T, B, D] -> [B, T*D]
        idxs = [(self.obs_history_ptr + i) % self.obs_history_length for i in range(self.obs_history_length)]
        obs_history = self.obs_history_buffer[idxs]         # [T, B, D]
        obs_history = obs_history.transpose(0, 1).reshape(self.num_envs, -1)  # [B, T*D]

        output = {
            'obs': obs,
            'privileged_obs': privileged_obs,
            'obs_history': obs_history
        }

        if 'parkour' in self.env.task_name:
            output['depth_buffer'] = depth_buffer
        if 'amp' in self.env.task_name:
            return output, rew, dones, infos, reset_env_ids, terminal_amp_states
        else:
            return output, rew, dones, infos

    def get_observations(self):
        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()

        # 不重置历史，仅追加（用于收集初始步）
        self.obs_history_buffer[self.obs_history_ptr] = obs
        self.obs_history_ptr = (self.obs_history_ptr + 1) % self.obs_history_length

        idxs = [(self.obs_history_ptr + i) % self.obs_history_length for i in range(self.obs_history_length)]
        obs_history = self.obs_history_buffer[idxs]  # [T, B, D]
        obs_history = obs_history.transpose(0, 1).reshape(self.num_envs, -1)

        return {'obs': obs, 'privileged_obs': privileged_obs, 'obs_history': obs_history}

    def reset_idx(self, env_ids):
        ret = super().reset_idx(env_ids)
        self.obs_history_buffer[:, env_ids] = 0.0
        return ret

    def reset(self):
        ret = super().reset()
        privileged_obs = self.env.get_privileged_observations()
        self.obs_history_buffer[:] = 0.0
        self.obs_history_ptr = 0
        return {
            "obs": ret,
            "privileged_obs": privileged_obs,
            "obs_history": torch.zeros(
                self.num_envs, self.obs_history_length * self.num_obs,
                dtype=torch.float, device=self.device
            )
        }
