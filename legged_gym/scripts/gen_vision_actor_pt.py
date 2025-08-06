import os
import torch
import torch.nn as nn
import copy
from rsl_rl.modules.actor_critic import ActorCritic, StateHistoryEncoder, get_activation

class CombinedActorExporter(nn.Module):
    def __init__(self, adaptation_module, actor_body):
        super().__init__()
        self.adaptation_module = copy.deepcopy(adaptation_module)
        self.actor_body = copy.deepcopy(actor_body)
        self.adaptation_module.cpu()
        self.actor_body.cpu()

    def forward(self, obs, depth_latent, obs_history):
        # obs: [B, 47]
        # depth_latent: [B, 36]
        # obs_history: [B, 10, 47]
        latent_state = self.adaptation_module(obs_history)
        full_input = torch.cat((obs, depth_latent, latent_state), dim=-1)
        actions_mean = self.actor_body(full_input)
        return actions_mean

    def export(self, save_path, model_name):
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(os.path.join(save_path, model_name))


if __name__ == '__main__':
    path = '/home/ehr/wxy/legged_gym_wxy/logs/parkour_lite3/Jun13_20-00-34_student_position_noise/'
    model_name = 'model_59500.pt'
    loaded_dict = torch.load(os.path.join(path, model_name), map_location='cpu')

    actor_critic = ActorCritic(
        num_actor_obs=47,
        num_critic_obs=47,
        num_obs_history=10*47,  # 注意：10帧历史，每帧47维
        if_depth=True,
        num_privileged_obs=183,
        num_actions=12,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        adaptation_hidden_dims=[256, 32],  # 实际上没用，因为你用的是StateHistoryEncoder
        terrain_hidden_dims=[512, 256, 128],
        terrain_input_dims=132,
        encoder_latent_dims=12,
        activation='elu'
    ).to('cpu')

    actor_critic.actor.load_state_dict(loaded_dict['student_actor_state_dict'])
    actor_critic.adaptation_module.load_state_dict(loaded_dict['student_adaptation_state_dict'])

    exporter = CombinedActorExporter(actor_critic.adaptation_module, actor_critic.actor)

    export_dir = os.path.join(path, 'traced')
    os.makedirs(export_dir, exist_ok=True)

    # trace示例
    dummy_obs = torch.ones(1, 47)
    dummy_depth_latent = torch.ones(1, 36)
    dummy_obs_history = torch.ones(1, 470)
    traced_model = torch.jit.script(exporter)
    traced_model.save(os.path.join(export_dir, 'actor_exported.pt'))

    print("✅ 成功导出actor模型")
