import torch
import torch.nn as nn
import sys
import torchvision
from rsl_rl.utils import unpad_trajectories

class RecurrentDepthBackbone(nn.Module):
    def __init__(self, base_backbone, env_cfg, depth_cfg) -> None:
        super().__init__()
        activation = nn.ELU()
        last_activation = nn.Tanh()
        self.base_backbone = base_backbone
        if env_cfg == None:
            self.combination_mlp = nn.Sequential(
                                    nn.Linear(36 + 45, 128),
                                    activation,
                                    nn.Linear(128, 32)
                                )
        else:
            self.combination_mlp = nn.Sequential(
                                        nn.Linear(36 + env_cfg.env.num_observations, 128),
                                        activation,
                                        nn.Linear(128, 32)
                                    )
        # self.rnn = nn.GRU(input_size=32, hidden_size=512, batch_first=True)
        self.memory = Memory(32, type=depth_cfg['rnn_type'], num_layers=depth_cfg['rnn_num_layers'], hidden_size=depth_cfg['hidden_dims'])
        self.output_mlp = nn.Sequential(
                                nn.Linear(depth_cfg['hidden_dims'], 36),
                                last_activation
                            )
        self.hidden_states = None

    def forward(self, depth_image, proprioception, masks=None, hidden_states=None):
        depth_features = self.base_backbone(depth_image)
        fused_input = self.combination_mlp(torch.cat((depth_features, proprioception), dim=-1))  # [B, 32]

        rnn_output = self.memory(fused_input, masks=masks, hidden_states=hidden_states)
        latent = self.output_mlp(rnn_output.squeeze(0) if masks is None else rnn_output)
        return latent

    def reset(self, dones=None):
        self.memory.reset(dones)

    def get_hidden_states(self):
        return self.memory.hidden_states

    def set_hidden_states(self, hidden_states):
        self.memory.hidden_states = hidden_states

    def detach_hidden_states(self):
        if self.memory.hidden_states is not None:
            if isinstance(self.memory.hidden_states, tuple):
                self.memory.hidden_states = tuple(h.detach().clone() for h in self.memory.hidden_states)
            else:
                self.memory.hidden_states = self.memory.hidden_states.detach().clone()

class Memory(torch.nn.Module):
    def __init__(self, input_size, type='lstm', num_layers=1, hidden_size=256):
        super().__init__()
        # RNN
        rnn_cls = nn.GRU if type.lower() == 'gru' else nn.LSTM
        self.rnn = rnn_cls(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        self.hidden_states = None

    def forward(self, input, masks=None, hidden_states=None):
        batch_mode = masks is not None
        if batch_mode:
            # batch mode (policy update): need saved hidden states
            if hidden_states is None:
                raise ValueError("Hidden states not passed to memory module during policy update")
            out, _ = self.rnn(input, hidden_states)
            #out = unpad_trajectories(out, masks)
        else:
            # inference mode (collection): use hidden states of last step
            out, self.hidden_states = self.rnn(input.unsqueeze(0), self.hidden_states)
        return out

    def reset(self, dones=None):
        # When the RNN is an LSTM, self.hidden_states_a is a list with hidden_state and cell_state
        for hidden_state in self.hidden_states:
            hidden_state[..., dones, :] = 0.0
    
class DepthOnlyFCBackbone58x87(nn.Module):
    def __init__(self, prop_dim, scandots_output_dim, hidden_state_dim, output_activation=None, num_frames=1):
        super().__init__()

        self.num_frames = num_frames
        activation = nn.ELU()
        self.image_compression = nn.Sequential(
            # [1, 58, 87]
            nn.Conv2d(in_channels=self.num_frames, out_channels=32, kernel_size=5),
            # [32, 54, 83]
            nn.MaxPool2d(kernel_size=2, stride=2),
            # [32, 27, 41]
            activation,
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            activation,
            nn.Flatten(),
            # [32, 25, 39]
            nn.Linear(64 * 25 * 39, 128),
            activation,
            nn.Linear(128, scandots_output_dim)
        )

        if output_activation == "tanh":
            self.output_activation = nn.Tanh()
        else:
            self.output_activation = activation

    def forward(self, images: torch.Tensor):
        input_shape = images.shape  # 记录原始输入形状

        # 处理单批次 [B, H, W] → [B, 1, H, W]
        if images.dim() == 3:
            images = images.unsqueeze(1)  # [B, 1, H, W]
            is_multi_batch = False
        # 处理多批次 [T, B, H, W] → [T*B, 1, H, W]
        elif images.dim() == 4:
            T, B, H, W = images.shape
            images = images.reshape(T * B, 1, H, W)  # [T*B, 1, H, W]
            is_multi_batch = True
        else:
            raise ValueError(f"Expected input shape [B, H, W] or [T, B, H, W], got {images.shape}")

        # 正常进行卷积计算
        images_compressed = self.image_compression(images)  # [T*B, 36] 或 [B, 36]

        # 如果是多批次输入，恢复形状 [T*B, 36] → [T, B, 36]
        if is_multi_batch:
            T, B = input_shape[0], input_shape[1]  # 直接取原始输入的 T 和 B
            images_compressed = images_compressed.reshape(T, B, -1)  # [T, B, 36]

        latent = self.output_activation(images_compressed)
        return latent