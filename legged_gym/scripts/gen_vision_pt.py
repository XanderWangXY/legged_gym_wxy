import os
import torch
import torch.nn as nn

# CNN backbone for 58x87 input
class DepthCNNBackbone(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ELU(),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ELU(),
            nn.Flatten(),
            nn.Linear(64 * 25 * 39, 128),
            nn.ELU(),
            nn.Linear(128, output_dim),
            nn.ELU()  # 和你训练时一致
        )

    def forward(self, depth_image):
        if depth_image.ndim == 3:
            depth_image = depth_image.unsqueeze(1)  # [B, 1, 58, 87]
        return self.cnn(depth_image)


class DepthRNNWrapper(nn.Module):
    def __init__(self, prop_dim, cnn_output_dim, rnn_hidden_dim, rnn_type='gru', rnn_num_layers=1):
        super().__init__()

        self.prop_dim = prop_dim
        self.cnn = DepthCNNBackbone(cnn_output_dim)

        self.combination_mlp = nn.Sequential(
            nn.Linear(cnn_output_dim + prop_dim, 128),
            nn.ELU(),
            nn.Linear(128, 32)
        )

        rnn_cls = nn.GRU if rnn_type == 'gru' else nn.LSTM
        self.rnn = rnn_cls(input_size=32, hidden_size=rnn_hidden_dim, num_layers=rnn_num_layers)

        self.output_mlp = nn.Sequential(
            nn.Linear(rnn_hidden_dim, cnn_output_dim + 2),
            nn.Tanh()
        )

        self.rnn_type = rnn_type
        self.num_layers = rnn_num_layers
        self.hidden_size = rnn_hidden_dim

        # 内部维护隐状态
        self.register_buffer("hidden_state", torch.zeros(self.num_layers, 1, self.hidden_size))
        if self.rnn_type == "lstm":
            self.register_buffer("cell_state", torch.zeros(self.num_layers, 1, self.hidden_size))

    def forward(self, depth_image, proprioception):
        cnn_feature = self.cnn(depth_image)
        fused = self.combination_mlp(torch.cat((cnn_feature, proprioception), dim=-1))
        rnn_input = fused.unsqueeze(0)  # [1, B, 32]
        rnn_output, h_n = self.rnn(rnn_input, self.hidden_state)
        self.hidden_state[:] = h_n

        latent = self.output_mlp(rnn_output.squeeze(0))
        return latent

    @torch.jit.export
    def reset(self):
        self.hidden_state[:] = 0.0

def convert_state_dict(loaded_dict):
    converted = {}
    cnn_mapping = [
        ("base_backbone.image_compression.0", "cnn.cnn.0"),
        ("base_backbone.image_compression.3", "cnn.cnn.3"),
        ("base_backbone.image_compression.6", "cnn.cnn.6"),
        ("base_backbone.image_compression.8", "cnn.cnn.8"),
    ]
    for old_prefix, new_prefix in cnn_mapping:
        for param in ["weight", "bias"]:
            converted[f"{new_prefix}.{param}"] = loaded_dict[f"{old_prefix}.{param}"]

    rnn_mapping = [
        ("memory.rnn.weight_ih_l0", "rnn.weight_ih_l0"),
        ("memory.rnn.weight_hh_l0", "rnn.weight_hh_l0"),
        ("memory.rnn.bias_ih_l0", "rnn.bias_ih_l0"),
        ("memory.rnn.bias_hh_l0", "rnn.bias_hh_l0"),
    ]
    for old_key, new_key in rnn_mapping:
        converted[new_key] = loaded_dict[old_key]

    for k, v in loaded_dict.items():
        if k.startswith("combination_mlp.") or k.startswith("output_mlp."):
            converted[k] = v
    return converted

if __name__ == "__main__":
    path = '/home/ehr/wxy/legged_gym_wxy/logs/parkour_lite3/Jun13_20-00-34_student_position_noise/'
    model_name = 'model_59500.pt'
    loaded_dict = torch.load(os.path.join(path, model_name), map_location='cpu')
    depth_encoder_state_dict = loaded_dict['depth_encoder_state_dict']
    converted_state_dict = convert_state_dict(depth_encoder_state_dict)

    prop_dim = 47
    cnn_output_dim = 36
    rnn_hidden_dim = 512
    rnn_type = 'gru'
    rnn_num_layers = 1

    model = DepthRNNWrapper(prop_dim, cnn_output_dim, rnn_hidden_dim, rnn_type, rnn_num_layers)
    model.load_state_dict(converted_state_dict, strict=False)
    model.eval()

    dummy_depth = torch.ones(1, 58, 87)
    dummy_prop = torch.ones(1, prop_dim)

    traced = torch.jit.script(model)
    export_dir = os.path.join(path, 'traced')
    os.makedirs(export_dir, exist_ok=True)
    traced.save(os.path.join(export_dir, 'vision_encoder_with_memory.pt'))
    print("✅ 成功导出 fully exportable vision encoder！")
