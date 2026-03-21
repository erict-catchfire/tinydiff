from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn


def hw_linear(input_tensor: torch.Tensor, weights_tensor: torch.Tensor, output_size: int) -> torch.Tensor:
    if input_tensor.ndim == 1:
        input_tensor = input_tensor.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False

    if input_tensor.ndim != 2:
        raise ValueError("input_tensor must be 1D or 2D.")
    if weights_tensor.ndim != 2:
        raise ValueError("weights_tensor must be 2D.")
    if input_tensor.shape[1] != weights_tensor.shape[1]:
        raise ValueError("input feature size must match weights in_features.")
    if output_size < 0 or output_size > weights_tensor.shape[0]:
        raise ValueError("output_size must be between 0 and weights out_features.")

    output = torch.zeros(
        (input_tensor.shape[0], output_size),
        device=input_tensor.device,
        dtype=input_tensor.dtype,
    )
    for out_idx in range(output_size):
        for in_idx in range(input_tensor.shape[1]):
            output[:, out_idx] += input_tensor[:, in_idx] * weights_tensor[out_idx, in_idx]

    if squeeze_output:
        return output[0]
    return output


def hw_relu(input_tensor: torch.Tensor) -> torch.Tensor:
    output = torch.zeros_like(input_tensor)

    if input_tensor.ndim == 1:
        for idx in range(input_tensor.shape[0]):
            value = input_tensor[idx]
            output[idx] = value if value > 0 else 0
    elif input_tensor.ndim == 2:
        for row in range(input_tensor.shape[0]):
            for col in range(input_tensor.shape[1]):
                value = input_tensor[row, col]
                output[row, col] = value if value > 0 else 0
    else:
        raise ValueError("input_tensor must be 1D or 2D.")

    return output


def precompute_time_features_cpu(timesteps: torch.Tensor, num_steps: int) -> torch.Tensor:
    timesteps_cpu = timesteps.to("cpu", dtype=torch.float32)
    t = timesteps_cpu / max(num_steps - 1, 1)
    return torch.stack(
        [
            t,
            t * t,
            torch.sin(math.pi * t),
            torch.cos(math.pi * t),
        ],
        dim=-1,
    )


class TinyDiffusionModelHardware(nn.Module):
    def __init__(
        self,
        frame_count: int = 20,
        latent_size: int = 8,
        hidden_dim: int = 64,
        cond_dim: int = 16,
        time_dim: int = 16,
    ) -> None:
        super().__init__()
        self.frame_count = frame_count
        self.latent_size = latent_size
        self.latent_dim = frame_count * latent_size * latent_size
        self.work_profile_path = Path("outputs/work_profile.txt")
        self._profile_pass_index = 0
        self.work_profile_path.parent.mkdir(parents=True, exist_ok=True)
        self.work_profile_path.write_text("", encoding="utf-8")

        self.digit_embedding = nn.Embedding(3, cond_dim)
        self.time_proj = nn.Sequential(
            nn.Linear(4, time_dim),
            nn.ReLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.net = nn.Sequential(
            nn.Linear(self.latent_dim + cond_dim + time_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.latent_dim),
        )

    def _next_pass_id(self) -> int:
        self._profile_pass_index += 1
        return self._profile_pass_index

    def _profile_write(self, pass_id: int, section: str, message: str) -> None:
        lines = message.splitlines() or [""]
        with self.work_profile_path.open("a", encoding="utf-8") as profile_file:
            for line in lines:
                profile_file.write(f"pass={pass_id} section={section} {line}\n")

    def _tensor_histogram_256(
        self, tensor: torch.Tensor, fixed_min: float, fixed_max: float
    ) -> tuple[list[int], float, float]:
        flat = tensor.detach().cpu().reshape(-1).float()
        if flat.numel() == 0:
            return [0] * 256, fixed_min, fixed_max

        clipped = flat.clamp(min=fixed_min, max=fixed_max)
        counts = torch.histc(clipped, bins=256, min=fixed_min, max=fixed_max)
        histogram = [int(value.item()) for value in counts]
        return histogram, fixed_min, fixed_max

    def _profile_linear(
        self,
        pass_id: int,
        section: str,
        name: str,
        input_tensor: torch.Tensor,
        output_tensor: torch.Tensor,
        weights: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> None:
        weight_histogram, w_min, w_max = self._tensor_histogram_256(weights, fixed_min=-3.0, fixed_max=3.0)
        if bias is None:
            bias_histogram = [0] * 256
            b_min = -1.0
            b_max = 1.0
        else:
            bias_histogram, b_min, b_max = self._tensor_histogram_256(bias, fixed_min=-1.0, fixed_max=1.0)
        self._profile_write(
            pass_id,
            section,
            (
                f"{name} linear input_size={tuple(input_tensor.shape)} output_size={tuple(output_tensor.shape)} "
                f"weights_hist_256={weight_histogram} weights_min={w_min} weights_max={w_max} "
                f"bias_hist_256={bias_histogram} bias_min={b_min} bias_max={b_max}"
            ),
        )

    def _profile_relu(self, pass_id: int, section: str, name: str, relu_tensor: torch.Tensor) -> None:
        self._profile_write(pass_id, section, f"{name} relu size={tuple(relu_tensor.shape)}")

    def _time_features_hw(self, precomputed_features: torch.Tensor, device: torch.device, pass_id: int) -> torch.Tensor:
        features = precomputed_features.to(device=device, dtype=self.time_proj[0].weight.dtype)
        layer0 = self.time_proj[0]
        layer2 = self.time_proj[2]

        z0 = hw_linear(features, layer0.weight, layer0.out_features)
        z0 = z0 + layer0.bias
        self._profile_linear(pass_id, "time_features_hw", "layer0", features, z0, layer0.weight, layer0.bias)
        a0 = hw_relu(z0)
        self._profile_relu(pass_id, "time_features_hw", "layer1", a0)
        z1 = hw_linear(a0, layer2.weight, layer2.out_features)
        z1 = z1 + layer2.bias
        self._profile_linear(pass_id, "time_features_hw", "layer2", a0, z1, layer2.weight, layer2.bias)
        return z1

    def _digit_embedding_hw(self, digits: torch.Tensor) -> torch.Tensor:
        # Hardware-style table lookup: read rows directly from embedding memory.
        table = self.digit_embedding.weight
        print(table)
        indices = digits.to(device=table.device, dtype=torch.long)
        return table.index_select(0, indices)

    def _net_hw(self, net_input: torch.Tensor, pass_id: int) -> torch.Tensor:
        layer0 = self.net[0]
        layer2 = self.net[2]
        layer4 = self.net[4]

        z0 = hw_linear(net_input, layer0.weight, layer0.out_features)
        z0 = z0 + layer0.bias
        self._profile_linear(pass_id, "net_hw", "layer0", net_input, z0, layer0.weight, layer0.bias)
        a0 = hw_relu(z0)
        self._profile_relu(pass_id, "net_hw", "layer1", a0)

        z1 = hw_linear(a0, layer2.weight, layer2.out_features)
        z1 = z1 + layer2.bias
        self._profile_linear(pass_id, "net_hw", "layer2", a0, z1, layer2.weight, layer2.bias)
        a1 = hw_relu(z1)
        self._profile_relu(pass_id, "net_hw", "layer3", a1)

        z2 = hw_linear(a1, layer4.weight, layer4.out_features)
        z2 = z2 + layer4.bias
        self._profile_linear(pass_id, "net_hw", "layer4", a1, z2, layer4.weight, layer4.bias)
        return z2

    def _forward_hw(
        self,
        noisy_latent: torch.Tensor,
        digits: torch.Tensor,
        precomputed_time_features: torch.Tensor,
    ) -> torch.Tensor:
        pass_id = self._next_pass_id()
        batch_size = noisy_latent.shape[0]
        x = noisy_latent.reshape(batch_size, -1)
        # Either in HW or in SW. Depends on the final size of this
        cond = self._digit_embedding_hw(digits)
        t_embed = self._time_features_hw(precomputed_time_features, noisy_latent.device, pass_id)
        #Possible future optimization to start work before t_embed is coompleted. 
        output = self._net_hw(torch.cat([x, cond, t_embed], dim=-1), pass_id)
        return output.view(batch_size, self.frame_count, self.latent_size, self.latent_size)

    def forward(self, noisy_latent: torch.Tensor, digits: torch.Tensor, timesteps: torch.Tensor, num_steps: int) -> torch.Tensor:
        precomputed_time_features = precompute_time_features_cpu(timesteps, num_steps)
        return self._forward_hw(noisy_latent, digits, precomputed_time_features)
