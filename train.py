from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from data import FRAME_COUNT, FRAME_SIZE, build_dataset
from model import DiffusionSchedule, TinyDiffusionModel, count_parameters, encode_video


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a tiny conditional video diffusion model.")
    parser.add_argument("--steps", type=int, default=2000, help="Number of gradient steps.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--num-samples", type=int, default=4096, help="Procedurally generated training samples.")
    parser.add_argument("--digits", type=int, nargs="*", default=list(range(10)), help="Digits to include in training.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--hidden-dim", type=int, default=64, help="Hidden width of the tiny MLP denoiser.")
    parser.add_argument("--latent-size", type=int, default=8, help="Spatial latent resolution per frame.")
    parser.add_argument("--diffusion-steps", type=int, default=50, help="Number of diffusion steps.")
    parser.add_argument("--save-path", type=Path, default=Path("tinyvid.pt"), help="Checkpoint output path.")
    parser.add_argument("--device", type=str, default="cpu", help="Device string, for example cpu or cuda.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--log-every", type=int, default=100, help="Print every N training steps.")
    return parser.parse_args()


def sample_batch(dataset, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    indices = torch.randint(0, len(dataset), (batch_size,))
    digits = []
    videos = []

    for index in indices.tolist():
        digit, video = dataset[index]
        digits.append(digit)
        videos.append(video)

    return torch.stack(digits, dim=0), torch.stack(videos, dim=0)


def main() -> None:
    args = parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    dataset = build_dataset(digits=args.digits, num_samples=args.num_samples)

    model = TinyDiffusionModel(
        frame_count=FRAME_COUNT,
        latent_size=args.latent_size,
        hidden_dim=args.hidden_dim,
    ).to(device)
    schedule = DiffusionSchedule(num_steps=args.diffusion_steps, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(f"training on {device} with {count_parameters(model):,} trainable parameters")

    model.train()
    for step in range(1, args.steps + 1):
        digits, videos = sample_batch(dataset, args.batch_size)
        digits = digits.to(device)
        videos = videos.to(device)
        latents = encode_video(videos, latent_size=args.latent_size)

        timesteps = torch.randint(0, schedule.num_steps, (digits.shape[0],), device=device)
        noise = torch.randn_like(latents)
        noisy_latents = schedule.q_sample(latents, noise, timesteps)
        predicted_noise = model(noisy_latents, digits, timesteps, schedule.num_steps)

        loss = F.mse_loss(predicted_noise, noise)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step == 1 or step % args.log_every == 0 or step == args.steps:
            print(f"step {step:04d}/{args.steps}  loss={loss.item():.6f}")

    checkpoint = {
        "model_state": model.state_dict(),
        "digits": list(args.digits),
        "frame_count": FRAME_COUNT,
        "frame_size": FRAME_SIZE,
        "latent_size": args.latent_size,
        "hidden_dim": args.hidden_dim,
        "diffusion_steps": args.diffusion_steps,
    }
    args.save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.save_path)
    print(f"saved checkpoint to {args.save_path}")


if __name__ == "__main__":
    main()
