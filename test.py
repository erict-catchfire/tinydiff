from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import tempfile

import torch

from hardware_model import TinyDiffusionModelHardware, precompute_time_features_cpu
from model import DiffusionSchedule, TinyDiffusionModel, decode_video, sample_video


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample a video from a trained tiny diffusion checkpoint.")
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/tinyvid.pt"), help="Checkpoint saved by train.py.")
    parser.add_argument("--digit", type=int, default=2, help="Digit to animate.")
    parser.add_argument("--device", type=str, default="cpu", help="Device string, for example cpu or cuda.")
    parser.add_argument("--out", type=Path, default=Path("outputs/output.gif"), help="Output animation path.")
    parser.add_argument(
        "--out-hardware",
        type=Path,
        default=None,
        help="Optional hardware output animation path (used with --compare-hardware).",
    )
    parser.add_argument("--fps", type=int, default=10, help="Frames per second for the output animation.")
    parser.add_argument(
        "--compare-hardware",
        action="store_true",
        help="Compare software and hardware-model outputs for forward pass and full sampling.",
    )
    parser.add_argument("--rtol", type=float, default=1e-5, help="Relative tolerance for comparisons.")
    parser.add_argument("--atol", type=float, default=1e-6, help="Absolute tolerance for comparisons.")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed for deterministic comparisons.")
    return parser.parse_args()


def _to_uint8_frames(video: torch.Tensor) -> list[torch.Tensor]:
    video = ((video + 1.0) * 0.5).clamp(0.0, 1.0)
    video = (video[:, 0] * 255.0).round().to(torch.uint8).cpu()
    return [frame.contiguous() for frame in video]


def _frame_to_bytes(frame: torch.Tensor) -> bytes:
    return bytes(frame.view(-1).tolist())


def _save_with_pillow(frames: list[torch.Tensor], output_path: Path, fps: int) -> bool:
    try:
        from PIL import Image
    except ImportError:
        return False

    pil_frames = [Image.frombytes("L", (frame.shape[1], frame.shape[0]), _frame_to_bytes(frame)) for frame in frames]
    pil_frames[0].save(
        output_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=int(1000 / fps),
        loop=0,
    )
    return True


def _write_pgm(frame: torch.Tensor, output_path: Path) -> None:
    header = f"P5\n{frame.shape[1]} {frame.shape[0]}\n255\n".encode("ascii")
    output_path.write_bytes(header + _frame_to_bytes(frame))


def _save_with_ffmpeg(frames: list[torch.Tensor], output_path: Path, fps: int) -> bool:
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        return False

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        for index, frame in enumerate(frames):
            _write_pgm(frame, temp_dir_path / f"frame_{index:03d}.pgm")

        command = [
            ffmpeg_path,
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(temp_dir_path / "frame_%03d.pgm"),
            str(output_path),
        ]
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    return True


def save_animation(video: torch.Tensor, output_path: Path, fps: int) -> None:
    frames = _to_uint8_frames(video)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if _save_with_pillow(frames, output_path, fps):
        return

    if _save_with_ffmpeg(frames, output_path, fps):
        return

    raise RuntimeError(
        "Saving animations requires Pillow or ffmpeg. Install one of them and run test.py again."
    )


def _comparison_stats(
    software_output: torch.Tensor, hardware_output: torch.Tensor, rtol: float, atol: float
) -> tuple[bool, float, float]:
    diff = (software_output - hardware_output).abs()
    max_abs = float(diff.max().item())
    rel = diff / hardware_output.abs().clamp_min(1e-8)
    max_rel = float(rel.max().item())
    is_close = torch.allclose(software_output, hardware_output, rtol=rtol, atol=atol)
    return is_close, max_abs, max_rel


@torch.no_grad()
def sample_video_hardware(
    model: TinyDiffusionModelHardware,
    schedule: DiffusionSchedule,
    digit: int,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    model.eval()
    device = torch.device(device)
    latent = torch.randn(1, model.frame_count, model.latent_size, model.latent_size, device=device)
    digits = torch.tensor([digit], device=device, dtype=torch.long)

    for step_index in reversed(range(schedule.num_steps)):
        batch_size = latent.shape[0]
        timesteps = torch.full((batch_size,), step_index, device=device, dtype=torch.long)
        time_features_cpu = precompute_time_features_cpu(timesteps, schedule.num_steps)
        predicted_x0 = model._forward_hw(latent, digits, time_features_cpu).clamp(-1.0, 1.0)

        beta_t = schedule.betas[step_index]
        alpha_t = schedule.alphas[step_index]
        alpha_bar_t = schedule.alpha_bars[step_index]
        alpha_bar_prev = schedule.alpha_bars_prev[step_index]
        posterior_mean_coef_x0 = beta_t * torch.sqrt(alpha_bar_prev) / (1.0 - alpha_bar_t)
        posterior_mean_coef_xt = torch.sqrt(alpha_t) * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)
        mean = posterior_mean_coef_x0 * predicted_x0 + posterior_mean_coef_xt * latent

        if step_index == 0:
            latent = predicted_x0
        else:
            noise = torch.randn_like(latent)
            latent = mean + torch.sqrt(schedule.posterior_variance[step_index]) * noise

    video = decode_video(latent, output_size=32)
    return video.clamp(-1.0, 1.0).cpu()


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location=args.device)

    model = TinyDiffusionModel(
        frame_count=checkpoint["frame_count"],
        latent_size=checkpoint["latent_size"],
        hidden_dim=checkpoint["hidden_dim"],
    ).to(args.device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    hardware_model = TinyDiffusionModelHardware(
        frame_count=checkpoint["frame_count"],
        latent_size=checkpoint["latent_size"],
        hidden_dim=checkpoint["hidden_dim"],
    ).to(args.device)
    hardware_model.load_state_dict(checkpoint["model_state"])
    hardware_model.eval()

    schedule = DiffusionSchedule(num_steps=checkpoint["diffusion_steps"], device=args.device)
    sampled_video = sample_video(model, schedule, digit=args.digit, device=args.device)
    hardware_sampled_video: torch.Tensor | None = None

    if args.compare_hardware:
        compare_seed = 0 if args.seed is None else args.seed
        torch.manual_seed(compare_seed)
        batch_size = 2
        noisy_latent = torch.randn(
            batch_size, model.frame_count, model.latent_size, model.latent_size, device=args.device
        )
        digits = torch.randint(0, 3, (batch_size,), device=args.device, dtype=torch.long)
        timesteps = torch.randint(0, schedule.num_steps, (batch_size,), device=args.device, dtype=torch.long)
        hardware_time_features = precompute_time_features_cpu(timesteps, schedule.num_steps)

        with torch.no_grad():
            software_forward = model(noisy_latent, digits, timesteps, schedule.num_steps)
            hardware_forward = hardware_model._forward_hw(noisy_latent, digits, hardware_time_features)

        is_close_forward, max_abs_forward, max_rel_forward = _comparison_stats(
            software_forward, hardware_forward, args.rtol, args.atol
        )
        print(
            f"forward comparison: {'PASS' if is_close_forward else 'FAIL'} "
            f"(max_abs={max_abs_forward:.3e}, max_rel={max_rel_forward:.3e}, "
            f"rtol={args.rtol:.1e}, atol={args.atol:.1e})"
        )

        torch.manual_seed(compare_seed)
        software_video = sample_video(model, schedule, digit=args.digit, device=args.device)
        torch.manual_seed(compare_seed)
        hardware_video = sample_video_hardware(hardware_model, schedule, digit=args.digit, device=args.device)
        is_close_sample, max_abs_sample, max_rel_sample = _comparison_stats(
            software_video, hardware_video, args.rtol, args.atol
        )
        print(
            f"sampling comparison: {'PASS' if is_close_sample else 'FAIL'} "
            f"(max_abs={max_abs_sample:.3e}, max_rel={max_rel_sample:.3e}, "
            f"rtol={args.rtol:.1e}, atol={args.atol:.1e}, seed={compare_seed})"
        )
        sampled_video = software_video
        hardware_sampled_video = hardware_video

    video = sampled_video[0]
    save_animation(video, args.out, args.fps)
    print(f"saved sample for digit {args.digit} to {args.out}")

    if args.compare_hardware and hardware_sampled_video is not None:
        if args.out_hardware is not None:
            hardware_out = args.out_hardware
        else:
            hardware_out = args.out.with_name(f"{args.out.stem}_hardware{args.out.suffix}")
        hardware_video_frame = hardware_sampled_video[0]
        save_animation(hardware_video_frame, hardware_out, args.fps)
        print(f"saved hardware sample for digit {args.digit} to {hardware_out}")


if __name__ == "__main__":
    main()
