from __future__ import annotations

import argparse
import random
from pathlib import Path

from tqdm import tqdm  # pyright: ignore[reportMissingModuleSource]

from .pla_to_verilog import convert_pla_to_verilog
from .truth_table import default_output_filename, generate_pla_truth_table

INT8_MIN = -128
INT8_MAX = 127


def _parse_xy_pair(text: str) -> tuple[int, int]:
    parts = text.split(",")
    if len(parts) != 2:
        raise ValueError(f"Invalid config '{text}'. Expected format like '2,1'.")
    x = int(parts[0].strip())
    y = int(parts[1].strip())
    if x <= 0 or y <= 0:
        raise ValueError(f"Invalid config '{text}'. x and y must be > 0.")
    if y > x:
        raise ValueError(f"Invalid config '{text}'. Requires y <= x.")
    return x, y


def _parse_xy_configs(config_texts: list[str]) -> list[tuple[int, int]]:
    return [_parse_xy_pair(text) for text in config_texts]


def _random_int8(rng: random.Random) -> int:
    return rng.randint(INT8_MIN, INT8_MAX)


def _sample_filename(x: int, y: int, weights: list[int], sample_idx: int) -> str:
    base_name = default_output_filename(x=x, y=y, weights=weights).removesuffix(".pla")
    return f"{base_name}_s{sample_idx:03d}.pla"


def _estimated_rows(x: int) -> int:
    return 256**x


def _metadata_path_from_verilog(verilog_path: Path) -> Path:
    return verilog_path.parent.parent / "verilog_metadata" / f"{verilog_path.stem}.txt"


def _latest_metadata_summary(metadata_path: Path) -> str:
    if not metadata_path.exists():
        return f"metadata_missing={metadata_path}"

    lines = metadata_path.read_text(encoding="utf-8").splitlines()
    wanted_prefixes = (
        "module=",
        "x=",
        "y=",
        "weights=",
        "max_or_across_outputs=",
        "combined_and_depth_histogram_1_to_24=",
    )
    selected: list[str] = []
    for line in lines:
        if line.startswith(wanted_prefixes):
            selected.append(line)
    return " | ".join(selected)


def generate_random_modules(
    xy_configs: list[tuple[int, int]],
    samples_per_config: int,
    pla_dir: str | Path,
    verilog_dir: str | Path,
    seed: int | None = None,
    bottom_up: bool = False,
    show_latest_metadata: bool = False,
) -> list[Path]:
    rng = random.Random(seed)
    pla_dir = Path(pla_dir)
    verilog_dir = Path(verilog_dir)
    pla_dir.mkdir(parents=True, exist_ok=True)
    verilog_dir.mkdir(parents=True, exist_ok=True)

    ordered_configs = list(xy_configs)
    if bottom_up:
        ordered_configs.reverse()

    generated_verilog_paths: list[Path] = []
    for x, y in ordered_configs:
        seen: set[tuple[int, ...]] = set()
        generated = 0
        attempts = 0
        max_attempts = samples_per_config * 20

        print(f"Starting config x={x}, y={y}, rows_per_pla={_estimated_rows(x)}", flush=True)
        with tqdm(total=samples_per_config, desc=f"{x},{y}", unit="module") as progress:
            while generated < samples_per_config:
                attempts += 1
                if attempts > max_attempts:
                    raise RuntimeError(
                        f"Could not generate enough unique random modules for x={x}, y={y}."
                    )

                weights = [_random_int8(rng) for _ in range(x)]
                key = tuple(weights)
                if key in seen:
                    continue
                seen.add(key)

                pla_filename = _sample_filename(x=x, y=y, weights=weights, sample_idx=generated)
                pla_path = pla_dir / pla_filename
                generate_pla_truth_table(
                    x=x,
                    y=y,
                    weights=weights,
                    output_path=pla_path,
                )
                verilog_path = convert_pla_to_verilog(pla_path=pla_path, output_dir=verilog_dir)
                generated_verilog_paths.append(verilog_path)

                generated += 1
                progress.update(1)

                if show_latest_metadata:
                    metadata_path = _metadata_path_from_verilog(verilog_path)
                    summary = _latest_metadata_summary(metadata_path)
                    progress.write(f"latest_metadata: {summary}")

    return generated_verilog_paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate random baked-unit modules: PLA truth tables, minimized Verilog, and metadata."
        )
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=["1,1", "2,2", "2,1", "3,3"],
        help="Space-separated x,y pairs. Example: --configs 1,1 2,1 3,3",
    )
    parser.add_argument(
        "--samples-per-config",
        type=int,
        default=100,
        help="Number of random modules to generate per (x,y) config.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--pla-dir",
        default="baked_unit_modeling/outputs",
        help="Directory for generated PLA files.",
    )
    parser.add_argument(
        "--verilog-dir",
        default="baked_unit_modeling/verilog_outputs",
        help="Directory for generated Verilog files.",
    )
    parser.add_argument(
        "--bottom-up",
        action="store_true",
        help="Process configs from the last item to the first item.",
    )
    parser.add_argument(
        "--show-latest-metadata",
        action="store_true",
        help="Print a compact summary of the latest generated metadata file.",
    )
    args = parser.parse_args()

    if args.samples_per_config <= 0:
        raise ValueError("--samples-per-config must be > 0")

    xy_configs = _parse_xy_configs(args.configs)
    generated = generate_random_modules(
        xy_configs=xy_configs,
        samples_per_config=args.samples_per_config,
        pla_dir=args.pla_dir,
        verilog_dir=args.verilog_dir,
        seed=args.seed,
        bottom_up=args.bottom_up,
        show_latest_metadata=args.show_latest_metadata,
    )
    print(f"Done. Generated {len(generated)} Verilog modules.", flush=True)


if __name__ == "__main__":
    main()

