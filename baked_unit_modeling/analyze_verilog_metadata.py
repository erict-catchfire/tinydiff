from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

import matplotlib.pyplot as plt  # pyright: ignore[reportMissingImports]


def _parse_histogram(text: str) -> list[int]:
    pairs = text.split()
    values: list[int] = []
    for pair in pairs:
        _, value = pair.split(":")
        values.append(int(value))
    return values


def _parse_metadata_file(path: Path) -> tuple[tuple[int, int], int, list[int]]:
    x: int | None = None
    y: int | None = None
    max_or: int | None = None
    combined_hist: list[int] | None = None

    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("x="):
            x = int(line.split("=", 1)[1].strip())
        elif line.startswith("y="):
            y = int(line.split("=", 1)[1].strip())
        elif line.startswith("max_or_across_outputs="):
            max_or = int(line.split("=", 1)[1].strip())
        elif line.startswith("combined_and_depth_histogram_1_to_24="):
            hist_text = line.split("=", 1)[1].strip()
            combined_hist = _parse_histogram(hist_text)

    if x is None or y is None:
        raise ValueError(f"Missing x or y in {path}")
    if max_or is None:
        raise ValueError(f"Missing max_or_across_outputs in {path}")
    if combined_hist is None:
        raise ValueError(f"Missing combined_and_depth_histogram_1_to_24 in {path}")
    if len(combined_hist) != 24:
        raise ValueError(f"Expected 24 histogram buckets in {path}, got {len(combined_hist)}")

    return (x, y), max_or, combined_hist


def _boxplot_by_group(
    data_by_group: dict[str, list[float]],
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    labels = sorted(data_by_group.keys())
    data = [data_by_group[label] for label in labels]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(data, tick_labels=labels, showfliers=True)
    ax.set_title(title)
    ax.set_xlabel("x,y configuration")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _write_summary_csv(
    output_csv: Path,
    max_or_by_group: dict[str, list[int]],
    bucket_by_group: dict[int, dict[str, list[int]]],
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "metric",
                "bucket",
                "xy",
                "count",
                "mean",
                "median",
                "min",
                "max",
            ]
        )

        for xy in sorted(max_or_by_group.keys()):
            values = max_or_by_group[xy]
            writer.writerow(
                [
                    "max_or_across_outputs",
                    "",
                    xy,
                    len(values),
                    f"{mean(values):.4f}",
                    f"{median(values):.4f}",
                    min(values),
                    max(values),
                ]
            )

        for bucket in range(1, 25):
            for xy in sorted(bucket_by_group[bucket].keys()):
                values = bucket_by_group[bucket][xy]
                writer.writerow(
                    [
                        "combined_and_depth_histogram",
                        bucket,
                        xy,
                        len(values),
                        f"{mean(values):.4f}",
                        f"{median(values):.4f}",
                        min(values),
                        max(values),
                    ]
                )


def _plot_bucket_range_same_plot(
    bucket_by_group: dict[int, dict[str, list[int]]],
    output_path: Path,
    start_bucket_zero_based: int,
    end_bucket_zero_based: int,
) -> None:
    if start_bucket_zero_based < 0 or end_bucket_zero_based < 0:
        raise ValueError("Bucket range must be >= 0")
    if end_bucket_zero_based < start_bucket_zero_based:
        raise ValueError("Bucket range end must be >= start")

    # Metadata buckets are 1..24. User-facing range here is zero-based (0..23).
    start_bucket = start_bucket_zero_based + 1
    end_bucket = end_bucket_zero_based + 1
    if end_bucket > 24:
        raise ValueError("Bucket range exceeds available metadata buckets (0..23).")

    all_groups = sorted(
        {
            group
            for bucket in range(start_bucket, end_bucket + 1)
            for group in bucket_by_group[bucket].keys()
        }
    )
    if not all_groups:
        raise ValueError("No data available for requested bucket range.")

    fig, ax = plt.subplots(figsize=(16, 6))

    group_width = 0.75
    box_width = group_width / len(all_groups)

    for group_idx, group_name in enumerate(all_groups):
        positions: list[float] = []
        data: list[list[int]] = []
        for bucket in range(start_bucket, end_bucket + 1):
            base = (bucket - start_bucket) + 1
            offset = -group_width / 2 + (group_idx + 0.5) * box_width
            positions.append(base + offset)
            data.append(bucket_by_group[bucket][group_name])
        box = ax.boxplot(
            data,
            positions=positions,
            widths=box_width * 0.9,
            patch_artist=True,
            showfliers=False,
        )
        color = f"C{group_idx % 10}"
        for patch in box["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.45)
        for median_line in box["medians"]:
            median_line.set_color("black")
            median_line.set_linewidth(1.2)

    # One legend entry per x,y group.
    legend_handles = [
        plt.Line2D([0], [0], color=f"C{i % 10}", lw=8, alpha=0.45) for i in range(len(all_groups))
    ]
    ax.legend(legend_handles, all_groups, loc="upper right")

    x_positions = list(range(1, end_bucket - start_bucket + 2))
    x_labels = [str(i) for i in range(start_bucket_zero_based, end_bucket_zero_based + 1)]
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("combined_and_depth histogram bucket (zero-based)")
    ax.set_ylabel("bucket count")
    ax.set_title(
        f"combined_and_depth histogram buckets {start_bucket_zero_based}-{end_bucket_zero_based} "
        "on one plot"
    )
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def analyze(metadata_dir: Path, output_dir: Path, xy_filters: set[tuple[int, int]] | None) -> None:
    metadata_files = sorted(metadata_dir.glob("*.txt"))
    if not metadata_files:
        raise FileNotFoundError(f"No metadata files found in {metadata_dir}")

    max_or_by_group: dict[str, list[int]] = defaultdict(list)
    bucket_by_group: dict[int, dict[str, list[int]]] = {
        bucket: defaultdict(list) for bucket in range(1, 25)
    }

    total_used = 0
    for path in metadata_files:
        xy, max_or, combined_hist = _parse_metadata_file(path)
        if xy_filters is not None and xy not in xy_filters:
            continue
        total_used += 1

        xy_key = f"x{xy[0]}_y{xy[1]}"
        max_or_by_group[xy_key].append(max_or)

        for idx, value in enumerate(combined_hist, start=1):
            bucket_by_group[idx][xy_key].append(value)

    if total_used == 0:
        raise ValueError("No files matched the requested x,y filters.")

    output_dir.mkdir(parents=True, exist_ok=True)

    _boxplot_by_group(
        data_by_group=max_or_by_group,
        ylabel="max_or_across_outputs",
        title="max_or_across_outputs by x,y config",
        output_path=output_dir / "max_or_across_outputs_boxplot.png",
    )

    for bucket in range(1, 25):
        _boxplot_by_group(
            data_by_group=bucket_by_group[bucket],
            ylabel=f"Combined AND depth count (bucket {bucket})",
            title=f"combined_and_depth_histogram bucket {bucket} by x,y config",
            output_path=output_dir / f"combined_bucket_{bucket:02d}_boxplot.png",
        )

    _write_summary_csv(
        output_csv=output_dir / "metadata_summary_stats.csv",
        max_or_by_group=max_or_by_group,
        bucket_by_group=bucket_by_group,
    )

    print(f"Analyzed {total_used} metadata files.")
    print(f"Wrote plots and stats to: {output_dir}")


def _parse_xy_filter_values(raw_xy: list[str] | None) -> set[tuple[int, int]] | None:
    if not raw_xy:
        return None
    parsed: set[tuple[int, int]] = set()
    for item in raw_xy:
        parts = item.split(",")
        if len(parts) != 2:
            raise ValueError(f"Invalid --xy value '{item}'. Expected format x,y (example 2,1).")
        parsed.add((int(parts[0].strip()), int(parts[1].strip())))
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze Verilog metadata files and create box-and-whisker plots for "
            "combined AND-depth histogram buckets and max_or_across_outputs."
        )
    )
    parser.add_argument(
        "--metadata-dir",
        type=Path,
        default=Path("baked_unit_modeling/verilog_metadata"),
        help="Directory containing metadata .txt files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("baked_unit_modeling/analysis_plots"),
        help="Directory to save plots and CSV summary.",
    )
    parser.add_argument(
        "--xy",
        nargs="*",
        help="Optional x,y filters (example: --xy 1,1 2,1 2,2).",
    )
    parser.add_argument(
        "--same-plot-range",
        nargs=2,
        type=int,
        metavar=("START_BUCKET", "END_BUCKET"),
        help=(
            "Optional zero-based bucket range to draw on one combined plot "
            "(example: --same-plot-range 0 15)."
        ),
    )
    args = parser.parse_args()

    xy_filters = _parse_xy_filter_values(args.xy)
    analyze(metadata_dir=args.metadata_dir, output_dir=args.output_dir, xy_filters=xy_filters)

    if args.same_plot_range is not None:
        start_bucket, end_bucket = args.same_plot_range

        # Re-read the grouped bucket data quickly from summary inputs by parsing metadata once more.
        # This keeps the CLI usage simple while preserving the existing analyze() flow.
        bucket_by_group: dict[int, dict[str, list[int]]] = {
            bucket: defaultdict(list) for bucket in range(1, 25)
        }
        for path in sorted(args.metadata_dir.glob("*.txt")):
            xy, _, combined_hist = _parse_metadata_file(path)
            if xy_filters is not None and xy not in xy_filters:
                continue
            xy_key = f"x{xy[0]}_y{xy[1]}"
            for idx, value in enumerate(combined_hist, start=1):
                bucket_by_group[idx][xy_key].append(value)

        output_name = f"combined_buckets_{start_bucket:02d}_to_{end_bucket:02d}_same_plot.png"
        _plot_bucket_range_same_plot(
            bucket_by_group=bucket_by_group,
            output_path=args.output_dir / output_name,
            start_bucket_zero_based=start_bucket,
            end_bucket_zero_based=end_bucket,
        )
        print(f"Wrote combined same-plot figure: {args.output_dir / output_name}")


if __name__ == "__main__":
    main()
