from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable


def _require_pyeda():
    try:
        from pyeda.inter import And, Or, expr, exprvar, espresso_exprs  # noqa: F401
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "PyEDA is required for PLA optimization. Install it first, then rerun."
        ) from exc


def _sanitize_identifier(name: str) -> str:
    sanitized = re.sub(r"[^a-zA-Z0-9_]", "_", name)
    if not sanitized:
        sanitized = "unnamed"
    if sanitized[0].isdigit():
        sanitized = f"n_{sanitized}"
    return sanitized


def _module_name_from_path(pla_path: Path) -> str:
    return _sanitize_identifier(pla_path.stem.replace("-", "m"))


def _parse_unit_metadata_from_filename(pla_path: Path) -> dict[str, int | list[int] | str]:
    pattern = re.compile(
        r"^.*x(?P<x>\d+)_y(?P<y>\d+)_w_(?P<weights>.+?)_b_(?P<bias>-?\d+)(?:_s\d+)?$"
    )
    match = pattern.match(pla_path.stem)
    if not match:
        return {
            "x": -1,
            "y": -1,
            "weights": [],
            "bias": 0,
            "source_name": pla_path.name,
        }

    weights_text = match.group("weights")
    weights = [int(token) for token in weights_text.split("_")] if weights_text else []
    return {
        "x": int(match.group("x")),
        "y": int(match.group("y")),
        "weights": weights,
        "bias": int(match.group("bias")),
        "source_name": pla_path.name,
    }


def _parse_labels(line: str) -> list[str]:
    parts = line.strip().split()
    return [token for token in parts[1:]]


def _parse_pla_file(pla_path: Path) -> dict:
    num_inputs: int | None = None
    num_outputs: int | None = None
    input_labels: list[str] = []
    output_labels: list[str] = []
    rows: list[tuple[str, str]] = []

    with pla_path.open("r", encoding="utf-8") as pla_file:
        for raw_line in pla_file:
            line = raw_line.strip()
            if not line:
                continue

            if line.startswith("#"):
                continue
            if line == ".e":
                break

            if line.startswith(".i "):
                num_inputs = int(line.split()[1])
                continue
            if line.startswith(".o "):
                num_outputs = int(line.split()[1])
                continue
            if line.startswith(".ilb "):
                input_labels = _parse_labels(line)
                continue
            if line.startswith(".ob "):
                output_labels = _parse_labels(line)
                continue
            if line.startswith("."):
                # Ignore other directives, such as ".p".
                continue

            row_parts = line.split()
            if len(row_parts) != 2:
                raise ValueError(f"Malformed PLA row in {pla_path}: {line}")
            in_bits, out_bits = row_parts
            rows.append((in_bits, out_bits))

    if num_inputs is None or num_outputs is None:
        raise ValueError(f"PLA missing .i/.o header: {pla_path}")
    if not input_labels:
        input_labels = [f"in_{idx}" for idx in range(num_inputs)]
    if not output_labels:
        output_labels = [f"out_{idx}" for idx in range(num_outputs)]
    if len(input_labels) != num_inputs:
        raise ValueError(f".ilb label count does not match .i in {pla_path}")
    if len(output_labels) != num_outputs:
        raise ValueError(f".ob label count does not match .o in {pla_path}")

    for in_bits, out_bits in rows:
        if len(in_bits) != num_inputs:
            raise ValueError(f"Input row width mismatch in {pla_path}")
        if len(out_bits) != num_outputs:
            raise ValueError(f"Output row width mismatch in {pla_path}")

    return {
        "num_inputs": num_inputs,
        "num_outputs": num_outputs,
        "input_labels": [_sanitize_identifier(name) for name in input_labels],
        "output_labels": [_sanitize_identifier(name) for name in output_labels],
        "rows": rows,
    }


def _cube_to_term(cube_bits: str, input_vars: list):
    from pyeda.inter import And, expr

    literals = []
    for bit, var in zip(cube_bits, input_vars):
        if bit == "1":
            literals.append(var)
        elif bit == "0":
            literals.append(~var)
        elif bit == "-":
            continue
        else:
            raise ValueError(f"Unsupported input cube symbol: {bit}")

    if not literals:
        return expr(True)
    if len(literals) == 1:
        return literals[0]
    return And(*literals)


def _build_expr_for_output_bit(rows: list[tuple[str, str]], output_bit_idx: int, input_vars: list):
    from pyeda.inter import Or, expr

    onset_terms = []
    for in_bits, out_bits in rows:
        bit = out_bits[output_bit_idx]
        if bit == "1":
            onset_terms.append(_cube_to_term(in_bits, input_vars))
        elif bit == "0" or bit == "-":
            continue
        else:
            raise ValueError(f"Unsupported output bit symbol: {bit}")

    if not onset_terms:
        return expr(False)
    if len(onset_terms) == 1:
        return onset_terms[0]
    return Or(*onset_terms)


def _literal_sort_key(literal) -> tuple[int, int]:
    is_complement = int(_is_complement_literal(literal))
    base = literal.inputs[0] if is_complement else literal
    idx = base.indices[0] if getattr(base, "indices", ()) else 0
    return idx, is_complement


def _is_complement_literal(literal) -> bool:
    return literal.__class__.__name__ == "Complement"


def _literal_to_verilog(literal) -> str:
    is_complement = _is_complement_literal(literal)
    base = literal.inputs[0] if is_complement else literal
    if not getattr(base, "indices", ()):
        raise ValueError("Expected indexed variable from exprvar array.")
    var_idx = base.indices[0]
    signal = f"i{var_idx}"
    return f"~{signal}" if is_complement else signal


def _expr_to_verilog_sop(minimized_expr) -> str:
    dnf_expr = minimized_expr.to_dnf()
    if dnf_expr.is_zero():
        return "1'b0"
    if dnf_expr.is_one():
        return "1'b1"

    cubes = sorted(dnf_expr.cover, key=lambda cube: tuple(_literal_sort_key(lit) for lit in sorted(cube, key=_literal_sort_key)))
    terms = []
    for cube in cubes:
        sorted_literals = sorted(cube, key=_literal_sort_key)
        lit_exprs = [_literal_to_verilog(lit) for lit in sorted_literals]
        if not lit_exprs:
            terms.append("1'b1")
        elif len(lit_exprs) == 1:
            terms.append(lit_exprs[0])
        else:
            terms.append("(" + " & ".join(lit_exprs) + ")")

    if not terms:
        return "1'b0"
    if len(terms) == 1:
        return terms[0]
    return " | ".join(terms)


def _extract_cubes(minimized_expr) -> list:
    dnf_expr = minimized_expr.to_dnf()
    if dnf_expr.is_zero() or dnf_expr.is_one():
        return []
    return list(dnf_expr.cover)


def _and_depth_histogram(cubes: list, max_depth: int = 24) -> dict[int, int]:
    histogram = {depth: 0 for depth in range(1, max_depth + 1)}
    for cube in cubes:
        depth = len(cube)
        if 1 <= depth <= max_depth:
            histogram[depth] += 1
    return histogram


def _or_count(cubes: list) -> int:
    if len(cubes) <= 1:
        return 0
    return len(cubes) - 1


def _histogram_text(histogram: dict[int, int]) -> str:
    return " ".join(f"{depth}:{histogram[depth]}" for depth in sorted(histogram))


def _build_metadata_text(
    module_name: str,
    unit_meta: dict[str, int | list[int] | str],
    output_labels: list[str],
    output_cubes: list[list],
) -> str:
    max_depth = 24
    combined_histogram = {depth: 0 for depth in range(1, max_depth + 1)}
    max_or = 0

    lines: list[str] = []
    lines.append(f"module={module_name}")
    lines.append(f"source_pla={unit_meta['source_name']}")
    lines.append(f"x={unit_meta['x']}")
    lines.append(f"y={unit_meta['y']}")
    lines.append(f"weights={unit_meta['weights']}")
    lines.append(f"bias={unit_meta['bias']}")
    lines.append("")

    for out_idx, (output_name, cubes) in enumerate(zip(output_labels, output_cubes)):
        histogram = _and_depth_histogram(cubes=cubes, max_depth=max_depth)
        for depth, count in histogram.items():
            combined_histogram[depth] += count
        ors = _or_count(cubes)
        max_or = max(max_or, ors)

        lines.append(f"output_index={out_idx}")
        lines.append(f"output_name={output_name}")
        lines.append(f"or_count={ors}")
        lines.append(f"and_depth_histogram_1_to_24={_histogram_text(histogram)}")
        lines.append("")

    lines.append(f"max_or_across_outputs={max_or}")
    lines.append(f"combined_and_depth_histogram_1_to_24={_histogram_text(combined_histogram)}")
    lines.append("")
    return "\n".join(lines)


def _build_verilog_module(
    module_name: str,
    input_labels: list[str],
    output_labels: list[str],
    output_exprs: list[str],
    source_pla_name: str,
) -> str:
    lines = []
    lines.append(f"// Auto-generated from {source_pla_name}")
    lines.append("// Minimized with PyEDA Espresso")
    lines.append(f"module {module_name}(")

    io_lines = [f"    input wire {name}" for name in input_labels]
    io_lines.extend(f"    output wire {name}" for name in output_labels)
    lines.append(",\n".join(io_lines))
    lines.append(");")
    lines.append("")

    for idx, input_name in enumerate(input_labels):
        lines.append(f"    wire i{idx} = {input_name};")
    lines.append("")

    for output_name, output_expr in zip(output_labels, output_exprs):
        lines.append(f"    assign {output_name} = {output_expr};")

    lines.append("")
    lines.append("endmodule")
    lines.append("")
    return "\n".join(lines)


def convert_pla_to_verilog(pla_path: str | Path, output_dir: str | Path) -> Path:
    """
    Convert one PLA file into a minimized Verilog module.

    Each output bit is minimized independently through PyEDA Espresso.
    """
    _require_pyeda()
    from pyeda.inter import exprvar, espresso_exprs

    pla_path = Path(pla_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir = output_dir.parent / "verilog_metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    parsed = _parse_pla_file(pla_path)
    num_inputs = parsed["num_inputs"]
    num_outputs = parsed["num_outputs"]
    input_labels = parsed["input_labels"]
    output_labels = parsed["output_labels"]
    rows = parsed["rows"]

    input_vars = [exprvar("i", idx) for idx in range(num_inputs)]
    output_exprs: list[str] = []
    output_cubes: list[list] = []
    for bit_idx in range(num_outputs):
        raw_expr = _build_expr_for_output_bit(rows=rows, output_bit_idx=bit_idx, input_vars=input_vars)
        raw_dnf = raw_expr.to_dnf()
        if raw_dnf.is_zero() or raw_dnf.is_one() or len(raw_dnf.support) == 0:
            minimized_expr = raw_dnf
        else:
            minimized_expr = espresso_exprs(raw_dnf)[0]
        output_exprs.append(_expr_to_verilog_sop(minimized_expr))
        output_cubes.append(_extract_cubes(minimized_expr))

    module_name = _module_name_from_path(pla_path)
    verilog_text = _build_verilog_module(
        module_name=module_name,
        input_labels=input_labels,
        output_labels=output_labels,
        output_exprs=output_exprs,
        source_pla_name=pla_path.name,
    )

    output_path = output_dir / f"{module_name}.v"
    output_path.write_text(verilog_text, encoding="utf-8")

    unit_meta = _parse_unit_metadata_from_filename(pla_path)
    metadata_text = _build_metadata_text(
        module_name=module_name,
        unit_meta=unit_meta,
        output_labels=output_labels,
        output_cubes=output_cubes,
    )
    metadata_path = metadata_dir / f"{module_name}.txt"
    metadata_path.write_text(metadata_text, encoding="utf-8")
    return output_path


def convert_all_pla_in_directory(
    pla_directory: str | Path,
    output_directory: str | Path = "baked_unit_modeling/verilog_outputs",
) -> list[Path]:
    pla_directory = Path(pla_directory)
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    pla_files = sorted(pla_directory.glob("*.pla"))
    generated: list[Path] = []
    for pla_file in pla_files:
        generated.append(convert_pla_to_verilog(pla_file, output_directory))
    return generated


def _format_paths(paths: Iterable[Path]) -> str:
    return "\n".join(str(path) for path in paths)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert PLA files to minimized Verilog with PyEDA Espresso."
    )
    parser.add_argument(
        "--pla-dir",
        default="baked_unit_modeling/outputs",
        help="Directory containing input .pla files.",
    )
    parser.add_argument(
        "--verilog-dir",
        default="baked_unit_modeling/verilog_outputs",
        help="Directory to write output .v files.",
    )
    args = parser.parse_args()

    generated = convert_all_pla_in_directory(
        pla_directory=args.pla_dir,
        output_directory=args.verilog_dir,
    )
    print(f"Generated {len(generated)} Verilog files:")
    if generated:
        print(_format_paths(generated))


if __name__ == "__main__":
    main()

