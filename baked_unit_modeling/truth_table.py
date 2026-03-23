from __future__ import annotations

from itertools import product
from pathlib import Path


SIGNED_MIN = -128
SIGNED_MAX = 127
BYTE_WIDTH = 8


def _validate_signed_byte(value: int, name: str) -> None:
    if not isinstance(value, int):
        raise TypeError(f"{name} must be an int.")
    if value < SIGNED_MIN or value > SIGNED_MAX:
        raise ValueError(f"{name} must be in [{SIGNED_MIN}, {SIGNED_MAX}].")


def _signed_from_u8(value: int) -> int:
    if value < 128:
        return value
    return value - 256


def _u8_from_signed(value: int) -> int:
    return value & 0xFF


def _bits_from_u8(value: int) -> str:
    return f"{value:08b}"


def _clamp_signed_8(value: int) -> int:
    if value < SIGNED_MIN:
        return SIGNED_MIN
    if value > SIGNED_MAX:
        return SIGNED_MAX
    return value


def _split_sizes(x: int, y: int) -> list[int]:
    """
    Split x inputs across y outputs.

    We keep earlier outputs as floor-sized chunks and assign the remainder
    to later outputs. For y=2 this matches:
    - out0: floor(x/2)
    - out1: ceil(x/2)
    """
    base = x // y
    remainder = x % y
    sizes = [base] * y
    for idx in range(y - remainder, y):
        if remainder == 0:
            break
        sizes[idx] += 1
    return sizes


def _compute_outputs_signed(
    input_values_u8: tuple[int, ...],
    weights: list[int],
    y: int,
) -> list[int]:
    signed_inputs = [_signed_from_u8(value) for value in input_values_u8]
    products = [signed_inputs[idx] * weights[idx] for idx in range(len(weights))]

    split_sizes = _split_sizes(len(weights), y)
    outputs: list[int] = []
    start = 0
    for out_idx, chunk_size in enumerate(split_sizes):
        end = start + chunk_size
        partial_sum = sum(products[start:end])
        outputs.append(_clamp_signed_8(partial_sum))
        start = end
    return outputs


def _input_label_names(x: int) -> list[str]:
    labels: list[str] = []
    for input_idx in range(x):
        for bit_idx in range(BYTE_WIDTH - 1, -1, -1):
            labels.append(f"in{input_idx}_b{bit_idx}")
    return labels


def _output_label_names(y: int) -> list[str]:
    labels: list[str] = []
    for output_idx in range(y):
        for bit_idx in range(BYTE_WIDTH - 1, -1, -1):
            labels.append(f"out{output_idx}_b{bit_idx}")
    return labels


def generate_pla_truth_table(
    x: int,
    y: int,
    weights: list[int],
    output_path: str | Path,
) -> Path:
    """
    Generate an Espresso-compatible PLA truth table for a baked unit.

    Rules:
    - Arithmetic is signed int8 for inputs/weights semantics.
    - Outputs are signed and saturating-clamped to [-128, 127].
    - Inputs are split into y contiguous groups.
    - Group sizes use floor/ceil partitioning (earlier outputs get floor-size groups).
    - Each output is a clamped partial sum of its group (no bias term).

    Parameters
    ----------
    x : int
        Number of 8-bit inputs.
    y : int
        Number of 8-bit outputs (1 <= y <= x).
    weights : list[int]
        Signed int8 fixed weights. Must have length x.
    output_path : str | Path
        Destination PLA file path.
    """
    if not isinstance(x, int) or x <= 0:
        raise ValueError("x must be a positive integer.")
    if not isinstance(y, int) or y <= 0:
        raise ValueError("y must be a positive integer.")
    if y > x:
        raise ValueError("y must be less than or equal to x.")
    if len(weights) != x:
        raise ValueError("weights length must equal x.")

    for idx, weight in enumerate(weights):
        _validate_signed_byte(weight, f"weights[{idx}]")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    num_input_bits = x * BYTE_WIDTH
    num_output_bits = y * BYTE_WIDTH
    row_count = 256**x

    input_labels = " ".join(_input_label_names(x))
    output_labels = " ".join(_output_label_names(y))

    with output_path.open("w", encoding="utf-8") as pla_file:
        pla_file.write(f".i {num_input_bits}\n")
        pla_file.write(f".o {num_output_bits}\n")
        pla_file.write(f".ilb {input_labels}\n")
        pla_file.write(f".ob {output_labels}\n")
        pla_file.write(f".p {row_count}\n")

        for input_values_u8 in product(range(256), repeat=x):
            outputs_signed = _compute_outputs_signed(
                input_values_u8=input_values_u8,
                weights=weights,
                y=y,
            )

            input_bits = "".join(_bits_from_u8(value) for value in input_values_u8)
            output_bits = "".join(_bits_from_u8(_u8_from_signed(value)) for value in outputs_signed)
            pla_file.write(f"{input_bits} {output_bits}\n")

        pla_file.write(".e\n")

    return output_path


def default_output_filename(x: int, y: int, weights: list[int]) -> str:
    weight_text = "_".join(str(weight) for weight in weights)
    return f"tt_x{x}_y{y}_w_{weight_text}.pla"

