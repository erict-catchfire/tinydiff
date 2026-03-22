# Baked Unit Modeling

This module generates Espresso-compatible `.pla` truth tables for baked hardware units
with fixed signed int8 weights and bias.

## Current behavior

- Inputs are 8-bit signed values (two's complement encoding in the truth table bits).
- Weights and bias are signed int8 (`-128..127`).
- Products and sums are computed in Python integer precision.
- Outputs are clamped with signed int8 saturation to `[-128, 127]`.
- Bias is added exactly once:
  - `y=1`: `out0 = sum(all products) + bias`
  - `y=2`: `out0 = sum(first floor(x/2) products) + bias`, `out1 = sum(remaining products)`

## API

```python
from baked_unit_modeling import default_output_filename, generate_pla_truth_table

x = 2
y = 2
weights = [13, -7]
bias = 9

output_name = default_output_filename(x, y, weights, bias)
output_path = f"baked_unit_modeling/outputs/{output_name}"

generate_pla_truth_table(
    x=x,
    y=y,
    weights=weights,
    bias=bias,
    output_path=output_path,
)
```

The generated file contains standard PLA headers (`.i`, `.o`, `.ilb`, `.ob`, `.p`, `.e`)
followed by rows in the form:

`<all_input_bits> <all_output_bits>`

## PLA -> Verilog (PyEDA Espresso)

`pla_to_verilog.py` converts `.pla` files to minimized Verilog:

- reads every `.pla` in an input folder
- minimizes each output bit independently with PyEDA Espresso
- writes one Verilog module per `.pla` into a single output folder

Default folders:

- PLA input: `baked_unit_modeling/outputs/`
- Verilog output: `baked_unit_modeling/verilog_outputs/`

Run:

```bash
python -m baked_unit_modeling.pla_to_verilog
```

Or with explicit folders:

```bash
python -m baked_unit_modeling.pla_to_verilog --pla-dir baked_unit_modeling/outputs --verilog-dir baked_unit_modeling/verilog_outputs
```

## Batch random module generation

`generate_random_modules.py` automates:

1. random `(weights, bias)` sampling
2. PLA generation
3. PyEDA/Espresso minimization
4. Verilog emission
5. metadata emission

Default run (your requested sweep):

```bash
python -m baked_unit_modeling.generate_random_modules --configs 1,1 2,2 2,1 3,3 --samples-per-config 100 --seed 0
```

Fast smoke test:

```bash
python -m baked_unit_modeling.generate_random_modules --configs 1,1 2,1 --samples-per-config 2 --seed 0
```

