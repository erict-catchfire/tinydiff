# TinyDiff

This is a small diffusion model for video generation

## 1. Dataset Preparation

The dataset is generated procedurally in `data.py`. Each sample is a `20`-frame grayscale `32x32` video for digits `0`, `1`, or `2`, using simple seven-segment-style strokes with small style variations.

Generate preview GIFs:

```bash
python data.py --out-dir dataset_previews --variant 0
```

Visual samples:

<img src="dataset_previews/digit_0.gif" alt="Digit 0" width="220" />
<img src="dataset_previews/digit_1.gif" alt="Digit 1" width="220" />
<img src="dataset_previews/digit_2.gif" alt="Digit 2" width="220" />

Training data can also be cached to `dataset_cache.pt` automatically by `train.py`.

## 2. Model Architecture

`model.py` defines a very small conditional diffusion model:

- Videos are downsampled to an `8x8` latent per frame.
- The denoiser is a 3-layer MLP over the flattened video latent.
- Conditioning comes from a digit embedding plus a small timestep feature projection.
- Sampling uses a basic DDPM-style diffusion schedule and decodes latents back to `32x32` frames.

## 3. Training

Train with the default settings:

```bash
python train.py
```

Useful options:

```bash
python train.py \
  --steps 100000 \
  --batch-size 128 \
  --num-samples 4096 \
  --digits 0 1 2 \
  --hidden-dim 128 \
  --latent-size 8 \
  --diffusion-steps 10 \
  --dataset-path dataset_cache.pt \
  --save-path outputs/tinyvid.pt \
  --device cuda
```

## 4. Inference

Generate a video from a trained checkpoint:

```bash
python test.py --checkpoint outputs/tinyvid_relu_100k_h128.pt --digit 2 --out outputs/output_2.gif --device cuda
```

Sample outputs:

<img src="outputs/output_0.gif" alt="Sample output 0" width="280" />
<img src="outputs/output_1.gif" alt="Sample output 1" width="280" />
<img src="outputs/output_2.gif" alt="Sample output 2" width="280" />

`test.py` saves an animated GIF using Pillow if installed, otherwise `ffmpeg` if available.
