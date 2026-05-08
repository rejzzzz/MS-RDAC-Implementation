# RDAC — Residual-enhanced Deep Animation Codec

Implementation of animation-based video compression with residual coding,
based on the predictive coding framework from:

> Konuko et al., "Predictive coding for animation-based video compression" (ICIP 2023)
> Konuko et al., "Improved predictive coding for animation-based video compression" (EUVIP 2024)

## Project Layout

```
rdac-implementation/
├── apps/                   # Entry points (train, test, data prep)
│   ├── run_experiment.py   # Main CLI dispatcher
│   ├── train.py            # Training loop
│   ├── test.py             # Evaluation / inference
│   └── prepare_data.py     # Dataset preparation helper
├── configs/                # YAML experiment configurations
│   ├── train/
│   └── test/
├── src/                    # Core library (installed via pip install -e .)
│   ├── codec/              # Generator, KPD, DMG, discriminator, nn blocks
│   ├── training/           # Trainer wrappers, loss utilities
│   ├── data/               # Dataset loaders and augmentation
│   ├── entropy/            # Arithmetic + PPM entropy coding
│   ├── assessment/         # Quality metrics (PSNR, LPIPS, VMAF, …)
│   ├── media/              # Anchor codecs (HEVC, VVC) and image coders
│   └── common/             # Shared helpers (logger, visualizer, coding utils)
├── pyproject.toml          # Build config (setuptools, src-layout)
└── requirements.txt
```

## Quick Start

```bash
# 1. Create and activate a virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1          # PowerShell
# venv\Scripts\activate.bat          # CMD

# 2. Upgrade pip
python -m pip install --upgrade pip

# 3. Install dependencies
pip install compressai
pip install -r requirements.txt

# 4. Install project in editable mode (registers src/ packages)
pip install -e .

# 5. Train
python -m apps.run_experiment \
    --config configs/train/train_config.yaml \
    --mode train \
    --model_id rdac

# 6. Test / evaluate
python -m apps.run_experiment \
    --config configs/test/test_config.yaml \
    --mode test \
    --model_id rdac \
    --checkpoint artifacts/<run_dir>/00000009-new-checkpoint.pth.tar

# 7. (Optional) Prepare a Kaggle-style face video dataset
python -m apps.prepare_data
```

## Training Steps

| Step | Description | Recommended Epochs |
|------|-------------|-------------------|
| 0 | End-to-end (animation + residual coding) | 30 |
| 1 | Fine-tune residual coders only | 20 |
| 2 | Train refinement network | 10 |

Set the `step` field in the YAML config and provide the previous step's
checkpoint via `cpk_path`.