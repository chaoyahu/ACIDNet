# ACIDNet Project Guide

This document describes the public project layout and the expected workflow for
using the open-source ACIDNet codebase.

## Public Layout

```text
ACIDNet/
  acidnet/
    data/              Dataset wrappers, loaders, schedulers, train options
    losses/            Loss functions and image-quality utilities
    models/            Main ACIDNet model and model components
    checkpoints.py     Checkpoint loading helpers
    paths.py           Repository, data, output, and weight paths
  train/
    train.py           Main training entrypoint
  eval/
    benchmark.py       End-to-end benchmark inference and metric evaluation
    infer.py           Dataset-specific inference helper
    infer_sid_blur.py  Optional SID/blur inference helper
  measure/
    paired_metrics.py  PSNR, SSIM, and LPIPS for paired datasets
    unpaired_metrics.py NIQE and BRISQUE for unpaired outputs
    sid_blur_metrics.py Optional SID/blur paired metrics
  docs/
    figures/           Public figures
```

## Ignored Local Content

The following content is intentionally not part of the public source tree:

- `datasets/`: local dataset copies
- `outputs/`: generated images, metrics, logs, and checkpoints
- `experiments/weights/`: local pretrained weights
- `ablation/`: private ablation code and checkpoints
- `visualization/`: private paper-analysis and figure-generation scripts
- `docs/reports/`: locally generated benchmark reports

These paths are covered by `.gitignore` so they can remain in a local working
copy without being published.

## Configuration

Datasets are resolved through `ACIDNET_DATASET_ROOT`.

```bash
export ACIDNET_DATASET_ROOT=/path/to/datasets
```

If the environment variable is not set, the code falls back to `datasets/`
inside the repository root. That fallback is only for local convenience and is
ignored by git.

Outputs are written under `outputs/`. This directory is generated at runtime and
should not be committed.

## Main Commands

Training initialization check:

```bash
python -m train.train --dataset lol_v1 --dry_run
```

Training:

```bash
python -m train.train --dataset lol_v1
```

Benchmark evaluation:

```bash
python -m eval.benchmark --dataset-root "$ACIDNET_DATASET_ROOT" --paired-only
```

Single-dataset paired metrics:

```bash
python -m measure.paired_metrics \
  --pred-dir outputs/benchmark/LOLv2_syn \
  --gt-dir "$ACIDNET_DATASET_ROOT/LOLv2/Synthetic/Test/Normal"
```

Unpaired metrics:

```bash
python -m measure.unpaired_metrics --pred-dir outputs/benchmark/DICM
```

## Public Import Surface

The public model import is:

```python
from acidnet.models.acidnet import ACIDNet
```

Private ablation classes are not part of the public API.

## Release Checklist

Before publishing:

1. Run `python -m compileall acidnet train eval measure`.
2. Run a benchmark smoke test with `--limit 1`.
3. Confirm no local absolute paths are present in public source files.
4. Confirm generated files, datasets, weights, and private experiments are
   excluded by `.gitignore`.
