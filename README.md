# ACIDNet

Official implementation of ACIDNet for low-light image enhancement.

This repository contains the core model, dataset loaders, training entrypoint,
inference scripts, and metric tools required to reproduce the main ACIDNet
experiments. Datasets and generated outputs are intentionally kept outside
version control.

## Installation

```bash
conda create -n acidnet python=3.9
conda activate acidnet
pip install -r requirements.txt
```

Install a PyTorch build that matches your CUDA version before running GPU
training or evaluation.

## Data

Set `ACIDNET_DATASET_ROOT` to the directory containing the benchmark datasets:

```bash
export ACIDNET_DATASET_ROOT=/path/to/datasets
```

Expected dataset layout:

```text
$ACIDNET_DATASET_ROOT/
  LOLdataset/
    our485/
    eval15/low/
    eval15/high/
  LOLv2/
    Real_captured/
      Train/
      Test/Low/
      Test/Normal/
    Synthetic/
      Train/
      Test/Low/
      Test/Normal/
```

## Evaluation

Run benchmark inference and metrics:

```bash
python -m eval.benchmark --dataset-root "$ACIDNET_DATASET_ROOT" --paired-only
```

Run a quick smoke test on one image per dataset:

```bash
python -m eval.benchmark --dataset-root "$ACIDNET_DATASET_ROOT" --paired-only --limit 1
```

Results are written to `outputs/benchmark_metrics.json` and
`outputs/benchmark_metrics.csv`.

## Training

Check that the training pipeline can initialize:

```bash
python -m train.train --dataset lol_v1 --dry_run
```

Start training:

```bash
python -m train.train --dataset lol_v1
```

Checkpoints are written under `outputs/checkpoints/`.

## Metrics

Calculate paired metrics from existing predictions:

```bash
python -m measure.paired_metrics \
  --pred-dir outputs/benchmark/LOLv2_syn \
  --gt-dir "$ACIDNET_DATASET_ROOT/LOLv2/Synthetic/Test/Normal"
```

Calculate unpaired metrics:

```bash
python -m measure.unpaired_metrics --pred-dir outputs/benchmark/DICM
```

## Repository Layout

```text
acidnet/      Core ACIDNet package: model, data loaders, losses, shared paths
train/        Training entrypoint
eval/         Inference and benchmark evaluation entrypoints
measure/      Metric scripts
docs/         Public documentation and figures
```

Large files, generated outputs, local datasets, private experiments, and
analysis scripts are ignored by default.
