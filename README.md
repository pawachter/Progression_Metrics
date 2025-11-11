# Training with Domain Adaptation Metrics

Training pipeline for evaluating domain shift between synthetic and real image datasets. The goal is to identify metrics that can guide optimal mixing ratios when combining synthetic (source) and real (target) data within training batches. Currently, the focus is on metrics that estimate the current progress or status of the training process.

## Status
✅ **Ready to Use** - All core components implemented and functional

## Overview

This project trains DNNs while continuously monitoring three categories of metrics to understand domain shift:

1. **Target Domain Performance** - How well the model performs on real data
2. **Entropy Gap** - Difference in prediction uncertainty between synthetic and real samples  
3. **Representation Mismatch** - Distribution distances across network layers (MMD, KL divergence, Wasserstein)

These metrics will eventually inform a dynamic batch composition strategy.

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### Running Training

```bash
# Update DATA_ROOT_DIR in main.py to point to your dataset
python main.py
```

### Enable Domain Adaptation Tracking

Edit `config.yaml` and set:
```yaml
domain_adaptation:
  enabled: true  # Change from false to true
```

## Project Structure

```
├── main.py                          # Training orchestration
├── callbacks.py                     # All training callbacks (convergence + domain adaptation)
├── distance_metrics.py              # MMD, KL, Wasserstein implementations
├── model_creator.py                 # Model architecture definitions
├── trainer.py                       # Custom training loop
├── load_and_create_datasets.py      # Dataset loading and preprocessing
└── config.yaml                      # Hyperparameters and metric settings
```

## Configuration

Edit `config.yaml` to adjust training parameters and metric computation frequency:

```yaml
training:
  epochs: 50
  batch_size: 32
  learning_rate: 0.001

domain_adaptation:
  enabled: true
  n_steps: 100          # Compute metrics every N steps
  subset_size: 500      # Held-out samples for metric computation
```

## Metrics

All metrics are computed on held-out subsets every N training steps:

- **Target Gap**: Tracks validation loss on real data vs. best achieved
- **Entropy Gap**: `H_real - H_synthetic` measures prediction confidence difference
- **Representation Mismatch**: 9 distance measurements (3 metrics × 3 layer depths)

Logs are written to:
- `logs/training.log` - Gradient analysis, Fisher trace, activation saturation
- `logs/domain_adaptation.log` - All domain shift metrics

## Dataset Structure

Expected format:
```
data/
├── REAL/
│   ├── 0/*.jpg  (or class_0, label_0, etc.)
│   ├── 1/*.jpg
│   ├── 2/*.jpg
│   ├── ...
│   └── 9/*.jpg
└── FAKE/
    ├── 0/*.jpg
    ├── 1/*.jpg
    ├── 2/*.jpg
    ├── ...
    └── 9/*.jpg
```

- Each image type (REAL/FAKE) has subdirectories for each class
- Class labels are automatically inferred from subdirectory names using TensorFlow's `image_dataset_from_directory`
- Subdirectories are sorted alphanumerically to determine class indices
- Supports any valid folder names (e.g., `0`, `class_0`, `label_0`) - labels are inferred from sort order

### DataLoader Features

The DataLoader now uses **TensorFlow's best practices**:
- ✅ Built-in `image_dataset_from_directory` API (faster, simpler, more reliable)
- ✅ Automatic label inference from directory structure
- ✅ Efficient batching and prefetching with `tf.data.AUTOTUNE`
- ✅ Proper caching and shuffling for optimal performance
- ✅ Reproducible dataset splits with seed control

## Callbacks

The `CallbackFactory` provides unified access to all monitoring:

```python
from callbacks import CallbackFactory

factory = CallbackFactory(config_path='config.yaml')

# Standard training callbacks
callbacks = factory.create_standard_callbacks()

# Convergence monitoring  
callbacks.extend(factory.create_all_convergence_callbacks(val_dataset, test_dataset))

# Domain adaptation metrics
callbacks.extend(factory.create_all_domain_adaptation_callbacks(
    dataset_a_subset, dataset_b_subset, n_steps=100
))
```

## Current Focus

Determining which metrics are most predictive of:
- When to increase real data proportion in batches
- When synthetic data has diminishing returns
- Optimal switching points during training

## Requirements

- TensorFlow >= 2.10.0
- NumPy >= 1.23.0
- SciPy >= 1.9.0
- PyYAML >= 6.0

## Notes

- Batch mixing strategy not yet implemented (next phase)
- Metrics are being validated for correlation with downstream task performance