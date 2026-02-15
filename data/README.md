# Data Directory

## Data Sources

### Workload-1: Synthetic Signal Data
- Generated in code via `src/utils/data.py`
- No manual download needed
- Configurable: num_samples, seq_len, num_classes, noise_level
- Reproducible with fixed random seed

### Workload-2: CIFAR-10
- Downloaded automatically via `torchvision.datasets.CIFAR10`
- Will be saved to `data/cifar10/` on first run
- 60,000 images (50K train + 10K test), 32x32 RGB, 10 classes
- Total size: ~170MB

## Usage

Data loading is handled by `src/utils/data.py`. You do NOT need to manually
download anything. Just run the training script and data will be prepared
automatically.

```python
from src.utils.data import get_cifar10_loaders, get_synthetic_signal_loaders

# CIFAR-10
train_loader, test_loader = get_cifar10_loaders(batch_size=64)

# Synthetic signal
train_loader, test_loader = get_synthetic_signal_loaders(batch_size=64)
```
