# CM4126: Working with Images using TorchVision (CIFAR-10)

A guided lab notebook for the CM4126 lecture **"Working with Images using TorchVision"**. You run it top to bottom and it explains every step. You don't need any PyTorch experience.

## What it covers

| Section | Topic |
|---|---|
| 1 | Tensors: images as grids of numbers, colour channels, `[C, H, W]` vs `[H, W, C]` |
| 2 | Transforms: `ToTensor`, resize/crop/pad, normalisation, augmentation, train vs test pipelines |
| 3 | `Dataset` and `DataLoader`, class balance, a proper validation split |
| 4 | Pretrained ResNet-18: making predictions, looking inside the model |
| 5 | Transfer learning: feature extraction vs fine-tuning, saving and loading |
| 6 | Evaluation: confusion matrix, per-class metrics, class imbalance |
| 7 | Best practices: overfit one batch, learning rates, schedulers, early stopping, Optuna (optional) |
| 8 | Beyond classification: bounding boxes and a pretrained detector (optional) |

The saved notebook already includes its outputs, so you can read it before running anything.

## Setup

1. Install Python 3.10 or newer, and VS Code with the **Jupyter** extension.
2. Clone the repository and open this folder in a terminal:
   `git clone https://github.com/heyad/Teaching.git && cd Teaching/TorchVision-CIFAR10`
3. Create a virtual environment:
   - macOS / Linux: `python3 -m venv .venv && source .venv/bin/activate`
   - Windows: `python -m venv .venv` then `.venv\Scripts\activate`
4. Install the packages: `pip install -r requirements.txt`
   (For an NVIDIA GPU, first install PyTorch using the command from [pytorch.org](https://pytorch.org/get-started/locally/).)
5. Register the kernel: `python -m ipykernel install --user --name cm4126-torchvision --display-name "CM4126 TorchVision"`
6. Open `torchvision_cifar10_walkthrough.ipynb`, choose the **CM4126 TorchVision** kernel (top right), and click **Run All**, or run the cells one by one.

The first run downloads CIFAR-10 (about 170 MB) into `./data` and the ResNet-18 weights (about 45 MB) into your PyTorch cache. Later runs reuse them.

**SSL errors on macOS** (`CERTIFICATE_VERIFY_FAILED`): run `/Applications/Python 3.x/Install Certificates.command`, or start Jupyter with `export SSL_CERT_FILE=$(python -m certifi)`.

## Runtime

Set `QUICK_RUN` in the configuration cell near the top.

| Mode | Data used | Expected runtime |
|---|---|---|
| `QUICK_RUN = True` (default) | 5k train / 1k val / 2k test, 128 × 128 px, 2 + 2 epochs | measured: about 1.5 min on an Apple M3 Max GPU (MPS), about 10 min on the same machine's CPU only; older laptop CPUs will be slower |
| `QUICK_RUN = False` | 45k train / 5k val / 10k test, 224 × 224 px, 5 + 5 epochs | well over an hour on a CPU; use a GPU (lab machine or Google Colab) |
