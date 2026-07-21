# Pediatric Myoelectric Prosthetic Control with Hybrid TCN Architecture

This repository contains the implementation of a hybrid Temporal Convolutional Network (TCN) with multi-head spatial attention for pediatric myoelectric prosthetic control with minimal calibration requirements.

## Key Features

- **Hybrid Architecture**: Combines temporal feature extraction (TCN) with spatial attention across EMG channels
- **Continual Learning**: Implements Elastic Weight Consolidation (EWC) to prevent catastrophic forgetting
- **Minimal Calibration**: Achieves 81-86% accuracy with only 5-10% subject-specific data
- **Receptive Field Ablation**: Systematic study of temporal context requirements (150ms - 1000ms)

## Architecture Details

- **TCN Backbone**: 8-layer temporal convolutional network with exponential dilation
- **Receptive Field**: 510ms (1021 samples at 2000Hz)
- **Parameters**: ~1.9M total (TCN: 1.64M, Attention: 263k, Classifier: 33k)
- **Dataset**: NinaPro DB8 (12 subjects, 52 gestures, 16 EMG channels)

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended)
- 16GB+ RAM

### Step-by-Step Setup

```bash
# Clone repository
git clone https://github.com/yourusername/pediatric-myoelectric-tcn.git
cd pediatric-myoelectric-tcn

# Create conda environment
conda create -n myoelectric python=3.9
conda activate myoelectric

# Install PyTorch (adjust for your CUDA version)
# For CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CPU only:
# pip install torch torchvision torchaudio

# Install other dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .

# Verify installation
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
