# Pediatric Myoelectric Prosthetic Control with Hybrid TCN Architecture

Hybrid Temporal Convolutional Network (TCN) + multi-head spatial attention for pediatric myoelectric prosthetic control with minimal calibration.

## Key Features

- 81-86% accuracy with only 5-10% subject-specific calibration data
- 8-layer TCN with 510ms receptive field (1.9M parameters)
- Elastic Weight Consolidation (EWC) prevents catastrophic forgetting
- Validated on NinaPro DB8 (12 subjects, 52 gestures, 16 EMG channels)

## Quick Start

```bash
# Install
git clone https://github.com/avaril22/myoelectric-hybrid-architecture.git
cd myoelectric-hybrid-architecture
pip install -r requirements.txt
pip install -e .

# Download NinaPro DB8 to data/ninapro_db8/ (https://ninapro.hevs.ch/instructions/DB8.html)

# Run quick test (1 subject, 3 configs)
bash scripts/run_rf_ablation_quick.sh

# Train full model
python experiments/run_hybrid.py --subjects 1 2 3 4 5 6 7 8 9 10 11 12 --calibration 0.05
```
## Architecture
EMG (16 channels, 2000Hz) 
  → Per-Channel TCN (8 layers, dilations [1,2,4,8,16,32,64,128])
  → Multi-Head Spatial Attention (4 heads)
  → Global Avg Pool → FC Classifier → 52 classes

## Repository Structure
├── src/models/          # TCN, attention, hybrid architecture <br>
├── src/training/        # EWC, training loops  <br>
├── src/utils/           # Preprocessing, data loading <br>
├── experiments/         # Training scripts, ablation study <br>
├── configs/             # Model hyperparameters (YAML) <br>
├── tests/               # Unit tests for RF calculation <br>
└── notebooks/           # Results analysis<br>
