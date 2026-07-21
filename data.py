"""
Data loading and preprocessing for NinaPro DB8.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy import signal
from scipy.interpolate import CubicSpline
from pathlib import Path
import scipy.io as sio


# ==================== Preprocessing ====================

def upsample_signal(emg, original_fs=1111, target_fs=2000):
    """Upsample using cubic spline interpolation."""
    n_channels, n_samples = emg.shape
    
    t_original = np.arange(n_samples) / original_fs
    duration = n_samples / original_fs
    n_samples_new = int(duration * target_fs)
    t_new = np.linspace(0, duration, n_samples_new)
    
    upsampled = np.zeros((n_channels, n_samples_new))
    for ch in range(n_channels):
        cs = CubicSpline(t_original, emg[ch, :])
        upsampled[ch, :] = cs(t_new)
    
    return upsampled


def bandpass_filter(emg, lowcut=20, highcut=450, fs=2000, order=4):
    """Apply Butterworth bandpass filter."""
    nyquist = fs / 2
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(order, [low, high], btype='band')
    
    filtered = np.zeros_like(emg)
    for ch in range(emg.shape[0]):
        filtered[ch, :] = signal.filtfilt(b, a, emg[ch, :])
    
    return filtered


def create_windows(emg, labels, window_size=4000, overlap=0.5):
    """Create sliding windows with majority vote labeling."""
    n_channels, n_samples = emg.shape
    step_size = int(window_size * (1 - overlap))
    
    windows = []
    window_labels = []
    
    for start in range(0, n_samples - window_size + 1, step_size):
        end = start + window_size
        window = emg[:, start:end]
        
        # Majority vote
        label_window = labels[start:end]
        unique, counts = np.unique(label_window, return_counts=True)
        majority_label = unique[np.argmax(counts)]
        
        windows.append(window)
        window_labels.append(majority_label)
    
    return np.array(windows), np.array(window_labels)


def normalize_signal(emg, method='zscore'):
    """Normalize per channel."""
    if method == 'zscore':
        mean = emg.mean(axis=-1, keepdims=True)
        std = emg.std(axis=-1, keepdims=True)
        std[std == 0] = 1
        return (emg - mean) / std
    elif method == 'minmax':
        min_val = emg.min(axis=-1, keepdims=True)
        max_val = emg.max(axis=-1, keepdims=True)
        range_val = max_val - min_val
        range_val[range_val == 0] = 1
        return (emg - min_val) / range_val
    else:
        raise ValueError(f"Unknown method: {method}")


def preprocess_ninapro(raw_emg, labels, config):
    """Full preprocessing pipeline."""
    # 1. Upsample
    emg = upsample_signal(raw_emg, config['raw_fs'], config['target_fs'])
    
    # 2. Filter
    emg = bandpass_filter(emg, config['bandpass_low'], config['bandpass_high'], 
                         config['target_fs'], config['filter_order'])
    
    # 3. Window
    window_size = int(config['window_size_sec'] * config['target_fs'])
    windows, window_labels = create_windows(emg, labels, window_size, config['window_overlap'])
    
    # 4. Normalize
    windows = normalize_signal(windows, config['normalization'])
    
    return windows, window_labels


# ==================== Dataset ====================

class EMGDataset(Dataset):
    """Simple PyTorch dataset for EMG windows."""
    def __init__(self, data, labels):
        self.data = torch.FloatTensor(data)
        self.labels = torch.LongTensor(labels)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def create_dataloaders(train_data, train_labels, val_data=None, val_labels=None, 
                       batch_size=128, num_workers=4):
    """Create train and optionally validation dataloaders."""
    train_dataset = EMGDataset(train_data, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, pin_memory=True)
    
    if val_data is not None:
        val_dataset = EMGDataset(val_data, val_labels)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                               num_workers=num_workers, pin_memory=True)
        return train_loader, val_loader
    
    return train_loader


# ==================== NinaPro Loading ====================

def load_ninapro_subject(subject_id, data_dir='data/ninapro_db8', config=None):
    """
    Load and preprocess NinaPro DB8 subject data.
    
    Returns:
        Dictionary with 'A1', 'A2', 'A3' keys containing windows and labels
    """
    subject_path = Path(data_dir) / f"S{subject_id}"
    subject_data = {}
    
    for acq_id in [1, 2, 3]:
        mat_file = subject_path / f"S{subject_id}_A{acq_id}_E1.mat"
        
        if not mat_file.exists():
            print(f"Warning: {mat_file} not found")
            continue
        
        # Load .mat file
        mat_data = sio.loadmat(str(mat_file))
        raw_emg = mat_data['emg'].T  # (channels, samples)
        labels = mat_data['restimulus'].flatten()
        
        # Preprocess
        if config is None:
            config = {
                'raw_fs': 1111,
                'target_fs': 2000,
                'bandpass_low': 20,
                'bandpass_high': 450,
                'filter_order': 4,
                'window_size_sec': 2.0,
                'window_overlap': 0.5,
                'normalization': 'zscore'
            }
        
        windows, window_labels = preprocess_ninapro(raw_emg, labels, config)
        
        subject_data[f'A{acq_id}'] = {
            'windows': windows,
            'labels': window_labels
        }
    
    return subject_data


def get_calibration_split(data, labels, calibration_percent=0.05, seed=42):
    """Split data into calibration and test sets."""
    np.random.seed(seed)
    n_samples = len(labels)
    n_cal = int(n_samples * calibration_percent)
    
    indices = np.random.permutation(n_samples)
    cal_idx = indices[:n_cal]
    test_idx = indices[n_cal:]
    
    return data[cal_idx], labels[cal_idx], data[test_idx], labels[test_idx]