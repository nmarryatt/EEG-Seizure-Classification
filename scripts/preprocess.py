import os
import numpy as np
from scipy.signal import butter, filtfilt
import numpy as np


def lowpass_filter(signal, fs=173.61, cutoff=40):
    """
    Apply a low-pass Butterworth filter to a 1D EEG signal.

    Parameters:
    - signal: 1D numpy array of EEG samples
    - fs: sampling rate in Hz (default 173.61 for Bonn EEG)
    - cutoff: cutoff frequency in Hz (default 40)

    Returns:
    - filtered_signal: 1D numpy array
    """
    # Butterworth filter coefficients
    b, a = butter(N=4, Wn=cutoff/(0.5*fs), btype='low')
    
    # Apply filter with zero-phase distortion
    filtered_signal = filtfilt(b, a, signal)
    
    return filtered_signal


def load_and_preprocess(base_path, folders):
    """
    Load EEG segments from folders and apply preprocessing.
    
    Returns:
        X: numpy array of EEG segments
        y: numpy array of labels
    """
    X, y = [], []

    for folder, label in folders.items():
        folder_path = os.path.join(base_path, folder)
        for fname in os.listdir(folder_path):
            if fname.lower().endswith(".txt"):
                file_path = os.path.join(folder_path, fname)
                eeg_segment = np.loadtxt(file_path)
                eeg_segment = lowpass_filter(eeg_segment)
                X.append(eeg_segment)
                y.append(label)

    return np.array(X), np.array(y)



def normalise_segments(X_train, X_test):
    """
    Normalize EEG segments using global mean and standard deviation
    computed from the training set.

    Parameters:
    - X_train: list or array of training segments (each segment: 1D numpy array)
    - X_test: list or array of test segments (each segment: 1D numpy array)

    Returns:
    - X_train_norm: list of normalized training segments
    - X_test_norm: list of normalized test segments
    """
    # Concatenate all training segments to compute global mean and std
    training_all = np.concatenate(X_train, axis=0)
    training_mean = np.mean(training_all, axis=0)
    training_std = np.std(training_all, axis=0)

    # Normalize training and test segments
    X_train_norm = [(segment - training_mean) / training_std for segment in X_train]
    X_test_norm = [(segment - training_mean) / training_std for segment in X_test]

    return X_train_norm, X_test_norm



def normalize_fold_data(X_train, X_val):
    """
    Normalize validation data using training statistics
    
    Args:
        X_train: Training data
        X_val: Validation data
        
    Returns:
        Normalized X_train, X_val
    """
    train_mean = X_train.mean()
    train_std = X_train.std()
    
    X_train_norm = (X_train - train_mean) / train_std
    X_val_norm = (X_val - train_mean) / train_std
    
    return X_train_norm, X_val_norm


def create_fold_dataloaders(X_train, X_val, y_train, y_val, batch_size=32):
    """
    Create PyTorch dataloaders for a fold
    
    Args:
        X_train, X_val: Normalized data arrays
        y_train, y_val: Label arrays
        batch_size: Batch size
        
    Returns:
        train_loader, val_loader
    """
    # Convert to tensors with channel dimension
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).unsqueeze(1)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)
    
    # Create datasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader


def calculate_class_weights(y_train, device):
    """
    Calculate class weights for imbalanced data
    
    Args:
        y_train: Training labels
        device: torch device
        
    Returns:
        Class weights tensor
    """
    class_counts = np.bincount(y_train)
    class_weights = 1.0 / class_counts
    return torch.tensor(class_weights, dtype=torch.float32).to(device)
