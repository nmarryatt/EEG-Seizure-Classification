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

                # Trim so it's divisible by 4
                n = len(eeg_segment)
                n_trim = n - (n % 4)
                eeg_segment = eeg_segment[:n_trim]

                part_len = n_trim // 4

                # Split into 4 equal parts
                for i in range(4):
                    start = i * part_len
                    end = (i + 1) * part_len
                    X.append(eeg_segment[start:end])
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
