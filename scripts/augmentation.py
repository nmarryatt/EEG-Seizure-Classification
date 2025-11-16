import numpy as np
import matplotlib.pyplot as plt

def augment_eeg_signal(signal, aug_type='noise'):
    """
    Augment a single EEG signal
    Args:
        signal: 1D array (4097,)
        aug_type: type of augmentation
    Returns:
        Augmented signal
    """
    if aug_type == 'noise':
        # Add small Gaussian noise (SNR ~20-30 dB)
        noise_level = 0.05 * np.std(signal)
        noise = np.random.normal(0, noise_level, signal.shape)
        return signal + noise
    
    elif aug_type == 'scale':
        # Random amplitude scaling
        scale_factor = np.random.uniform(0.85, 1.15)
        return signal * scale_factor
    
    elif aug_type == 'shift':
        # Time shift (circular)
        shift = np.random.randint(-100, 100)
        return np.roll(signal, shift)
    
    elif aug_type == 'warp':
        # Simple time warping
        warp_factor = np.random.uniform(0.95, 1.05)
        indices = np.arange(len(signal))
        warped_indices = (indices * warp_factor).astype(int) % len(signal)
        return signal[warped_indices]
    
    else:
        return signal


def create_augmented_dataset(X_seizure, y_seizure, target_count=400, 
                             aug_methods=['noise', 'scale', 'shift']):
    """
    Augment seizure samples to balance the dataset
    
    Args:
        X_seizure: Seizure samples (100, 4097)
        y_seizure: Seizure labels (100,)
        target_count: How many total seizure samples you want
        aug_methods: List of augmentation methods to use
    
    Returns:
        X_augmented: All seizure samples (original + augmented)
        y_augmented: Corresponding labels
    """
    n_original = len(X_seizure)
    n_to_generate = target_count - n_original
    
    X_augmented = [X_seizure]  # Start with originals
    y_augmented = [y_seizure]
    
    print(f"Original seizure samples: {n_original}")
    print(f"Generating {n_to_generate} augmented samples...")
    
    while len(np.vstack(X_augmented)) < target_count:
        # Randomly select a seizure sample
        idx = np.random.randint(0, n_original)
        sample = X_seizure[idx]
        
        # Randomly select augmentation method
        aug_method = np.random.choice(aug_methods)
        
        # Augment
        aug_sample = augment_eeg_signal(sample, aug_method)
        
        X_augmented.append(aug_sample.reshape(1, -1))
        y_augmented.append(np.array([1]))  # Seizure label
    
    X_final = np.vstack(X_augmented)[:target_count]
    y_final = np.hstack(y_augmented)[:target_count]
    
    print(f"Final augmented seizure count: {len(X_final)}")
    
    return X_final, y_final



def visualize_augmentation_comparison(X_original, sample_idx=0):
    """
    Show original vs augmented side-by-side for easy comparison
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 8))
    axes = axes.flatten()
    
    original_signal = X_original[sample_idx]
    time_axis = np.arange(len(original_signal)) / 173.61
    
    aug_methods = ['noise', 'scale', 'shift', 'warp']
    aug_labels = {
        'noise': 'Gaussian Noise Addition',
        'scale': 'Amplitude Scaling (±15%)', 
        'shift': 'Temporal Shift',
        'warp': 'Time Warping'
    }
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
    
    for i, (method, color) in enumerate(zip(aug_methods, colors)):
        augmented = augment_eeg_signal(original_signal.copy(), aug_type=method)
        
        # Plot both on same axis for comparison
        axes[i].plot(time_axis, original_signal, 'black', linewidth=1.2, 
                     alpha=0.5, label='Original')
        axes[i].plot(time_axis, augmented, color=color, linewidth=1.2, 
                     alpha=0.9, label='Augmented')
        
        axes[i].set_title(aug_labels[method], fontweight='bold', fontsize=13)
        axes[i].set_xlabel('Time (s)', fontsize=10)
        axes[i].set_ylabel('Amplitude (μV)', fontsize=10)
        axes[i].legend(loc='upper right', fontsize=9)
        axes[i].grid(True, alpha=0.3, linestyle='--')
        axes[i].set_xlim([0, time_axis[-1]])
    
    plt.suptitle('EEG Data Augmentation Techniques', 
                 fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig('augmentation_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


