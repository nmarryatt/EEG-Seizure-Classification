import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, SubsetRandomSampler

from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import (
    balanced_accuracy_score, 
    f1_score, 
    precision_score, 
    recall_score, 
    confusion_matrix, 
    classification_report
)

from scripts.train import train, evaluate
from scripts.CV import (
    normalize_fold_data,
    create_fold_dataloaders,
    calculate_class_weights,
    print_cv_summary,
    train_single_fold
)

def augment_eeg_signal(signal, aug_type):
    """
    Augment a single EEG signal with MORE AGGRESSIVE parameters
    """
    if aug_type == 'noise':
        noise_level = 0.05 * np.std(signal)
        noise = np.random.normal(0, noise_level, signal.shape)
        return signal + noise
    
    elif aug_type == 'scale':
        scale_factor = np.random.uniform(0.85, 1.15)
        return signal * scale_factor
    
    elif aug_type == 'shift':
        shift = np.random.randint(-100, 100)
        return np.roll(signal, shift)
    
    elif aug_type == 'warp':
        warp_factor = np.random.uniform(0.96, 1.05)
        indices = np.arange(len(signal))
        warped_indices = (indices * warp_factor).astype(int) % len(signal)
        return signal[warped_indices]
    
    else:
        return signal
    

def create_augmented_dataset(X_seizure, y_seizure, target_count=150, 
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


def visualize_augmentation_comparison(X_original, sample_idx=0, start_time=5, duration=5):
    """
    Show original vs augmented for a zoomed-in segment
    
    Args:
        X_original: Original data
        sample_idx: Which sample to use
        start_time: Start time in seconds (default: 5s)
        duration: How many seconds to show (default: 5s)
    """
    fig, axes = plt.subplots(4, 1, figsize=(16, 12))
    
    original_signal = X_original[sample_idx]
    
    # Calculate sample indices for the window
    fs = 173.61  # Sampling rate
    start_idx = int(start_time * fs)
    end_idx = int((start_time + duration) * fs)
    
    # Extract windowed signal
    original_window = original_signal[start_idx:end_idx]
    time_axis = np.arange(len(original_window)) / fs + start_time  # Offset by start_time
    
    aug_methods = ['noise', 'scale', 'shift', 'warp']
    aug_labels = {
        'noise': 'Gaussian Noise Addition',
        'scale': 'Amplitude Scaling (±15%)', 
        'shift': 'Temporal Shift',
        'warp': 'Time Warping'
    }
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
    
    for i, (method, color) in enumerate(zip(aug_methods, colors)):
        # Augment the FULL signal first
        augmented_full = augment_eeg_signal(original_signal.copy(), aug_type=method)
        # Then extract the same window
        augmented_window = augmented_full[start_idx:end_idx]
        
        # Plot augmented in color
        axes[i].plot(time_axis, augmented_window, color=color, linewidth=2.0, 
                     alpha=0.8, label='Augmented')
        
        # Plot original in black on top
        axes[i].plot(time_axis, original_window, 'black', linewidth=2.0, 
                     alpha=0.9, label='Original')
        
        axes[i].set_title(aug_labels[method], fontweight='bold', fontsize=13)
        axes[i].set_xlabel('Time (s)', fontsize=10)
        axes[i].set_ylabel('Amplitude (μV)', fontsize=10)
        axes[i].legend(loc='upper right', fontsize=9)
        axes[i].grid(True, alpha=0.3, linestyle='--')
        axes[i].set_xlim([start_time, start_time + duration])
    
    plt.suptitle(f'EEG Data Augmentation Techniques (Zoomed: {start_time}-{start_time+duration}s)', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('augmentation_comparison_zoomed.png', dpi=300, bbox_inches='tight')
    plt.show()

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split


def tune_hyperparameters_on_fold(X_train_val, y_train_val, model_class, device='cuda', 
                                   n_trials=3, max_epochs=15):
    """
    Hyperparameter tuning for a single fold using random search
    
    Args:
        X_train_val: Training data for this fold (NOT normalized)
        y_train_val: Training labels for this fold
        model_class: Model class to instantiate
        device: 'cuda' or 'cpu'
        n_trials: Number of random search trials
        max_epochs: Epochs per trial (keep low for speed)
        
    Returns:
        Dictionary with best hyperparameters
    """
    
    # Hyperparameter search space
    param_options = {
        'learning_rate': [1e-4, 5e-4, 1e-3],
        'dropout': [0.3, 0.4, 0.5],
        'batch_size': [16, 32],
        'weight_decay': [1e-5, 1e-4]
    }
    
    best_params = None
    best_val_acc = 0
    
    print(f"  Tuning hyperparameters ({n_trials} random trials)...")
    
    for trial in range(n_trials):
        # Randomly sample parameters - IMPORTANT: convert to native Python types!
        params = {
            'learning_rate': float(np.random.choice(param_options['learning_rate'])),
            'dropout': float(np.random.choice(param_options['dropout'])),
            'batch_size': int(np.random.choice(param_options['batch_size'])),  # MUST be Python int
            'weight_decay': float(np.random.choice(param_options['weight_decay']))
        }
        
        # Split into train and validation for tuning
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=0.2, 
            random_state=trial, stratify=y_train_val
        )
        
        # Normalize using training stats
        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0) + 1e-8
        X_train_norm = (X_train - mean) / std
        X_val_norm = (X_val - mean) / std
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train_norm)
        y_train_tensor = torch.LongTensor(y_train)
        X_val_tensor = torch.FloatTensor(X_val_norm)
        y_val_tensor = torch.LongTensor(y_val)
        
        # Create dataloaders
        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = torch.utils.data.TensorDataset(X_val_tensor, y_val_tensor)
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=params['batch_size'], shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=params['batch_size'], shuffle=False
        )
        
        # Calculate class weights
        class_counts = np.bincount(y_train)
        class_weights = torch.FloatTensor([1.0 / count for count in class_counts]).to(device)
        
        # Initialize model
        model = model_class(input_length=4097, dropout=params['dropout']).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(
            model.parameters(), 
            lr=params['learning_rate'],
            weight_decay=params['weight_decay']
        )
        
        # Quick training
        best_trial_acc = 0
        
        for epoch in range(max_epochs):
            # Train
            model.train()
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
            
            # Validate
            model.eval()
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    outputs = model(X_batch)
                    _, predicted = torch.max(outputs, 1)
                    val_total += y_batch.size(0)
                    val_correct += (predicted == y_batch).sum().item()
            
            val_acc = val_correct / val_total
            best_trial_acc = max(best_trial_acc, val_acc)
        
        print(f"    Trial {trial+1}/{n_trials}: LR={params['learning_rate']:.0e}, "
              f"BS={params['batch_size']}, Dropout={params['dropout']:.2f}, "
              f"WD={params['weight_decay']:.0e} → Val Acc: {best_trial_acc:.4f}")
        
        # Track best parameters
        if best_trial_acc > best_val_acc:
            best_val_acc = best_trial_acc
            best_params = params.copy()
    
    print(f"  → Best params: LR={best_params['learning_rate']:.0e}, "
          f"BS={best_params['batch_size']}, Dropout={best_params['dropout']:.2f}, "
          f"WD={best_params['weight_decay']:.0e} (Tuning Val Acc: {best_val_acc:.4f})")
    
    return best_params



def train_with_kfold_cv_augmented(X_data, y_data, model_class, n_splits=5, 
                                   max_epochs=50, batch_size=32, device='cuda',
                                   patience=5, min_delta=0.001, model_kwargs=None,
                                   use_augmentation=True, 
                                   aug_methods=['noise', 'scale', 'shift', 'warp'],
                                   tune_hyperparams=False, n_tuning_trials=3):
    """
    Train model with k-fold CV and optional data augmentation
    
    Args:
        X_data: Full dataset (500, 4097) - filtered but NOT normalized
        y_data: Full labels (500,)
        model_class: Model class to instantiate
        n_splits: Number of folds
        max_epochs: Maximum epochs per fold
        batch_size: Batch size (ignored if tune_hyperparams=True)
        device: 'cuda' or 'cpu'
        patience: Early stopping patience
        min_delta: Minimum improvement threshold
        model_kwargs: Dict of kwargs for model initialization
        use_augmentation: Whether to augment training data
        aug_methods: List of augmentation methods to use
        tune_hyperparams: Whether to tune hyperparameters per fold (nested CV)
        n_tuning_trials: Number of random search trials per fold
        
    Returns:
        Dictionary with results for all folds
    """
    if model_kwargs is None:
        model_kwargs = {'input_length': 4097}
    
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    all_results = {
        'fold_histories': [],
        'best_val_accs': [],
        'best_val_losses': [],
        'fold_best_params': [] if tune_hyperparams else None
    }
    
    print(f"Starting {n_splits}-Fold Cross-Validation")
    if tune_hyperparams:
        print(f"Using NESTED CV with hyperparameter tuning ({n_tuning_trials} trials/fold)")
    if use_augmentation:
        print(f"Using augmentation: {aug_methods}")
    print("=" * 60)
    
    for fold, (train_ids, val_ids) in enumerate(kfold.split(X_data), 1):
        print(f"\nFOLD {fold}/{n_splits}")
        print("-" * 60)
        
        # Split data
        X_train_fold = X_data[train_ids]
        X_val_fold = X_data[val_ids]
        y_train_fold = y_data[train_ids]
        y_val_fold = y_data[val_ids]
        
        # Hyperamater tuning (if enabled)
        if tune_hyperparams:
            best_params = tune_hyperparameters_on_fold(
                X_train_fold, y_train_fold, model_class, 
                device=device, n_trials=n_tuning_trials
            )
            all_results['fold_best_params'].append(best_params)
            
            # Update model_kwargs and optimizer params with tuned values
            model_kwargs['dropout'] = best_params['dropout']
            fold_batch_size = best_params['batch_size']
            fold_lr = best_params['learning_rate']
            fold_weight_decay = best_params['weight_decay']
        else:
            # Use default/provided hyperparameters
            fold_batch_size = batch_size
            fold_lr = 1e-3
            fold_weight_decay = 1e-4
        
        # Augmentation - only on training data!
        if use_augmentation:
            # Separate seizure and non-seizure in training set
            train_seizure_mask = y_train_fold == 1
            X_train_seizure = X_train_fold[train_seizure_mask]
            X_train_non_seizure = X_train_fold[~train_seizure_mask]
            y_train_seizure = y_train_fold[train_seizure_mask]
            y_train_non_seizure = y_train_fold[~train_seizure_mask]
            
            print(f"  Before augmentation: {len(X_train_seizure)} seizures, {len(X_train_non_seizure)} non-seizures")
            
            # Augment seizures using all methods
            X_train_seizure_aug, y_train_seizure_aug = create_augmented_dataset(
                X_train_seizure, y_train_seizure, aug_methods=aug_methods, target_count = 150
            )
            
            # Combine augmented seizures with non-seizures
            X_train_fold = np.vstack([X_train_seizure_aug, X_train_non_seizure])
            y_train_fold = np.hstack([y_train_seizure_aug, y_train_non_seizure])
            
            # Shuffle augmented training data
            shuffle_idx = np.random.permutation(len(X_train_fold))
            X_train_fold = X_train_fold[shuffle_idx]
            y_train_fold = y_train_fold[shuffle_idx]
            
            print(f"  After augmentation: {len(X_train_seizure_aug)} seizures, {len(X_train_non_seizure)} non-seizures")
        # Normalize (using training statistics only)
        X_train_norm, X_val_norm = normalize_fold_data(X_train_fold, X_val_fold)
        
        # Create dataloaders (with tuned or default batch size)
        train_loader, val_loader = create_fold_dataloaders(
            X_train_norm, X_val_norm, y_train_fold, y_val_fold, fold_batch_size
        )
        
        # Calculate class weights (on augmented training data)
        class_weights = calculate_class_weights(y_train_fold, device)
        
        # Initialize model and training components
        model = model_class(**model_kwargs).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(model.parameters(), lr=fold_lr, weight_decay=fold_weight_decay)
        
        # Train this fold
        fold_results = train_single_fold(
            model, train_loader, val_loader, criterion, optimizer,
            device, max_epochs, patience, min_delta, fold
        )
        
        # Store results
        all_results['fold_histories'].append(fold_results)
        all_results['best_val_accs'].append(fold_results['best_val_acc'])
        all_results['best_val_losses'].append(fold_results['best_val_loss'])
        
        print(f"  Fold {fold} Summary:")
        print(f"    Best Val Acc: {fold_results['best_val_acc']:.4f} at epoch {fold_results['best_epoch']+1}")
        print(f"    Total epochs: {fold_results['total_epochs']}")
    
    # Print overall summary
    print_cv_summary(all_results, n_splits)
    
    # Print hyperparameter summary if tuning was done
    if tune_hyperparams:
        print("\n" + "=" * 60)
        print("HYPERPARAMETER TUNING SUMMARY")
        print("=" * 60)
        print("\nBest parameters per fold:")
        for i, params in enumerate(all_results['fold_best_params'], 1):
            print(f"  Fold {i}: LR={params['learning_rate']:.0e}, BS={params['batch_size']}, "
                  f"Dropout={params['dropout']:.2f}, WD={params['weight_decay']:.0e}")
        
        # Most common hyperparameters across folds
        print("\nRecommended hyperparameters (most common):")
        param_keys = all_results['fold_best_params'][0].keys()
        recommended = {}
        for key in param_keys:
            values = [params[key] for params in all_results['fold_best_params']]
            recommended[key] = max(set(values), key=values.count)
        
        print(f"  Learning Rate: {recommended['learning_rate']:.0e}")
        print(f"  Batch Size: {recommended['batch_size']}")
        print(f"  Dropout: {recommended['dropout']:.2f}")
        print(f"  Weight Decay: {recommended['weight_decay']:.0e}")
        print("=" * 60)
    
    return all_results


