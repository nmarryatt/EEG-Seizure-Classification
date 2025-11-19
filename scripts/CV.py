import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, SubsetRandomSampler
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
from scripts.train import train, evaluate


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


def train_single_fold(model, train_loader, val_loader, criterion, optimizer, 
                     device, max_epochs=50, patience=10, min_delta=0.001, 
                     fold_num=1):
    """
    Train a single fold with early stopping
    
    Args:
        model: PyTorch model
        train_loader, val_loader: Data loaders
        criterion: Loss function
        optimizer: Optimizer
        device: torch device
        max_epochs: Maximum epochs
        patience: Early stopping patience
        min_delta: Minimum improvement threshold
        fold_num: Fold number for logging
        
    Returns:
        Dictionary with training history and best metrics
    """
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    best_val_loss = float('inf')
    best_val_acc = 0
    best_epoch = 0
    patience_counter = 0
    
    for epoch in range(max_epochs):
        # Training
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        
        # Validation
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        # Record metrics
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Early stopping check
        if val_loss < (best_val_loss - min_delta):
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), f'best_model_fold{fold_num}.pth')
        else:
            patience_counter += 1
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{max_epochs} | "
                  f"Train: {train_loss:.4f}/{train_acc:.4f} | "
                  f"Val: {val_loss:.4f}/{val_acc:.4f} | "
                  f"Patience: {patience_counter}/{patience}")
        
        # Check early stopping
        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break
    
    return {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'val_losses': val_losses,
        'val_accs': val_accs,
        'best_val_loss': best_val_loss,
        'best_val_acc': best_val_acc,
        'best_epoch': best_epoch,
        'total_epochs': len(train_losses)
    }

def print_cv_summary(results, n_splits):
    """Print cross-validation summary statistics"""
    print("\n" + "=" * 60)
    print("CROSS-VALIDATION SUMMARY")
    print("=" * 60)
    
    best_accs = results['best_val_accs']
    print(f"Best Val Accuracy per fold:")
    for i, acc in enumerate(best_accs, 1):
        print(f"  Fold {i}: {acc:.4f}")
    
    print(f"\nMean Val Accuracy: {np.mean(best_accs):.4f} ± {np.std(best_accs):.4f}")
    print(f"Min Val Accuracy:  {np.min(best_accs):.4f}")
    print(f"Max Val Accuracy:  {np.max(best_accs):.4f}")



    # 5-fold CV
def train_with_kfold_cv(X_data, y_data, model_class, n_splits=5, max_epochs=50, 
                       batch_size=32, device='cuda', patience=10, min_delta=0.001,
                       model_kwargs=None):
    """
    Train model with k-fold cross-validation
    
    Args:
        X_data: Full dataset (n_samples, 4097) - filtered but NOT normalized
        y_data: Full labels (n_samples,)
        model_class: Model class to instantiate (e.g., simple_EEG_CNN)
        n_splits: Number of folds
        max_epochs: Maximum epochs per fold
        batch_size: Batch size
        device: 'cuda' or 'cpu'
        patience: Early stopping patience
        min_delta: Minimum improvement threshold
        model_kwargs: Dict of kwargs for model initialization
        
    Returns:
        Dictionary with results for all folds
    """
    if model_kwargs is None:
        model_kwargs = {'input_length': 4097}
    
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    all_results = {
        'fold_histories': [],
        'best_val_accs': [],
        'best_val_losses': []
    }
    
    print(f"Starting {n_splits}-Fold Cross-Validation")
    print("=" * 60)
    
    for fold, (train_ids, val_ids) in enumerate(kfold.split(X_data), 1):
        print(f"\nFOLD {fold}/{n_splits}")
        print("-" * 60)
        
        # Split data
        X_train_fold = X_data[train_ids]
        X_val_fold = X_data[val_ids]
        y_train_fold = y_data[train_ids]
        y_val_fold = y_data[val_ids]
        
        # Normalize
        X_train_norm, X_val_norm = normalize_fold_data(X_train_fold, X_val_fold)
        
        # Create dataloaders
        train_loader, val_loader = create_fold_dataloaders(
            X_train_norm, X_val_norm, y_train_fold, y_val_fold, batch_size
        )
        
        # Calculate class weights
        class_weights = calculate_class_weights(y_train_fold, device)
        
        # Initialize model and training components
        model = model_class(**model_kwargs).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        
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
    
    return all_results

