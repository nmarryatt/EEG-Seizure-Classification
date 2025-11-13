import torch
from torch.utils.data import DataLoader, TensorDataset, SubsetRandomSampler
import numpy as np
from scripts.train import train
from scripts.train import evaluate
from sklearn.model_selection import KFold
from sklearn.metrics import (
    balanced_accuracy_score, 
    f1_score, 
    precision_score, 
    recall_score,
    confusion_matrix,
    classification_report
)
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

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



def calculate_metrics_from_saved_models(X_full, y_full, model_class, n_splits=5, 
                                        device='cuda', model_kwargs=None):
    """
    Calculate balanced accuracy and F1 from already-trained models
    Uses the saved best_model_fold*.pth files
    
    Args:
        X_full: Full dataset (500, 4097)
        y_full: Full labels (500,)
        model_class: Your model class (e.g., simple_EEG_CNN)
        n_splits: Number of folds (must match what you used in training)
        device: 'cuda' or 'cpu'
        model_kwargs: Model initialization args
        
    Returns:
        Dictionary with all metrics
    """
    if model_kwargs is None:
        model_kwargs = {'input_length': 4097}
    
    # Recreate the same splits (same random_state=42)
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    results = {
        'accuracy': [],
        'balanced_accuracy': [],
        'f1': [],
        'precision': [],
        'recall': [],
        'confusion_matrices': [],
        'per_fold_details': []
    }
    
    print("Calculating metrics from saved models...\n")
    
    for fold, (train_ids, val_ids) in enumerate(kfold.split(X_full), 1):
        # Get validation data
        X_val = X_full[val_ids]
        y_val = y_full[val_ids]
        
        # Normalize using training stats (same as during training)
        X_train = X_full[train_ids]
        train_mean, train_std = X_train.mean(), X_train.std()
        X_val_norm = (X_val - train_mean) / train_std
        
        # Convert to tensor
        X_val_tensor = torch.tensor(X_val_norm, dtype=torch.float32).unsqueeze(1).to(device)
        
        # Load the saved best model
        model = model_class(**model_kwargs).to(device)
        try:
            model.load_state_dict(torch.load(f'best_model_fold{fold}.pth'))
        except FileNotFoundError:
            print(f"⚠️  Warning: best_model_fold{fold}.pth not found. Skipping fold {fold}.")
            continue
        
        model.eval()
        
        # Get predictions
        with torch.no_grad():
            outputs = model(X_val_tensor)
            _, predicted = outputs.max(1)
            predicted = predicted.cpu().numpy()
        
        # Calculate all metrics
        acc = (predicted == y_val).mean()
        bal_acc = balanced_accuracy_score(y_val, predicted)
        f1 = f1_score(y_val, predicted, average='binary')
        prec = precision_score(y_val, predicted, average='binary', zero_division=0)
        rec = recall_score(y_val, predicted, average='binary')
        cm = confusion_matrix(y_val, predicted)
        
        # Store results
        results['accuracy'].append(acc)
        results['balanced_accuracy'].append(bal_acc)
        results['f1'].append(f1)
        results['precision'].append(prec)
        results['recall'].append(rec)
        results['confusion_matrices'].append(cm)
        
        # Store detailed info
        results['per_fold_details'].append({
            'fold': fold,
            'accuracy': acc,
            'balanced_accuracy': bal_acc,
            'f1': f1,
            'precision': prec,
            'recall': rec,
            'confusion_matrix': cm
        })
        
        print(f"Fold {fold}: Acc={acc:.4f} | Bal_Acc={bal_acc:.4f} | F1={f1:.4f} | "
              f"Prec={prec:.4f} | Rec={rec:.4f}")
    
    return results


def print_quick_summary(metrics):
    """
    Print a clean summary of key metrics
    """
    print("\n" + "="*60)
    print("KEY METRICS SUMMARY")
    print("="*60)
    
    # Main metrics
    acc = np.array(metrics['accuracy'])
    bal_acc = np.array(metrics['balanced_accuracy'])
    f1 = np.array(metrics['f1'])
    prec = np.array(metrics['precision'])
    rec = np.array(metrics['recall'])
    
    print(f"\n{'Metric':<20} {'Mean ± Std':<20} {'Range':<20}")
    print("-"*60)
    print(f"{'Accuracy':<20} {acc.mean():.4f} ± {acc.std():.4f}     "
          f"[{acc.min():.4f}, {acc.max():.4f}]")
    print(f"{'Balanced Accuracy':<20} {bal_acc.mean():.4f} ± {bal_acc.std():.4f}     "
          f"[{bal_acc.min():.4f}, {bal_acc.max():.4f}]")
    print(f"{'F1 Score':<20} {f1.mean():.4f} ± {f1.std():.4f}     "
          f"[{f1.min():.4f}, {f1.max():.4f}]")
    print(f"{'Precision':<20} {prec.mean():.4f} ± {prec.std():.4f}     "
          f"[{prec.min():.4f}, {prec.max():.4f}]")
    print(f"{'Recall (Sensitivity)':<20} {rec.mean():.4f} ± {rec.std():.4f}     "
          f"[{rec.min():.4f}, {rec.max():.4f}]")
    
    # Average confusion matrix
    avg_cm = np.mean(metrics['confusion_matrices'], axis=0)
    tn, fp, fn, tp = avg_cm.ravel()
    
    print(f"\n{'Average Confusion Matrix:':<20}")
    print(f"  TN (Non-Seizure correct): {tn:.1f}")
    print(f"  FP (False alarm):         {fp:.1f}")
    print(f"  FN (Missed seizure):      {fn:.1f}  ⚠️ Most critical!")
    print(f"  TP (Seizure caught):      {tp:.1f}")
    
    print("="*60)


def plot_metrics_comparison(metrics):
    """
    Simple bar plot comparing key metrics
    """
    metric_names = ['accuracy', 'balanced_accuracy', 'f1', 'precision', 'recall']
    display_names = ['Accuracy', 'Balanced\nAccuracy', 'F1', 'Precision', 'Recall\n(Sensitivity)']
    
    means = [np.mean(metrics[m]) * 100 for m in metric_names]
    stds = [np.std(metrics[m]) * 100 for m in metric_names]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(metric_names))
    
    bars = ax.bar(x, means, yerr=stds, alpha=0.75, color='steelblue', 
                  edgecolor='black', capsize=7, linewidth=1.5)
    
    # Color code the bars
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    ax.set_ylabel('Score (%)', fontsize=13, fontweight='bold')
    ax.set_title('Model Performance Metrics (5-Fold CV)', fontsize=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(display_names, fontsize=11)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.set_ylim([90, 101])
    
    # Add value labels
    for i, (mean, std) in enumerate(zip(means, stds)):
        ax.text(i, mean + std + 0.5, f'{mean:.2f}%\n±{std:.2f}%', 
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Add horizontal line at 95%
    ax.axhline(95, color='red', linestyle='--', alpha=0.5, linewidth=1.5, 
               label='95% threshold')
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig('metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()