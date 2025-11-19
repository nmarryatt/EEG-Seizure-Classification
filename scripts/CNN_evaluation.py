import torch
from torch.utils.data import DataLoader, TensorDataset, SubsetRandomSampler
import numpy as np
import torch.nn as nn
import torch.optim as optim
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

from torch.utils.data import DataLoader, TensorDataset, SubsetRandomSampler
from scripts.CV import normalize_fold_data, create_fold_dataloaders, calculate_class_weights, print_cv_summary, train_single_fold



def plot_kfold_results(results, n_splits=5):
    """Plot training curves with aligned x and y axes"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fold_histories = results['fold_histories']
    
    # Find max epochs
    max_epochs = max(len(fold['train_losses']) for fold in fold_histories)
    
    # Find y-axis ranges across ALL folds
    all_train_losses = [loss for fold in fold_histories for loss in fold['train_losses']]
    all_val_losses = [loss for fold in fold_histories for loss in fold['val_losses']]
    all_train_accs = [acc for fold in fold_histories for acc in fold['train_accs']]
    all_val_accs = [acc for fold in fold_histories for acc in fold['val_accs']]
    
    loss_min = min(min(all_train_losses), min(all_val_losses))
    loss_max = max(max(all_train_losses), max(all_val_losses))
    acc_min = min(min(all_train_accs), min(all_val_accs))
    acc_max = max(max(all_train_accs), max(all_val_accs))
    
    # Add some padding (5%)
    loss_padding = (loss_max - loss_min) * 0.05
    acc_padding = (acc_max - acc_min) * 0.05
    
    # Training loss
    ax = axes[0, 0]
    for i, fold in enumerate(fold_histories, 1):
        epochs = range(1, len(fold['train_losses']) + 1)
        ax.plot(epochs, fold['train_losses'], label=f'Fold {i}', alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Loss')
    ax.set_title('Training Loss - All Folds')
    ax.set_xlim(1, max_epochs)
    ax.set_ylim(loss_min - loss_padding, loss_max + loss_padding)  # ← SAME Y-AXIS
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Validation loss
    ax = axes[0, 1]
    for i, fold in enumerate(fold_histories, 1):
        epochs = range(1, len(fold['val_losses']) + 1)
        ax.plot(epochs, fold['val_losses'], label=f'Fold {i}', alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Loss')
    ax.set_title('Validation Loss - All Folds')
    ax.set_xlim(1, max_epochs)
    ax.set_ylim(loss_min - loss_padding, loss_max + loss_padding)  # ← SAME Y-AXIS
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Training accuracy
    ax = axes[1, 0]
    for i, fold in enumerate(fold_histories, 1):
        epochs = range(1, len(fold['train_accs']) + 1)
        ax.plot(epochs, fold['train_accs'], label=f'Fold {i}', alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Accuracy')
    ax.set_title('Training Accuracy - All Folds')
    ax.set_xlim(1, max_epochs)
    ax.set_ylim(acc_min - acc_padding, acc_max + acc_padding)  # ← SAME Y-AXIS
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Validation accuracy
    ax = axes[1, 1]
    for i, fold in enumerate(fold_histories, 1):
        epochs = range(1, len(fold['val_accs']) + 1)
        ax.plot(epochs, fold['val_accs'], label=f'Fold {i}', alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Accuracy')
    ax.set_title('Validation Accuracy - All Folds')
    ax.set_xlim(1, max_epochs)
    ax.set_ylim(acc_min - acc_padding, acc_max + acc_padding)  # ← SAME Y-AXIS
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('kfold_training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    

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
    Clean bar plot comparing key metrics
    """
    metric_names = ['accuracy', 'balanced_accuracy', 'f1', 'precision', 'recall']
    display_names = ['Accuracy', 'Balanced\nAccuracy', 'F1', 'Precision', 'Recall\n(Sensitivity)']
    
    means = [np.mean(metrics[m]) * 100 for m in metric_names]
    stds = [np.std(metrics[m]) * 100 for m in metric_names]
    
    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(metric_names))
    
    # Color code the bars
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']
    
    bars = ax.bar(x, means, yerr=stds, alpha=0.8, 
                  color=colors, edgecolor='black', 
                  capsize=8, linewidth=1.5, ecolor='black')
    
    ax.set_ylabel('Score (%)', fontsize=14, fontweight='bold')
    ax.set_title('Model Performance Metrics (5-Fold CV)', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(display_names, fontsize=12)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.set_ylim([88, 102])
    
    # Add value labels on top of bars (above error bars)
    for i, (mean, std) in enumerate(zip(means, stds)):
        y_pos = mean + std + 0.8
        ax.text(i, y_pos, f'{mean:.2f}%\n±{std:.2f}%', 
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()