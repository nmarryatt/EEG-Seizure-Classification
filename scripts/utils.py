
import matplotlib.pyplot as plt
import numpy as np

def plot_eeg_segment(segment, label, num_samples=500, norm=False):
    """
    Plot the first num_samples of an EEG segment.

    Parameters:
    - segment: 1D numpy array of EEG samples
    - label: 0 (non-seizure) or 1 (seizure)
    - num_samples: number of samples to plot (default 500)
    """
    plt.figure(figsize=(12, 4))
    plt.plot(segment[:num_samples], color='blue')
    plt.title(f"EEG Segment Example - Label: {'Seizure' if label==1 else 'Non-seizure'}")
    plt.xlabel("Sample index")
    plt.ylabel(f"Amplitude ({'Normalised' if norm else 'Raw'})")
    plt.show()


import matplotlib.pyplot as plt

def plot_training_history(train_loss, val_loss, train_acc, val_acc, epochs=None):
    """
    Plot training and validation loss and accuracy curves.

    Parameters:
    - train_loss: list of training loss per epoch
    - val_loss: list of validation loss per epoch
    - train_acc: list of training accuracy per epoch
    - val_acc: list of validation accuracy per epoch
    - epochs: list of epoch numbers (optional). If None, defaults to 1..len(train_loss)
    """
    if epochs is None:
        epochs = list(range(1, len(train_loss) + 1))
    
    # Plot Loss
    plt.figure(figsize=(10,4))
    plt.plot(epochs, train_loss, label='Train Loss', marker='o')
    plt.plot(epochs, val_loss, label='Validation Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()
    
    # Plot Accuracy
    plt.figure(figsize=(10,4))
    plt.plot(epochs, train_acc, label='Train Accuracy', marker='o')
    plt.plot(epochs, val_acc, label='Validation Accuracy', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.show()


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
    
