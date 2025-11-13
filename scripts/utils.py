
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
    """Plot training curves for all folds"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    fold_histories = results['fold_histories']
    
    # Training loss
    ax = axes[0, 0]
    for i, fold in enumerate(fold_histories, 1):
        ax.plot(fold['train_losses'], label=f'Fold {i}', alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Loss')
    ax.set_title('Training Loss - All Folds')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Validation loss
    ax = axes[0, 1]
    for i, fold in enumerate(fold_histories, 1):
        ax.plot(fold['val_losses'], label=f'Fold {i}', alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Loss')
    ax.set_title('Validation Loss - All Folds')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Training accuracy
    ax = axes[1, 0]
    for i, fold in enumerate(fold_histories, 1):
        ax.plot(fold['train_accs'], label=f'Fold {i}', alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Accuracy')
    ax.set_title('Training Accuracy - All Folds')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Validation accuracy
    ax = axes[1, 1]
    for i, fold in enumerate(fold_histories, 1):
        ax.plot(fold['val_accs'], label=f'Fold {i}', alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Accuracy')
    ax.set_title('Validation Accuracy - All Folds')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('kfold_training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Bar plot of best accuracies
    fig, ax = plt.subplots(figsize=(10, 6))
    folds = np.arange(1, n_splits + 1)
    best_accs = results['best_val_accs']
    
    ax.bar(folds, best_accs, alpha=0.7, color='steelblue', edgecolor='black')
    ax.axhline(np.mean(best_accs), color='red', linestyle='--', 
               label=f'Mean: {np.mean(best_accs):.4f}')
    ax.set_xlabel('Fold')
    ax.set_ylabel('Best Validation Accuracy')
    ax.set_title('Best Validation Accuracy per Fold')
    ax.set_xticks(folds)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('kfold_best_accuracies.png', dpi=300, bbox_inches='tight')
    plt.show()