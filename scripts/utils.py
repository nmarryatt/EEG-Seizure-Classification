
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

