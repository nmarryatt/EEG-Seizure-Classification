
import matplotlib.pyplot as plt

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
