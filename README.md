# Seer Technical Interview: EEG Seizure Clasficiation using 1D CNN
This project uses the **BONN EEG Dataset** to train a **1D convolution neural network (CNN)** to distinguish **seizure (ictal) vs non seizure (interictal/healthy) EEG segments**. 
The Bonn dataset contains EEG recordings from healthy volunteers and epilepsy patients, with segments labeled as seizure (ictal) or non-seizure (comprised of interictal and healthy). The goal is to demonstrate a deep learning approach for seizure detection from single-channel EEG data.


### Dataset

**Bonn EEG dataset** contains **five sets (A–E)**, each with 100 single-channel EEG segments (~23.6 seconds each, 4097 samples per segment, 173.61 Hz sampling rate).  These segments were manually selected from continuous EEG recordings, artifact free and stationary. Specifically, sets C/D/E are used, with the goal of classifying ictal from interictal EEG segments. 

| Set | Description | Seizure? |
|-----|------------|-----------|
| A   | Healthy, eyes open | No |
| B   | Healthy, eyes closed | No |
| C   | Interictal, non-epileptogenic | No |
| D   | Interictal, epileptogenic | No |
| E   | Seizure (ictal) | Yes |


### Challenges and limitations of BONN dataset
**Small dataset**: Relatively small dataset, using only 300 segments, 200 which are non-seizure and 100 which are seizure. 
- Key consideration in model architecture: dropout, weight decay, batch normalisation, number of layers
- Data augmentation (seizure segments)

**Class imbalance**: Class imbalance with 2:1 non-seizure to seizure ratio.
- Class weights
- Data augmentation of seizure segments


**No patient-level data**: Patients may belong in both training and testing samples, could cause overfitting and over-optimistic performance metrics
- Limitation: only way to test would be to do external validation from new dataset (not included in project) 


### Goals

1. Load and preprocess EEG segments from all sets
2. Normalize and filter EEG signals
3. Build and train a MLP to classify seizure vs non-seizure segments as a baseline comparison model
4. Build and train a **1D CNN** to classify seizure vs non-seizure segments
5. Evaluate model performance using loss curves, accuracy and confusion matrix
6. Apply data augmentation to address potential overfitting and limited sample size


### Links:
- [Bonn EEG Dataset](https://www.upf.edu/web/ntsa/downloads/-/asset_publisher/xvT6E4pczrBw/content/2001-indications-of-nonlinear-deterministic-and-finite-dimensional-structures-in-time-series-of-brain-electrical-activity-dependence-on-recording-regi)
- [Original Paper: Andrzejak et al., 2001](https://www.upf.edu/documents/229517819/232450661/Andrzejak-PhysicalReviewE2001.pdf/0e9a54b8-8993-b400-743e-4d64fa29fb63)
