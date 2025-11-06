# EEG Seizure Clasficiation using 1D CNN
This project uses the **BONN EEG Dataset** to train a **1D convolution neural network (CNN)** to distinguish **seizure (ictal) vs non seizure (interictal) EEG segments**. 
The Bonn dataset contains EEG recordings from healthy volunteers and epilepsy patients, with segments labeled as seizure (ictal) or non-seizure (comprised of interictal and healthy). The goal is to demonstrate a deep learning approach for automated seizure detection from single-channel EEG data, specifically distinguishing between ictal and interictal EEG segments. 


### Dataset

**Bonn EEG dataset** contains **five sets (A–E)**, each with 100 single-channel EEG segments (~23.6 seconds each, 4097 samples per segment, 173.61 Hz sampling rate).  

| Set | Description | Seizure? |
|-----|------------|-----------|
| A   | Healthy, eyes open | No |
| B   | Healthy, eyes closed | No |
| C   | Interictal, non-epileptogenic | No |
| D   | Interictal, epileptogenic | No |
| E   | Seizure (ictal) | Yes |

I will look specifically at distinguishing between ictal and interictal EEG segments, using datasets C, D and E. 

### Links:
- [Bonn EEG Dataset](https://www.upf.edu/web/ntsa/downloads/-/asset_publisher/xvT6E4pczrBw/content/2001-indications-of-nonlinear-deterministic-and-finite-dimensional-structures-in-time-series-of-brain-electrical-activity-dependence-on-recording-regi)
- [Original Paper: Andrzejak et al., 2001](https://www.upf.edu/documents/229517819/232450661/Andrzejak-PhysicalReviewE2001.pdf/0e9a54b8-8993-b400-743e-4d64fa29fb63)


### Goals

1. Load and preprocess EEG segments from all sets
2. Normalize and filter EEG signals
3. Prepare training and test datasets  
4. Build and train a baseline **MLP** to classify seizure vs non-seizure segments
4. Build and train a **1D CNN** to classify seizure vs non-seizure segments, compare to baseline MLP model
5. Evaluate model performance using loss curves, accuracy and confusion matrix


## Results
The 1D CNN model effectively classified EEG segments into seizure (ictal) and non-seizure (interictal) classes. After training 10 epochs, the mdoel achieved approximately 98% accuracy on training and validation sets, with minimal loss (~0.03-0.05). Only a small number of seizure segments were misclassified as non-seizure segments, demonstrating the models ability to reliably capture temporal patterns in EEG data to make accurate classifications. 