import numpy as np

def fuse_features(eeg_features, ecg_features):
    """
    Concatenate EEG and ECG feature arrays along the feature axis.
    Ensures they have the same number of samples.
    """
    min_len = min(len(eeg_features), len(ecg_features))
    eeg_features = eeg_features[:min_len]
    ecg_features = ecg_features[:min_len]
    return np.concatenate((eeg_features, ecg_features), axis=1)

