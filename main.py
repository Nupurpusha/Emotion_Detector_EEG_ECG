import warnings
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import joblib
from pathlib import Path

from eeg import load_eeg
from ecg import load_ecg
from datafusion import fuse_features

if __name__ == "__main__":
    # Load EEG & ECG data
    eeg_features, eeg_labels = load_eeg("EEG_features.csv", n_components=20)
    ecg_features, ecg_labels = load_ecg("ECG_features.csv", n_components=20)

    # Use EEG labels
    labels = eeg_labels

    # Convert continuous labels to binary
    threshold = np.median(labels)
    labels = np.where(labels >= threshold, 1, 0)

    # Fuse features
    fused_features = fuse_features(eeg_features, ecg_features)

    # Balance data
    smote = SMOTE(random_state=42)
    fused_features, labels = smote.fit_resample(fused_features, labels)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        fused_features, labels, test_size=0.3, random_state=42
    )

    # Models
    rf = RandomForestClassifier(random_state=42)
    gbdt = GradientBoostingClassifier(random_state=42)

    ensemble = VotingClassifier(
        estimators=[("rf", rf), ("gbdt", gbdt)],
        voting="soft"
    )

    # Hyperparameter tuning
    param_grid = {
        "rf__n_estimators": [100, 200],
        "rf__max_depth": [10, 20, None],
        "gbdt__n_estimators": [100, 200],
        "gbdt__learning_rate": [0.05, 0.1]
    }

    grid = GridSearchCV(ensemble, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
    grid.fit(X_train, y_train)

    # Evaluation
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)

    print(f"Best Params: {grid.best_params_}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))

    # Save model in current directory
    model_path = Path(__file__).parent / "emotion_detection_model.pkl"
    joblib.dump(best_model, model_path)
    print(f"Model saved at {model_path}")
