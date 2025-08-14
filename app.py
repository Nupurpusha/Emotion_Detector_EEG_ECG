from fastapi import FastAPI, File, UploadFile
import joblib
import tempfile
import shutil
from pathlib import Path
import sys, os, numpy, sklearn, joblib
from pathlib import Path

print("Python:", sys.version)
print("NumPy:", numpy.__version__)
print("scikit-learn:", sklearn.__version__)
print("joblib:", joblib.__version__)
print("Files in /app:", os.listdir(Path(__file__).parent))


from eeg import load_eeg
from ecg import load_ecg
from datafusion import fuse_features

app = FastAPI(title="Emotion Detection API")

# Load trained model
model_path = Path(__file__).parent / "emotion_detection_model.pkl"
if not model_path.exists():
    raise FileNotFoundError(f"Model file not found: {model_path}")
model = joblib.load(model_path)

@app.post("/predict/")
async def predict(eeg_file: UploadFile = File(...), ecg_file: UploadFile = File(...)):
    # Save uploaded files
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as eeg_tmp:
        shutil.copyfileobj(eeg_file.file, eeg_tmp)
        eeg_path = eeg_tmp.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as ecg_tmp:
        shutil.copyfileobj(ecg_file.file, ecg_tmp)
        ecg_path = ecg_tmp.name

    # Load EEG & ECG
    eeg_features, _ = load_eeg(eeg_path, n_components=20)
    ecg_features, _ = load_ecg(ecg_path, n_components=20)

    # Fuse features
    fused_features = fuse_features(eeg_features, ecg_features)

    # Predict
    prediction = model.predict(fused_features)

    return {
        "predictions": prediction.tolist(),
        "meaning": ["High" if p == 1 else "Low" for p in prediction]
    }
