import sys
import types
import numpy as np

# ---- Patch old numpy._core structure ----
numpy_core = types.ModuleType('numpy._core')
sys.modules['numpy._core'] = numpy_core

numpy_core_multiarray = types.ModuleType('numpy._core.multiarray')
numpy_core_multiarray.__dict__.update(np.__dict__)
sys.modules['numpy._core.multiarray'] = numpy_core_multiarray

numpy_core_umath = types.ModuleType('numpy._core.umath')
numpy_core_umath.__dict__.update(np.__dict__)
sys.modules['numpy._core.umath'] = numpy_core_umath

# ---- Patch missing numpy.scalar before joblib loads ----
if not hasattr(np, "scalar"):
    def scalar(x):
        return x  # Return as-is
    np.scalar = scalar

# Make pickle think numpy has scalar at module level
import numpy as _np_module
setattr(_np_module, "scalar", np.scalar)

import joblib
from pathlib import Path

old_model_path = Path(__file__).parent / "emotion_detection_model.pkl"
print(f"Loading model from {old_model_path} ...")

model = joblib.load(old_model_path)

new_model_path = Path(__file__).parent / "emotion_detection_model_resaved.pkl"
joblib.dump(model, new_model_path)
print(f"✅ Model re-saved as {new_model_path}")
