import os
import joblib
from dotenv import load_dotenv

load_dotenv()

MODEL_PATH = os.getenv("MODEL_PATH", "models/model.pkl")
RETRAINED_PATH = os.getenv("RETRAINED_PATH", "models/model_retrained.pkl")

model = None
scaler = None
threshold = None
features = None

try:
    # path = RETRAINED_PATH if os.path.exists(RETRAINED_PATH) else MODEL_PATH
    path = MODEL_PATH
    raw = joblib.load(path)

    model = raw["model"]
    scaler = raw["scaler"]
    threshold = raw["threshold"]
    features = raw.get("features")

    print(f"🧠 Loaded model from: {path}")
    print("📊 Features:", features)

except Exception as e:
    print(f"❌ Failed to load model: {e}")

