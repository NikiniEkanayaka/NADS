import os
import joblib
from dotenv import load_dotenv

load_dotenv()

MODEL_PATH = os.getenv("MODEL_PATH", "models/xgb_model.pkl")
RETRAINED_PATH = os.getenv("RETRAINED_PATH", "models/model_retrained.pkl")
PACKET_LEVEL_PATH = os.getenv("PACKET_LEVEL_PATH", "models/xgb_model_packet_level.pkl")

model = None
scaler = None
threshold = None
features = None
label_encoder = None

try:

    if os.path.exists(RETRAINED_PATH):
        path = RETRAINED_PATH
    else:
        path = MODEL_PATH

    raw = joblib.load(path)

    model = raw["model"]
    scaler = raw["scaler"]
    threshold = raw.get("threshold", 0.5)
    features = raw.get("features")
    label_encoder = raw.get("label_encoder")

    print(f"🧠 Loaded model from: {path}")
    print(f"📊 Features count: {len(features) if features else 'Unknown'}")

except Exception as e:
    print(f"❌ Failed to load model: {e}")


print("✅ Model type:", type(model))