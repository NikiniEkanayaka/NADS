import os
import logging
import joblib
import pandas as pd
import numpy as np
import psycopg2

from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import f1_score
from xgboost import XGBClassifier

load_dotenv()

# ------------------------------
# Logging
# ------------------------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ------------------------------
# DB CONFIG
# ------------------------------
DB_CONFIG = {
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", 5432))
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model_retrained_xgboost.pkl")

# ------------------------------
# FEATURES (MUST MATCH TRAINING)
# ------------------------------
SELECTED_FEATURES = [
    'Flow Duration',
    'Total Fwd Packets',
    'Total Backward Packets',
    'Down/Up Ratio',
    'Average Packet Size',
    'Packet Length Mean',
    'Packet Length Std',
    'Min Packet Length',
    'Max Packet Length',
    'Packet Length Variance',
    'Fwd Packets/s',
    'Bwd Packets/s',
    'SYN Flag Count',
    'FIN Flag Count',
    'RST Flag Count',
    'PSH Flag Count',
    'ACK Flag Count',
    'URG Flag Count',
    'Init_Win_bytes_forward',
    'Init_Win_bytes_backward',
    'Avg Fwd Segment Size',
    'Avg Bwd Segment Size',
    'Destination Port',
    'Fwd Header Length',
    'Bwd Header Length',
    'Subflow Fwd Packets',
    'Subflow Bwd Packets'
]

# ------------------------------
# FETCH FEEDBACK DATA
# ------------------------------
def fetch_feedback_data():
    query = """
        SELECT f.bytes_sent, f.packets, f.duration, f.dst_port,
               fb.label
        FROM flows f
        JOIN alerts a ON f.id = a.flow_id
        JOIN feedback fb ON a.id = fb.alert_id
    """
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        df = pd.read_sql(query, conn)
        conn.close()
        logging.info(f"📥 Retrieved {len(df)} feedback samples")
        return df
    except Exception as e:
        logging.error(f"DB Error: {e}")
        return None


# ------------------------------
# FEATURE ENGINEERING (MATCH API)
# ------------------------------
def build_features(df):
    X = pd.DataFrame()

    X['Flow Duration'] = df['duration']
    X['Total Fwd Packets'] = df['packets']
    X['Total Backward Packets'] = 0
    X['Down/Up Ratio'] = 0

    X['Average Packet Size'] = df['bytes_sent'] / (df['packets'] + 1e-6)
    X['Packet Length Mean'] = X['Average Packet Size']
    X['Packet Length Std'] = 0
    X['Min Packet Length'] = X['Average Packet Size']
    X['Max Packet Length'] = df['bytes_sent']
    X['Packet Length Variance'] = 0

    X['Fwd Packets/s'] = df['packets'] / (df['duration'] + 1e-6)
    X['Bwd Packets/s'] = 0

    # Flags (not available → default)
    for col in [
        'SYN Flag Count','FIN Flag Count','RST Flag Count',
        'PSH Flag Count','ACK Flag Count','URG Flag Count'
    ]:
        X[col] = 0

    X['Init_Win_bytes_forward'] = 0
    X['Init_Win_bytes_backward'] = 0
    X['Avg Fwd Segment Size'] = 0
    X['Avg Bwd Segment Size'] = 0

    X['Destination Port'] = df['dst_port']
    X['Fwd Header Length'] = 0
    X['Bwd Header Length'] = 0
    X['Subflow Fwd Packets'] = df['packets']
    X['Subflow Bwd Packets'] = 0

    return X[SELECTED_FEATURES]


# ------------------------------
# RETRAIN MODEL
# ------------------------------
def retrain_model(min_samples=10):
    logging.info("🚀 Starting XGBoost retraining...")

    df = fetch_feedback_data()

    if df is None or len(df) < min_samples:
        logging.warning(f"❗ Not enough data ({len(df) if df is not None else 0})")
        return

    # Build dataset
    X = build_features(df)
    y = df['label'].astype(int)

    # Encode labels (important if future multiclass)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Handle imbalance
    sample_weights = compute_sample_weight("balanced", y_encoded)

    # Train XGBoost
    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method='hist',
        eval_metric='mlogloss',
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_scaled, y_encoded, sample_weight=sample_weights)

    # ------------------------------
    # THRESHOLD OPTIMIZATION
    # ------------------------------
    probs = model.predict_proba(X_scaled)
    thresholds = np.arange(0.1, 0.95, 0.05)

    best_thresh = 0.5
    best_f1 = 0

    benign_class = 0

    for t in thresholds:
        preds = np.where(np.max(probs, axis=1) >= t,
                         np.argmax(probs, axis=1),
                         benign_class)

        f1 = f1_score(y_encoded, preds, average='macro')

        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t

    logging.info(f"✅ Best Threshold: {best_thresh:.2f}")
    logging.info(f"✅ Best F1 Score: {best_f1:.4f}")

    # ------------------------------
    # SAVE MODEL
    # ------------------------------
    artifact = {
        "model": model,
        "scaler": scaler,
        "threshold": best_thresh,
        "features": SELECTED_FEATURES,
        "label_encoder": le,
        "samples": len(df)
    }

    joblib.dump(artifact, MODEL_PATH)

    logging.info(f"💾 Model saved → {MODEL_PATH}")


# ------------------------------
# MAIN
# ------------------------------
if __name__ == "__main__":
    retrain_model()