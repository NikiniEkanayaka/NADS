# retrain_pipeline.py
import os
import logging
import joblib
import pandas as pd
import numpy as np
import psycopg2
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve
)
from dotenv import load_dotenv

load_dotenv()

# ------------------------------
# Logging
# ------------------------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ------------------------------
# PostgreSQL config
# ------------------------------
DB_CONFIG = {
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", 5432))
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OLD_MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
NEW_MODEL_PATH = os.path.join(BASE_DIR, "model_retrained.pkl")

# ------------------------------
# SELECTED FEATURES
# ------------------------------
SELECTED_FEATURES = [
    'Destination Port',
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
    'Fwd Header Length',
    'Bwd Header Length',
    'Subflow Fwd Packets',
    'Subflow Bwd Packets'
]

# ------------------------------
# Fetch feedback data
# ------------------------------
def fetch_feedback_data():

    query = """
        SELECT f.id, f.bytes_sent, f.packets, f.duration, f.dst_port,
               fb.label
        FROM flows f
        JOIN alerts a ON f.id = a.flow_id
        JOIN feedback fb ON a.id = fb.alert_id
    """
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        df = pd.read_sql(query, conn)
        conn.close()
        logging.info(f"📥 Retrieved {len(df)} feedback samples from DB")
        return df
    except Exception as e:
        logging.error(f"Database connection failed: {e}")
        return None

# ------------------------------
# Reconstruct features from DB
# ------------------------------
def build_features(df):
    X = pd.DataFrame()
    
    # Basic features
    X['Destination Port'] = df['dst_port']
    X['Flow Duration'] = df['duration']
    X['Total Fwd Packets'] = df['packets']
    X['Total Backward Packets'] = 0  
    X['Fwd Packets/s'] = df['packets'] / (df['duration'] + 1e-6)
    X['Bwd Packets/s'] = 0
    X['Min Packet Length'] = 0
    X['Max Packet Length'] = df['bytes_sent']
    X['Packet Length Mean'] = df['bytes_sent'] / (df['packets'] + 1e-6)
    X['Packet Length Std'] = 0
    X['Packet Length Variance'] = 0
    X['Average Packet Size'] = df['bytes_sent'] / (df['packets'] + 1e-6)
    X['Down/Up Ratio'] = 0
    X['SYN Flag Count'] = 0
    X['FIN Flag Count'] = 0
    X['RST Flag Count'] = 0
    X['PSH Flag Count'] = 0
    X['ACK Flag Count'] = 0
    X['URG Flag Count'] = 0
    X['Init_Win_bytes_forward'] = 0
    X['Init_Win_bytes_backward'] = 0
    X['Avg Fwd Segment Size'] = 0
    X['Avg Bwd Segment Size'] = 0
    X['Fwd Header Length'] = 0
    X['Bwd Header Length'] = 0
    X['Subflow Fwd Packets'] = 0
    X['Subflow Bwd Packets'] = 0
    

    X = X[SELECTED_FEATURES]
    
    return X

# ------------------------------
# Retrain model
# ------------------------------
def retrain_model(min_samples: int = 5):
    logging.info("🚀 Starting retraining pipeline...")

    # Load old model if exists
    if os.path.exists(OLD_MODEL_PATH):
        artifact = joblib.load(OLD_MODEL_PATH)
        logging.info(f"✅ Loaded existing model from {OLD_MODEL_PATH}")
    else:
        artifact = None
        logging.warning("⚠️ No existing model found. Training from scratch")

    # Load feedback data
    df_feedback = fetch_feedback_data()
    if df_feedback is None or len(df_feedback) < min_samples:
        logging.warning(f"❗ Not enough feedback data for retraining. Found: {len(df_feedback) if df_feedback is not None else 0}")
        return

    # Build features
    X = build_features(df_feedback)
    y = df_feedback['label'].astype(int)  # 0 = Normal, 1 = Attack

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train Isolation Forest
    contamination = max(0.01, min(y.mean(), 0.5))
    model = IsolationForest(
        n_estimators=300,
        max_samples="auto",
        contamination=contamination,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_scaled)

    # Threshold optimization
    scores = -model.decision_function(X_scaled)
    precision, recall, thresholds = precision_recall_curve(y, scores)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]

    # Evaluation metrics
    y_pred = (scores >= best_threshold).astype(int)
    f1 = f1_score(y, y_pred)
    roc_auc = roc_auc_score(y, scores)
    pr_auc = average_precision_score(y, scores)

    logging.info("===== RETRAINED MODEL METRICS =====")
    logging.info(f"F1 Score       : {f1:.4f}")
    logging.info(f"ROC-AUC        : {roc_auc:.4f}")
    logging.info(f"Avg Precision  : {pr_auc:.4f}")

    # Save retrained model
    artifact_new = {
        "model": model,
        "scaler": scaler,
        "threshold": best_threshold,
        "features": SELECTED_FEATURES,
        "retrained_on_samples": len(df_feedback)
    }
    joblib.dump(artifact_new, NEW_MODEL_PATH)
    logging.info(f"✅ Retrained model saved to {NEW_MODEL_PATH}")

# ------------------------------
# Main
# ------------------------------
if __name__ == "__main__":
    retrain_model()
