import pandas as pd
import numpy as np
import joblib
import psycopg2
import logging
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, precision_recall_curve
import os
from dotenv import load_dotenv

load_dotenv()

DB_CONFIG = {
    "dbname": os.getenv("DB_NAME"),     
    "user": os.getenv("DB_USER"),       
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST", "localhost"), 
    "port": int(os.getenv("DB_PORT", 5432))
}

OLD_MODEL_PATH = "model.pkl"
NEW_MODEL_PATH = "model_retrained.pkl"

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ------------------------------
# FETCH FEEDBACK
# ------------------------------
def fetch_feedback_data():
    """Fetch analyst-verified flows and labels."""
    query = """
    SELECT f.id, f.bytes_sent, f.packets, f.duration, fb.label
    FROM flows f JOIN alerts a ON f.id = a.flow_id JOIN feedback fb ON a.id = fb.alert_id
    """
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        df = pd.read_sql(query, conn)
        conn.close()
        return df
    except Exception as e:
        logging.error(f"Database connection failed: {e}")
        return None

# ------------------------------
# RETRAIN PIPELINE
# ------------------------------
                                            # min_samples = 5 ONLY for testing purpose
def retrain_model(min_samples: int = 5):
    logging.info("🚀 Starting retraining pipeline...")

    # 1️⃣ Load existing model artifact
    if not os.path.exists(OLD_MODEL_PATH):
        logging.warning(f"⚠️  Existing model not found at {OLD_MODEL_PATH}, training from scratch")
        artifact = {}
    else:
        artifact = joblib.load(OLD_MODEL_PATH)

    # 2️⃣ Load feedback data
    feedback_df = fetch_feedback_data()
    if feedback_df is None or len(feedback_df) < min_samples:
        logging.warning(f"❗ Not enough feedback data for retraining. Found: {len(feedback_df) if feedback_df is not None else 0}")
        return
    logging.info(f"📥 Loaded {len(feedback_df)} feedback samples")

    # 3️⃣ Build feature set from flows table + engineered features
    X = pd.DataFrame()
    X['bytes_sent'] = feedback_df['bytes_sent']
    X['packets'] = feedback_df['packets']
    X['duration'] = feedback_df['duration']
    X['packet_rate'] = feedback_df['packets'] / (feedback_df['duration'] + 1e-6)
    X['bytes_per_packet'] = feedback_df['bytes_sent'] / (feedback_df['packets'] + 1e-6)

    # 4️⃣ Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 5️⃣ Prepare labels (0=Normal, 1=Attack)
    y = feedback_df['label'].astype(int)

    # 6️⃣ Train Isolation Forest
    # contamination = y.mean()
    contamination = len(y[y == 1]) / len(y)
    contamination = max(0.01, min(contamination, 0.5))
    logging.info(f"🧪 Training Isolation Forest with contamination={contamination:.3f}")

    model = IsolationForest(
        n_estimators=200,
        max_samples=0.6,
        contamination=contamination,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_scaled)

    # 7️⃣ Threshold optimization
    scores = -model.decision_function(X_scaled)
    precision, recall, thresholds = precision_recall_curve(y, scores)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]

    # 8️⃣ Evaluation
    y_pred = (scores >= best_threshold).astype(int)
    f1 = f1_score(y, y_pred)
    roc_auc = roc_auc_score(y, scores)
    pr_auc = average_precision_score(y, scores)

    logging.info("===== RETRAINED MODEL METRICS =====")
    logging.info(f"F1 Score       : {f1:.4f}")
    logging.info(f"ROC-AUC        : {roc_auc:.4f}")
    logging.info(f"Avg Precision  : {pr_auc:.4f}")

    # 9️⃣ Save updated model
    new_artifact = {
        "model": model,
        "scaler": scaler,
        "threshold": best_threshold,
        "features": X.columns.tolist(),
        "retrained_on_samples": len(feedback_df)
    }
    joblib.dump(new_artifact, NEW_MODEL_PATH)
    logging.info(f"✅ Retrained model saved to {NEW_MODEL_PATH}")


# ------------------------------
# ENTRY POINT
# ------------------------------
if __name__ == "__main__":
    retrain_model()
