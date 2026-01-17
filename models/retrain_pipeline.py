import pandas as pd
import numpy as np
import joblib
import psycopg2
import logging
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, precision_recall_curve
from dotenv import load_dotenv
import os

load_dotenv()

DB_CONFIG = {
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT", 5432)
}

OLD_MODEL_PATH = "model.pkl"
NEW_MODEL_PATH = "model_retrained.pkl"

logging.basicConfig(level=logging.INFO)


# =========================
# FETCH FEEDBACK DATA
# =========================
def fetch_feedback_data():
    query = """
    SELECT f.bytes_sent, f.packets, f.duration, fb.label
    FROM flows f
    JOIN alerts a ON f.id = a.flow_id
    JOIN feedback fb ON a.id = fb.alert_id
    """
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        df = pd.read_sql(query, conn)
        conn.close()
        return df
    except Exception as e:
        logging.error(f"Database connection failed: {e}")
        return None


# =========================
# RETRAIN PIPELINE
# =========================
def retrain_model():
    logging.info("🚀 Starting retraining pipeline...")

    # 1️⃣ Load Existing Model Artifact
    artifact = joblib.load(OLD_MODEL_PATH)
    old_model = artifact["model"]
    scaler = artifact["scaler"]
    feature_cols = artifact["features"]

    # 2️⃣ Load Analyst Feedback Data
    feedback_df = fetch_feedback_data()
    if feedback_df is None or len(feedback_df) < 50:
        logging.warning("❗ Not enough feedback data for retraining.")
        return

    # Ensure correct label format
    feedback_df["AttackBinary"] = feedback_df["label"].map({"Normal": 0, "Attack": 1})

    X_new = feedback_df[feature_cols]
    y_new = feedback_df["AttackBinary"]

    # 3️⃣ Scale using OLD scaler
    X_scaled = scaler.transform(X_new)

    # 4️⃣ Retrain Isolation Forest
    contamination = y_new.mean()
    logging.info(f"Retraining with contamination={contamination:.3f}")

    new_model = IsolationForest(
        n_estimators=300,
        max_samples=0.6,
        contamination=contamination,
        random_state=42,
        n_jobs=-1
    )
    new_model.fit(X_scaled)

    # 5️⃣ Threshold Optimization
    scores = -new_model.decision_function(X_scaled)

    precision, recall, thresholds = precision_recall_curve(y_new, scores)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)

    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]

    # 6️⃣ Evaluation
    y_pred = (scores >= best_threshold).astype(int)

    f1 = f1_score(y_new, y_pred)
    roc_auc = roc_auc_score(y_new, scores)
    pr_auc = average_precision_score(y_new, scores)

    logging.info("===== RETRAINED MODEL METRICS =====")
    logging.info(f"F1 Score       : {f1:.4f}")
    logging.info(f"ROC-AUC        : {roc_auc:.4f}")
    logging.info(f"Avg Precision  : {pr_auc:.4f}")

    # 7️⃣ Save Updated Artifact
    new_artifact = {
        "model": new_model,
        "scaler": scaler,
        "threshold": best_threshold,
        "features": feature_cols,
        "retrained_on_samples": len(feedback_df)
    }

    joblib.dump(new_artifact, NEW_MODEL_PATH)
    logging.info(f"✅ Retrained model saved to {NEW_MODEL_PATH}")


# =========================
# ENTRY POINT
# =========================
if __name__ == "__main__":
    retrain_model()
