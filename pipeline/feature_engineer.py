import pandas as pd
import numpy as np
import os
import logging
import joblib
from sqlalchemy import create_engine
from dotenv import load_dotenv

# Load env
load_dotenv()

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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

# ===== DB column mapping =====
COLUMN_MAPPING = {
    "Source IP": "src_ip",
    "Destination IP": "dst_ip",
    "Source Port": "src_port",
    "Destination Port": "dst_port",
    "Protocol": "protocol",
    "Total Length of Fwd Packets": "bytes_sent",
    "Total Fwd Packets": "packets",
    "Flow Duration": "duration"
}

VALID_DB_COLS = [
    "src_ip", "dst_ip", "src_port", "dst_port",
    "protocol", "bytes_sent", "packets", "duration",
    "anomaly_score"
]

class NADSProcessor:
    def __init__(self, db_url=None):
        self.db_url = db_url or os.getenv(
            "DATABASE_URL", "postgresql://postgres:1234@localhost:5432/nads_db"
        )
        self.engine = create_engine(self.db_url)

        # Load model + scaler
        self.model_bundle_path = os.getenv("MODEL_PATH", "models/model.pkl")
        if os.path.exists(self.model_bundle_path):
            bundle = joblib.load(self.model_bundle_path)
            self.model = bundle.get("model")
            self.scaler = bundle.get("scaler")
            self.model_features = bundle.get("features", SELECTED_FEATURES)
            logger.info("Loaded model + scaler + features from bundle.")
        else:
            self.model = None
            self.scaler = None
            self.model_features = SELECTED_FEATURES
            logger.warning("Model bundle not found, anomaly scores will not be computed.")

        self.medians = {}

    # --------------------------
    # Data cleaning
    # --------------------------
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = [col.strip() for col in df.columns]

        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        for col in ["Flow Bytes/s", "Flow Packets/s"]:
            if col in df.columns:
                self.medians[col] = df[col].median()
                df[col] = df[col].fillna(self.medians[col])

        df.drop_duplicates(inplace=True)
        logger.info(f"Cleaned {len(df)} rows of network data.")
        return df

    # --------------------------
    # Feature engineering
    # --------------------------
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select features for ML model."""
        df = df.copy()
        available_features = [f for f in self.model_features if f in df.columns]
        if len(available_features) < len(self.model_features):
            missing = set(self.model_features) - set(available_features)
            logger.warning(f"Missing features dropped: {missing}")
        X = df[available_features]
        return X

    # --------------------------
    # Feature scaling
    # --------------------------
    def scale_features(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.scaler:
            scaled = self.scaler.transform(X)
            X_scaled = pd.DataFrame(scaled, columns=X.columns, index=X.index)
            return X_scaled
        return X

    # --------------------------
    # Anomaly scoring
    # --------------------------
    def calculate_anomaly_score(self, X: pd.DataFrame) -> pd.Series:
        if self.model:
            # IsolationForest decision_function returns higher = normal, lower = anomaly
            score = self.model.decision_function(X)
            return pd.Series(score, index=X.index)
        else:
            return pd.Series(np.nan, index=X.index)

    # --------------------------
    # Map to DB schema
    # --------------------------
    def map_to_db_schema(self, df: pd.DataFrame, raw_df: pd.DataFrame, anomaly_scores: pd.Series) -> pd.DataFrame:
        """Map raw CSV + anomaly scores to flows table schema."""
        df = raw_df.rename(columns=COLUMN_MAPPING)
        existing_cols = [c for c in VALID_DB_COLS if c in df.columns]
        df = df[existing_cols]

        # Fill missing data safely
        df["src_ip"] = df.get("src_ip", pd.Series("0.0.0.0", index=df.index)).fillna("0.0.0.0")
        df["dst_ip"] = df.get("dst_ip", pd.Series("0.0.0.0", index=df.index)).fillna("0.0.0.0")
        df["protocol"] = df.get("protocol", pd.Series("UNK", index=df.index)).fillna("UNK")

        for col in ["src_port", "dst_port", "packets", "bytes_sent", "duration"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
            else:
                df[col] = pd.Series(0, index=df.index)

        # Add anomaly scores
        df["anomaly_score"] = anomaly_scores

        return df

    # --------------------------
    # Store in DB
    # --------------------------
    def store_to_db(self, df: pd.DataFrame, table_name="flows"):
        try:
            df.to_sql(
                table_name,
                self.engine,
                if_exists="append",
                index=False,
                chunksize=10000,
                method="multi"
            )
            logger.info(f"Inserted {len(df)} rows into '{table_name}'.")
        except Exception as e:
            logger.error(f"Database insertion failed: {e}")

# ==============================
# Main pipeline
# ==============================
if __name__ == "__main__":
    processor = NADSProcessor()

    raw_data = pd.read_csv("data/All_dataset.csv")
    clean_df = processor.clean_data(raw_data)
    feat_df = processor.engineer_features(clean_df)
    scaled_df = processor.scale_features(feat_df)
    anomaly_scores = processor.calculate_anomaly_score(scaled_df)
    db_ready_df = processor.map_to_db_schema(scaled_df, raw_data, anomaly_scores)
    processor.store_to_db(db_ready_df, table_name="flows")
