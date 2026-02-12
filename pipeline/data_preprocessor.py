import pandas as pd
import numpy as np
import os
import logging
from sqlalchemy import create_engine
from dotenv import load_dotenv

# Load env
load_dotenv()

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    def __init__(self, db_url=None):
        self.db_url = db_url or os.getenv("DATABASE_URL", "postgresql://postgres:1234@localhost:5432/nads_db")
        self.engine = create_engine(self.db_url)
        self.medians = {}

        # Map dataset columns → DB columns
        self.COLUMN_MAPPING = {
            "Source IP": "src_ip",
            "Destination IP": "dst_ip",
            "Source Port": "src_port",
            "Destination Port": "dst_port",
            "Protocol": "protocol",
            "Total Length of Fwd Packets": "bytes_sent",
            "Total Fwd Packets": "packets",
            "Flow Duration": "duration",
        }

        self.VALID_COLS = [
            "src_ip", "dst_ip", "src_port", "dst_port",
            "protocol", "bytes_sent", "packets", "duration"
        ]

    def clean_flow_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = [col.strip() for col in df.columns]

        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        for col in ['Flow Bytes/s', 'Flow Packets/s']:
            if col in df.columns:
                self.medians[col] = df[col].median()
                df[col] = df[col].fillna(self.medians[col])

        df.drop_duplicates(inplace=True)

        logger.info(f"Cleaned {len(df)} rows of network data.")
        return df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if "Total Fwd Packets" in df.columns and "Flow Duration" in df.columns:
            df["Packet_Rate"] = df["Total Fwd Packets"] / (df["Flow Duration"] + 1e-6)

        return df

    def map_to_flows_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = df.rename(columns=self.COLUMN_MAPPING)

        existing_cols = [c for c in self.VALID_COLS if c in df.columns]
        df = df[existing_cols]

        # Safe defaults
        if "src_ip" in df.columns:
            df["src_ip"] = df["src_ip"].fillna("0.0.0.0")
        else:
            df["src_ip"] = "0.0.0.0"

        if "dst_ip" in df.columns:
            df["dst_ip"] = df["dst_ip"].fillna("0.0.0.0")
        else:
            df["dst_ip"] = "0.0.0.0"

        if "protocol" in df.columns:
            df["protocol"] = df["protocol"].fillna("UNK")
        else:
            df["protocol"] = "UNK"

        for col in ["src_port", "dst_port", "packets", "bytes_sent", "duration"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
            else:
                df[col] = 0

        return df


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


if __name__ == "__main__":
    processor = DataPreprocessor()

    raw_data = pd.read_csv("data/All_dataset.csv")

    clean_data = processor.clean_flow_data(raw_data)
    feat_data = processor.engineer_features(clean_data)
    db_ready_data = processor.map_to_flows_schema(feat_data)

    processor.store_to_db(db_ready_data)
