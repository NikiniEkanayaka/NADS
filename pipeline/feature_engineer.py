import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine
import joblib


# -----------------------------
# Database Configuration
# -----------------------------
# DB_URI = "postgresql://username:password@localhost:5432/cic_ids"
# INPUT_TABLE = "flows_cleaned"
# OUTPUT_TABLE = "flows_features"


# -----------------------------
# Feature Engineering
# -----------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Avoid divide-by-zero
    df['Flow Duration'] = df['Flow Duration'].replace(0, np.nan)

    # Packet rate
    df['packet_rate'] = (
        (df['Total Fwd Packets'] + df['Total Backward Packets']) /
        df['Flow Duration']
    )

    # Byte rate
    df['byte_rate'] = (
        (df['Total Length of Fwd Packets'] +
         df['Total Length of Bwd Packets']) /
        df['Flow Duration']
    )

    # Port-based features
    if 'Destination Port' in df.columns:
        df['is_well_known_port'] = (df['Destination Port'] < 1024).astype(int)
        df['is_high_port'] = (df['Destination Port'] > 49152).astype(int)

    # Replace NaN created by division
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    return df


# -----------------------------
# Normalization
# -----------------------------
def normalize_features(df: pd.DataFrame):
    exclude_cols = [
        'Label', 'Attack Type', 'AttackBinary'
    ]

    feature_cols = [
        c for c in df.columns
        if c not in exclude_cols and df[c].dtype != 'object'
    ]

    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    joblib.dump(scaler, "artifacts/standard_scaler.pkl")
    print("Scaler saved to artifacts/standard_scaler.pkl")

    return df, feature_cols


# -----------------------------
# Load & Save Helpers
# -----------------------------
def load_from_postgres() -> pd.DataFrame:
    engine = create_engine(DB_URI)
    return pd.read_sql(INPUT_TABLE, engine)


def save_to_postgres(df: pd.DataFrame):
    engine = create_engine(DB_URI)
    df.to_sql(OUTPUT_TABLE, engine, if_exists='replace', index=False)
    print(f"Feature-engineered data saved to '{OUTPUT_TABLE}'")


# -----------------------------
# Script Entry Point
# -----------------------------
if __name__ == "__main__":
    print("Loading cleaned data from PostgreSQL...")
    df = load_from_postgres()

    print("Engineering features...")
    df = engineer_features(df)

    print("Normalizing features...")
    df, feature_cols = normalize_features(df)

    save_to_postgres(df)
