import pandas as pd
import numpy as np
import argparse
import os
from sqlalchemy import create_engine


# -----------------------------
# Database Configuration
# -----------------------------
# DB_URI = "postgresql://username:password@localhost:5432/cic_ids"
# TABLE_NAME = "flows_cleaned"


# -----------------------------
# Label Cleaning & Mapping
# -----------------------------
def clean_labels(df: pd.DataFrame) -> pd.DataFrame:
    df['Label'] = (
        df['Label']
        .astype(str)
        .str.replace('ï¿½', '-', regex=False)
        .str.replace(' ', '-', regex=False)
        .str.strip()
    )

    attack_map = {
        'BENIGN': 'BENIGN',
        'DDoS': 'DDoS',
        'DoS-Hulk': 'DoS',
        'DoS-GoldenEye': 'DoS',
        'DoS-slowloris': 'DoS',
        'DoS-Slowhttptest': 'DoS',
        'PortScan': 'Port Scan',
        'FTP-Patator': 'Brute Force',
        'SSH-Patator': 'Brute Force',
        'Bot': 'Bot',
        'Web-Attack-Brute-Force': 'Web Attack',
        'Web-Attack-XSS': 'Web Attack',
        'Web-Attack-Sql-Injection': 'Web Attack',
        'Infiltration': 'Infiltration',
        'Heartbleed': 'Heartbleed'
    }

    df['Attack Type'] = df['Label'].map(attack_map)
    df['AttackBinary'] = (df['Label'] != 'BENIGN').astype(int)

    return df


# -----------------------------
# Missing & Infinite Handling
# -----------------------------
def handle_missing_and_infinite(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include=np.number).columns

    # Replace ±inf with NaN
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)

    # Drop rows with NaN in critical rate features
    critical_cols = ['Flow Bytes/s', 'Flow Packets/s']
    df.dropna(subset=[c for c in critical_cols if c in df.columns], inplace=True)

    return df


# -----------------------------
# Main Preprocessing Function
# -----------------------------
def preprocess_csv(csv_path: str) -> pd.DataFrame:
    print(f"Loading data from {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)

    print("Initial shape:", df.shape)

    df.drop_duplicates(inplace=True)
    df = clean_labels(df)
    df = handle_missing_and_infinite(df)

    print("Final shape after cleaning:", df.shape)
    return df


# -----------------------------
# Save to PostgreSQL
# -----------------------------
def save_to_postgres(df: pd.DataFrame):
    engine = create_engine(DB_URI)
    df.to_sql(TABLE_NAME, engine, if_exists='replace', index=False)
    print(f"Data saved to PostgreSQL table '{TABLE_NAME}'")


# -----------------------------
# Script Entry Point
# -----------------------------
if __name__ == "__main__":
    INPUT_CSV = "data/raw/cic_ids.csv"

    df_cleaned = preprocess_csv(INPUT_CSV)
    save_to_postgres(df_cleaned)
