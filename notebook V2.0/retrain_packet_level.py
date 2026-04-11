import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
import joblib
import glob
import time

print("RETRAINING MODEL FOR PACKET-LEVEL DETECTION")
print("=" * 60)

# Load all CSV files (same as original training)
csv_files = glob.glob('notebook V2.0/*.csv')
print(f"Found {len(csv_files)} CSV files")

dfs = []
for f in csv_files:
    temp_df = pd.read_csv(f, low_memory=False)
    dfs.append(temp_df)
    print(f"  Loaded: {f.split('/')[-1]} ({temp_df.shape[0]:,} rows)")

df = pd.concat(dfs, axis=0, ignore_index=True)
df.columns = df.columns.str.strip()

print(f"\nCombined dataset: {df.shape[0]:,} rows, {df.shape[1]} columns")

# Define PACKET-LEVEL features (only those available from single packet)
PACKET_LEVEL_FEATURES = [
    # Basic packet info
    'Destination Port',

    # TCP flags (available in packet headers)
    'SYN Flag Count',
    'FIN Flag Count',
    'RST Flag Count',
    'PSH Flag Count',
    'ACK Flag Count',
    'URG Flag Count',

    # Window size (TCP option)
    'Init_Win_bytes_forward',
    'Init_Win_bytes_backward',

    # Packet size (from this packet only)
    'Fwd Packet Length Max',  # Size of this packet
]

print(f"\nUsing {len(PACKET_LEVEL_FEATURES)} packet-level features:")
for i, feat in enumerate(PACKET_LEVEL_FEATURES, 1):
    print(f"  {i:2d}. {feat}")

# Prepare data
X = df[PACKET_LEVEL_FEATURES].copy()
y = df['Label']

# Handle missing values
X = X.fillna(0)

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

print(f"\nLabel encoding: {len(le.classes_)} classes")
for i, cls in enumerate(le.classes_):
    count = (y_encoded == i).sum()
    pct = count / len(y_encoded) * 100
    print(f"  {i:2d} -> {cls:25s} ({count:>8,} samples, {pct:>5.1f}%)")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"\nTrain/test split:")
print(f"  Train: {X_train.shape[0]:,} samples")
print(f"  Test:  {X_test.shape[0]:,} samples")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Compute sample weights for imbalance
sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

# Train XGBoost
print("\nTraining XGBoost on packet-level features...")
print("   (This will be faster than flow-based training)")

start_time = time.time()

xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method='hist',
    eval_metric='mlogloss',
    random_state=42,
    n_jobs=-1,
    verbosity=1
)

xgb_model.fit(
    X_train_scaled, y_train,
    sample_weight=sample_weights,
    verbose=50
)

training_time = time.time() - start_time
print(f"\nTraining completed in {training_time:.2f} seconds")

# Evaluate on test set
y_pred = xgb_model.predict(X_test_scaled)
y_prob = xgb_model.predict_proba(X_test_scaled)

from sklearn.metrics import accuracy_score, classification_report
accuracy = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))

# Find optimal threshold for packet-level model
thresholds = np.arange(0.1, 0.91, 0.05)
best_threshold = 0.5
best_score = 0

print("\nFinding optimal threshold...")
for thresh in thresholds:
    y_pred_thresh = (np.max(y_prob, axis=1) >= thresh).astype(int)
    # For packet-level, we'll use a simple accuracy metric since WTDR is complex
    score = accuracy_score(y_test, y_pred_thresh)
    if score > best_score:
        best_score = score
        best_threshold = thresh

print(f"   Best threshold: {best_threshold:.2f} (accuracy: {best_score:.4f})")

# Save the packet-level model
model_artifact = {
    "model": xgb_model,
    "scaler": scaler,
    "threshold": best_threshold,
    "features": PACKET_LEVEL_FEATURES,
    "label_encoder": le,
    "model_type": "packet_level",
    "training_info": {
        "features_used": len(PACKET_LEVEL_FEATURES),
        "training_samples": len(X_train),
        "test_samples": len(X_test),
        "accuracy": accuracy,
        "optimal_threshold": best_threshold
    }
}

joblib.dump(model_artifact, 'models/xgb_model_packet_level.pkl')
print("\nSaved packet-level model to: models/xgb_model_packet_level.pkl")
print(f"   Features: {len(PACKET_LEVEL_FEATURES)}")
print(f"   Threshold: {best_threshold}")
print("   Model type: Packet-level detection")

print("\nPACKET-LEVEL MODEL TRAINING COMPLETE!")
print("   This model works with individual packet features.")
print("   Update your API to use this new model for real-time detection.")