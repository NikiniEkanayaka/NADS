from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
import numpy as np
import pandas as pd

from backend.core.deps import require_role
from backend.database import SessionLocal
from backend.models.db_models import Flow, Alert, Feedback
from backend.core.model_loader import model, scaler, threshold, features

router = APIRouter()

# ---------------- INPUT MODELS ---------------- #

class FlowInput(BaseModel):
    bytes: int
    packets: int
    duration: float
    unique_dst_ports: int = 1
    src_ip: str = "0.0.0.0"
    dst_ip: str = "0.0.0.0"
    src_port: int = 0
    dst_port: int = 0
    protocol: str = "TCP"


class FeedbackInput(BaseModel):
    alert_id: int
    label: bool
    analyst: str


# ---------------- DB SESSION ---------------- #

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ---------------- FEATURE MAPPING ---------------- #

def map_flow_to_model_features(flow: FlowInput) -> np.ndarray:
    """
    Map API input to the exact feature vector used during model training.
    Must match SELECTED_FEATURES from the XGBoost notebook.
    """

    # Calculate derived features to match training data
    flow_duration = flow.duration
    total_fwd_packets = flow.packets
    total_bwd_packets = 0  # Assume unidirectional flow from API
    down_up_ratio = 0  # No backward packets

    # Packet length features (assume all packets are forward)
    avg_packet_size = flow.bytes / max(flow.packets, 1)
    packet_length_mean = avg_packet_size
    packet_length_std = 0  # No variance info from API
    min_packet_length = avg_packet_size
    max_packet_length = avg_packet_size
    packet_length_variance = 0  # No variance info

    # Rate features
    fwd_packets_per_sec = flow.packets / max(flow.duration, 1e-6)
    bwd_packets_per_sec = 0  # No backward packets

    # TCP flags (no flag info from API, assume defaults)
    syn_flag_count = 0
    fin_flag_count = 0
    rst_flag_count = 0
    psh_flag_count = 0
    ack_flag_count = 0
    urg_flag_count = 0

    # Window and segment size features (no info from API)
    init_win_bytes_forward = 0
    init_win_bytes_backward = 0
    avg_fwd_segment_size = 0
    avg_bwd_segment_size = 0

    # Port and header features
    destination_port = flow.dst_port
    fwd_header_length = 0  # No header info
    bwd_header_length = 0  # No backward packets

    # Subflow features
    subflow_fwd_packets = flow.packets
    subflow_bwd_packets = 0

    # Build feature vector in exact order as SELECTED_FEATURES
    feature_vector = [
        flow_duration,           # 'Flow Duration'
        total_fwd_packets,       # 'Total Fwd Packets'
        total_bwd_packets,       # 'Total Backward Packets'
        down_up_ratio,           # 'Down/Up Ratio'
        avg_packet_size,         # 'Average Packet Size'
        packet_length_mean,      # 'Packet Length Mean'
        packet_length_std,       # 'Packet Length Std'
        min_packet_length,       # 'Min Packet Length'
        max_packet_length,       # 'Max Packet Length'
        packet_length_variance,  # 'Packet Length Variance'
        fwd_packets_per_sec,     # 'Fwd Packets/s'
        bwd_packets_per_sec,     # 'Bwd Packets/s'
        syn_flag_count,          # 'SYN Flag Count'
        fin_flag_count,          # 'FIN Flag Count'
        rst_flag_count,          # 'RST Flag Count'
        psh_flag_count,          # 'PSH Flag Count'
        ack_flag_count,          # 'ACK Flag Count'
        urg_flag_count,          # 'URG Flag Count'
        init_win_bytes_forward,  # 'Init_Win_bytes_forward'
        init_win_bytes_backward, # 'Init_Win_bytes_backward'
        avg_fwd_segment_size,    # 'Avg Fwd Segment Size'
        avg_bwd_segment_size,    # 'Avg Bwd Segment Size'
        destination_port,        # 'Destination Port'
        fwd_header_length,       # 'Fwd Header Length'
        bwd_header_length,       # 'Bwd Header Length'
        subflow_fwd_packets,     # 'Subflow Fwd Packets'
        subflow_bwd_packets      # 'Subflow Bwd Packets'
    ]

    return np.array(feature_vector).reshape(1, -1)


# ---------------- PREDICT ---------------- #

@router.post("/predict")
def predict(
    flow: FlowInput,
    db: Session = Depends(get_db),
    user=Depends(require_role(["admin", "analyst"]))
):
    try:
        if model is None:
            raise HTTPException(status_code=500, detail="Model not loaded")

        # Feature mapping
        x = map_flow_to_model_features(flow)

        # Convert to DataFrame (fix warning)
        x_df = pd.DataFrame(x, columns=features)

        # Scaling
        x_scaled = scaler.transform(x_df)

        # Prediction (XGBoost) with threshold logic
        probs = model.predict_proba(x_scaled)[0]
        max_prob = float(np.max(probs))

        # Apply threshold: if confidence >= threshold, use predicted class, else BENIGN
        if max_prob >= threshold:
            pred_class = int(np.argmax(probs))
            confidence = max_prob
        else:
            pred_class = 0  # BENIGN class
            confidence = max_prob

        # Define anomaly (anything that's not BENIGN)
        is_anomaly = pred_class != 0

        # Save flow
        db_flow = Flow(
            src_ip=flow.src_ip,
            dst_ip=flow.dst_ip,
            src_port=flow.src_port,
            dst_port=flow.dst_port,
            protocol=flow.protocol,
            bytes_sent=flow.bytes,
            packets=flow.packets,
            duration=flow.duration,
            anomaly_score=confidence
        )
        db.add(db_flow)
        db.commit()
        db.refresh(db_flow)

        # Save alert
        severity = "high" if is_anomaly else "low"

        alert = Alert(
            flow_id=db_flow.id,
            score=confidence,
            severity=severity
        )
        db.add(alert)
        db.commit()

        return {
            "prediction": pred_class,
            "confidence": confidence,
            "status": "anomalous" if is_anomaly else "normal"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------- ALERTS ---------------- #

@router.get("/alerts")
def get_alerts(
    db: Session = Depends(get_db),
    user=Depends(require_role(["admin", "analyst"]))
):
    return db.query(Alert).order_by(Alert.created_at.desc()).all()


# ---------------- FEEDBACK ---------------- #

@router.post("/feedback")
def submit_feedback(
    data: FeedbackInput,
    db: Session = Depends(get_db),
    user=Depends(require_role(["analyst"]))
):
    feedback = Feedback(**data.model_dump())
    db.add(feedback)
    db.commit()

    return {"message": "Feedback saved successfully"}