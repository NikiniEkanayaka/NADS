from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
import numpy as np
import joblib

from backend.models.db_models import Flow, Alert, Feedback
from backend.database import SessionLocal
from backend.core.model_loader import model, scaler, threshold, features

router = APIRouter()


# --- Input models ---
class FlowInput(BaseModel):
    bytes: int
    packets: int
    duration: float
    unique_dst_ports: int = 1  # optional, default 1
    src_ip: str = "0.0.0.0"    # optional placeholder
    dst_ip: str = "0.0.0.0"
    src_port: int = 0
    dst_port: int = 0
    protocol: str = "TCP"


class FeedbackInput(BaseModel):
    alert_id: int
    label: bool
    analyst: str


# --- DB session dependency ---
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# --- Feature mapping ---
def map_flow_to_model_features(flow: FlowInput) -> np.ndarray:
    """
    Map minimal API fields to all features expected by the model.
    Uses approximations for missing values.
    Returns a 1xN numpy array.
    """
    bytes_sent = flow.bytes
    packets = flow.packets
    duration = flow.duration
    unique_dst_ports = flow.unique_dst_ports

    # Approximate features
    fwd_packets = packets
    bwd_packets = 0
    fwd_packets_per_s = fwd_packets / max(duration, 1e-6)
    bwd_packets_per_s = 0
    min_packet_len = bytes_sent / max(packets, 1)
    max_packet_len = bytes_sent / max(packets, 1)
    avg_packet_size = bytes_sent / max(packets, 1)
    down_up_ratio = 0

    # Flags and other features (set to 0 if unknown)
    syn_flag = 0
    fin_flag = 0
    rst_flag = 0
    psh_flag = 0
    ack_flag = 0
    urg_flag = 0
    init_win_fwd = 0
    init_win_bwd = 0
    avg_fwd_seg = 0
    avg_bwd_seg = 0
    fwd_header_len = 0
    bwd_header_len = 0

    # Build array in the same order as SELECTED_FEATURES / model.features
    feature_vector = np.array([
        flow.dst_port,
        duration,
        fwd_packets,
        bwd_packets,
        down_up_ratio,
        avg_packet_size,
        avg_packet_size,  # Packet Length Mean
        0,                # Packet Length Std (approx)
        min_packet_len,
        max_packet_len,
        fwd_packets_per_s,
        bwd_packets_per_s,
        syn_flag,
        fin_flag,
        rst_flag,
        psh_flag,
        ack_flag,
        urg_flag,
        init_win_fwd,
        init_win_bwd,
        avg_fwd_seg,
        avg_bwd_seg,
        unique_dst_ports,
        fwd_header_len,
        bwd_header_len,
        fwd_packets,
        bwd_packets
    ]).reshape(1, -1)

    return feature_vector


# --- Predict endpoint ---
@router.post("/predict")
def predict(flow: FlowInput, db: Session = Depends(get_db)):
    try:
        if model is None:
            raise HTTPException(status_code=500, detail="Model not loaded")

        # 🔢 Map API fields -> model features
        x = map_flow_to_model_features(flow)

        # 📐 Scale features
        x_scaled = scaler.transform(x)

        # 🧠 Model scoring
        score = float(model.decision_function(x_scaled)[0])
        is_anomaly = score >= threshold

        # 🚨 Save flow in DB
        db_flow = Flow(
            src_ip=flow.src_ip,
            dst_ip=flow.dst_ip,
            src_port=flow.src_port,
            dst_port=flow.dst_port,
            protocol=flow.protocol,
            bytes_sent=flow.bytes,
            packets=flow.packets,
            duration=flow.duration,
            anomaly_score=score
        )
        db.add(db_flow)
        db.commit()
        db.refresh(db_flow)

        # 🚨 Save alert
        severity = "high" if is_anomaly else "low"
        alert = Alert(flow_id=db_flow.id, score=score, severity=severity)
        db.add(alert)
        db.commit()

        return {
            "score": score,
            "threshold": threshold,
            "status": "anomalous" if is_anomaly else "normal"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- Alerts endpoint ---
@router.get("/alerts")
def get_alerts(db: Session = Depends(get_db)):
    return db.query(Alert).order_by(Alert.created_at.desc()).all()


# --- Feedback endpoint ---
@router.post("/feedback")
def submit_feedback(data: FeedbackInput, db: Session = Depends(get_db)):
    feedback = Feedback(**data.model_dump())
    db.add(feedback)
    db.commit()
    return {"message": "Feedback saved successfully"}
