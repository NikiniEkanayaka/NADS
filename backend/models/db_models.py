from sqlalchemy import Column, Integer, String, Float, Boolean, ForeignKey, BigInteger, TIMESTAMP
from sqlalchemy.sql import func
from backend.database import Base

class Flow(Base):
    __tablename__ = "flows"

    id = Column(Integer, primary_key=True, index=True)
    src_ip = Column(String(50))
    dst_ip = Column(String(50))
    src_port = Column(Integer)
    dst_port = Column(Integer)
    protocol = Column(String(10))
    bytes_sent = Column(BigInteger)
    packets = Column(Integer)
    duration = Column(Float)
    anomaly_score = Column(Float)
    timestamp = Column(TIMESTAMP, server_default=func.now())


class Alert(Base):
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, index=True)
    flow_id = Column(Integer, ForeignKey("flows.id"))
    score = Column(Float)
    severity = Column(String(10))
    created_at = Column(TIMESTAMP, server_default=func.now())


class Feedback(Base):
    __tablename__ = "feedback"

    id = Column(Integer, primary_key=True, index=True)
    alert_id = Column(Integer, ForeignKey("alerts.id"))
    label = Column(Boolean) 
    analyst = Column(String(100))
    created_at = Column(TIMESTAMP, server_default=func.now())
