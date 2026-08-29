from sqlalchemy import Column, Integer, String, Float, DateTime, JSON
from datetime import datetime, timezone
from database import Base

class Analysis(Base):
    __tablename__ = "analyses"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True)
    quality_score = Column(Float)
    quality_label = Column(String)
    issues = Column(JSON)  # Stores the full issues list
    clean_vs_degraded_prob = Column(Float)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
