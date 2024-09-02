from sqlalchemy import Column, String, Integer, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime, timedelta
import uuid
from app.database import Base, engine


class APIKey(Base):
    __tablename__ = "api_keys"
    id = Column(String, primary_key=True, index=True, default=lambda: str(uuid.uuid4()))
    key = Column(String, unique=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    permissions = Column(String, nullable=True)
    rate_limit = Column(Integer, default=1000)
    expiry_date = Column(DateTime, default=lambda: datetime.utcnow() + timedelta(days=30))
    created_at = Column(DateTime, default=datetime.utcnow)


    #

    user = relationship("User", back_populates="api_keys")
