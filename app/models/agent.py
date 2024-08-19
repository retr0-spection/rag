from sqlalchemy import Boolean, Column, ForeignKey, Integer, String, DateTime, JSON, Text
from sqlalchemy.orm import relationship
from datetime import datetime
from sqlalchemy.sql import func

from app.database import Base, engine

class AIAgent(Base):
    __tablename__ = 'ai_agents'

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), nullable=False)
    agent_type = Column(String(50), nullable=False)  # e.g., 'chatbot', 'recommendation', 'nlp'
    configuration = Column(JSON, nullable=True)  # stores agent-specific configuration
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)

    def __repr__(self):
        return f"<AIAgent(name={self.name}, type={self.agent_type})>"
