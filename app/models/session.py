from sqlalchemy import Column, Integer, String, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from app.database import Base
from datetime import datetime

class Session(Base):
    __tablename__ = 'sessions'

    id = Column(Integer, primary_key=True, index=True)
    agent_id = Column(Integer, ForeignKey('agents.id'), nullable=False)
    user_id = Column(String, ForeignKey('users.id'))  # Assuming you track users by an ID or handle
    start_time = Column(DateTime, default=datetime.utcnow)
    end_time = Column(DateTime, nullable=True)

    # Relationships
    agent = relationship('Agent', back_populates='sessions')
    user = relationship('User', back_populates='sessions')
    messages = relationship('Message', back_populates='session')
