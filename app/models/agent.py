from sqlalchemy import Boolean, Column, ForeignKey, Integer, String, DateTime, JSON, Text
from sqlalchemy.orm import relationship
from datetime import datetime
from sqlalchemy.sql import func

from app.database import Base, engine

class Agent(Base):
    __tablename__ = 'agents'

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), nullable=False)
    agent_type = Column(String(50), nullable=False, default='chatbot')  # e.g., 'chatbot', 'recommendation', 'nlp'
    user_id = Column(Integer, ForeignKey('users.id'))  # Assuming you track users by an ID or handle
    # project_id = Column(Integer, ForeignKey("projects.id"), nullable=False)
    configuration = Column(JSON, nullable=False)  # stores agent-specific configuration
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)

      # Relationships
    knowledge_bases = relationship('AgentKnowledgeBaseAssociation', back_populates='agent')
    sessions = relationship('Session', back_populates='agent')
    user = relationship("User", back_populates="agents")
    # project = relationship("Project", back_populates="agents")

    def __repr__(self):
        return f"<Agent(name={self.name}, type={self.agent_type})>"

class AgentKnowledgeBaseAssociation(Base):
    __tablename__ = 'agent_knowledge_base_association'

    agent_id = Column(Integer, ForeignKey('agents.id'), primary_key=True)
    knowledge_base_id = Column(Integer, ForeignKey('knowledge_bases.id'), primary_key=True)

    # Relationships
    agent = relationship('Agent', back_populates='knowledge_bases')
    knowledge_base = relationship('KnowledgeBase', back_populates='agents')
