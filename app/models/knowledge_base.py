from sqlalchemy import Column, Integer, String, ForeignKey, Text
from sqlalchemy.orm import relationship
from app.database import Base

class KnowledgeBase(Base):
    __tablename__ = 'knowledge_bases'

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    description = Column(Text)
    data = Column(Text)  # You might want to store data elsewhere and just reference it here.
    # Relationships

    agents = relationship('AgentKnowledgeBaseAssociation', back_populates='knowledge_base')
