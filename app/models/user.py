from sqlalchemy import Boolean, Column, ForeignKey, Integer, String
from sqlalchemy.orm import relationship

from app.database import Base, engine

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)


    #relationships
    files = relationship("File", back_populates="owner")
    agents = relationship("Agent", back_populates="user")
    sessions = relationship("Session", back_populates="user")
