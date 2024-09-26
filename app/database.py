from typing import Annotated
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from fastapi import Depends
from functools import lru_cache


from pydantic_settings import BaseSettings, SettingsConfigDict




class Settings(BaseSettings):
    DATABASE_URL: str
    DEBUG: str
    GROQ_API: str
    HUGGINGFACE_API_KEY:str
    MONGO_DB: str

    model_config = SettingsConfigDict(env_file=".env")


@lru_cache
def get_settings():
    return Settings()


SQLALCHEMY_DATABASE_URL =  get_settings().DATABASE_URL

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

db_dependency = Annotated[Session, Depends(get_db)]
