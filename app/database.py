from typing import Annotated
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from fastapi import Depends
from functools import lru_cache


from pydantic_settings import BaseSettings, SettingsConfigDict




class Settings(BaseSettings):
    DEBUG_DATABASE_URL: str
    DATABASE_URL: str
    DEBUG: str
    GROQ_API: str
    HUGGINGFACE_API_KEY:str
    MONGO_DB: str
    DEBUG_MONGO_DB: str
    AWS_ACCESS_KEY_ID: str
    AWS_SECRET_ACCESS_KEY: str
    S3_BUCKET_NAME: str

    model_config = SettingsConfigDict(env_file=".env")


@lru_cache
def get_settings():
    return Settings()


DEBUG = int(get_settings().DEBUG)
MONGO_DB_URL = get_settings().MONGO_DB if not DEBUG else get_settings().DEBUG_MONGO_DB

if DEBUG:
    SQLALCHEMY_DATABASE_URL =  get_settings().DEBUG_DATABASE_URL
    engine = create_engine(
        SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
    )
else:
    SQLALCHEMY_DATABASE_URL =  get_settings().DATABASE_URL

    engine = create_engine(
        SQLALCHEMY_DATABASE_URL
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
