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

    model_config = SettingsConfigDict(env_file=".env")


@lru_cache
def get_settings():
    return Settings()


SQLALCHEMY_DATABASE_URL =  "sqlite:///./sql_app.db" if int(get_settings().DEBUG) else "postgres://rag_muddy_fog_9690:f5yr0NFXEEB39OL@posidon.flycast:5432/rag_muddy_fog_9690?sslmode=disable"

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
