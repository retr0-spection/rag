from app.utils.auth import auth_dependency
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from app.database import get_db
from app.models.models import Agent, User  # Import your SQLAlchemy models

router = APIRouter(tags=["api_keys"])
