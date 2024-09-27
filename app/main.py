from typing import Annotated, Union

from app.utils.jwtman import JWTManager
from fastapi import FastAPI, Depends
from app.database import engine
from app.routers import users, chat, files, agent, suggestions
from app.models import models
from sqlalchemy.orm import Session
import logging
import os
from fastapi.middleware.cors import CORSMiddleware

logging.getLogger('passlib').setLevel(logging.ERROR)



app = FastAPI()
origins = [
    "http://localhost",
    "http://localhost:8080",
    "http://localhost:8080",
    "http://localhost:5173",
    "http://ec2-13-245-211-234.af-south-1.compute.amazonaws.com",
    "https://ec2-13-245-211-234.af-south-1.compute.amazonaws.com",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



models.Base.metadata.create_all(engine)





app.include_router(users.router)
app.include_router(chat.router)
app.include_router(files.router)
app.include_router(agent.router)
app.include_router(suggestions.router)
