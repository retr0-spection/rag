from typing import Annotated, Union

from app.utils.jwtman import JWTManager
from fastapi import FastAPI, Depends
from app.database import engine
from app.routers import users, chat, files, agent
from app.models import models
from sqlalchemy.orm import Session
import logging
import os
import nltk

logging.getLogger('passlib').setLevel(logging.ERROR)



nltk.download('stopwords')
# Download NLTK data (run this once)
nltk.download('punkt')

app = FastAPI()



models.Base.metadata.create_all(engine)





app.include_router(users.router)
app.include_router(chat.router)
app.include_router(files.router)
app.include_router(agent.router)
