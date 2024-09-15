from app.database import get_db
from app.ingestion.utils import Ingestion
from app.utils.auth import auth_dependency
from sqlalchemy.orm import Session
from fastapi import Body, APIRouter, UploadFile, Depends, File, HTTPException
from pydantic import BaseModel
from datetime import datetime
from typing import Tuple, Annotated, List
#-----------------------------------
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from app.models import models
import os

router = APIRouter(tags=["file"])

# At the start of your FastAPI app or in the file upload function
upload_dir = "uploads"
os.makedirs(upload_dir, exist_ok=True)



chats = {}

class FileUploadResponse(BaseModel):
    id: int
    filename: str
    content_type: str
    size: int
    upload_date: datetime
    file_path: str
    owner_id: int



class FileUploadRequest(BaseModel):
    id: int
    filename: str
    content_type: str
    size: int
    upload_date: datetime
    file_path: str
    owner_id: int

    class Config:
        orm_mode = True


@router.post('/file', response_model=FileUploadResponse)
async def upload_file(file: UploadFile = File(...), user = Depends(auth_dependency), db: Session = Depends(get_db)):
    # Create file path
    file_location = os.path.join(upload_dir, file.filename)

    # Save file
    with open(file_location, "wb+") as file_object:
        file_object.write(file.file.read())

    # Create database entry
    db_file = models.File(
        filename=file.filename,
        file_path=file_location,
        content_type=file.content_type,
        size=os.path.getsize(file_location),
        owner_id=user.id
    )
    db.add(db_file)
    db.commit()
    db.refresh(db_file)

    #chroma db
    ingestion = Ingestion()
    ingestion.get_or_create_collection('embeddings')

    # Add a file (test)
    await ingestion.add_file(file_location, user.id)

    return db_file

@router.get("/file", response_model=List[FileUploadResponse], responses={200:{'description':'Success'}, 403:{'description':'Unauthorised'}})
def read_user_files(user = Depends(auth_dependency), skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    files = db.query(models.File).filter(models.File.owner_id == user.id).offset(skip).limit(limit).all()
    return files


@router.delete("/file/{file_id}", responses={200:{'description':'Success'}, 400:{'description':'Bad Request'}})
def delete_file(file_id: int, user = Depends(auth_dependency),db: Session = Depends(get_db)):
    file = db.query(models.File).filter(models.File.owner_id == user.id, models.File.id == file_id).first()
    # Loop through the files and delete them one by one
    #
    # first remove from vector db
    if file:
        ingestion = Ingestion()
        ingestion.delete_document(file.file_path, user.id)
        db.delete(file)
        db.commit()
        return {'message':'file deleted'}
    else:
        raise HTTPException(status_code=400, detail='Bad Request')
