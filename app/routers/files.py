import os
import tempfile
from typing import List, Tuple
from datetime import datetime

import boto3
from botocore.exceptions import ClientError
from fastapi import APIRouter, Depends, File, UploadFile, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel

from app.database import DEBUG, get_db, get_settings
from app.ingestion.utils import Ingestion
from app.utils.auth import auth_dependency
from app.models import models

router = APIRouter(tags=["file"])

# Environment variables
S3_BUCKET = get_settings().S3_BUCKET_NAME
AWS_ACCESS_KEY = get_settings().AWS_ACCESS_KEY_ID
AWS_SECRET_KEY = get_settings().AWS_SECRET_ACCESS_KEY

# Setup for local storage
upload_dir = "uploads"
os.makedirs(upload_dir, exist_ok=True)

# Setup for S3
if not DEBUG:
    s3_client = boto3.client(
        's3',
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
        region_name='af-south-1'
    )

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

def upload_file_to_s3(file_content: bytes, object_name: str) -> Tuple[str, int]:
    try:
        file_size = len(file_content)
        s3_client.put_object(Body=file_content, Bucket=S3_BUCKET, Key=object_name)
        return f"s3://{S3_BUCKET}/{object_name}", file_size
    except ClientError as e:
        print(f"Error uploading file to S3: {e}")
        raise HTTPException(status_code=500, detail="Failed to upload file to S3")

@router.post('/file', response_model=FileUploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    user = Depends(auth_dependency),
    db: Session = Depends(get_db)
):
    try:
        # Read file content
        file_content = await file.read()
        file_size = len(file_content)

        if not DEBUG:
            # S3 upload
            s3_object_name = f"user_{user.id}/{file.filename}"
            file_location, _ = upload_file_to_s3(file_content, s3_object_name)
        else:
            # Local file upload
            file_location = os.path.join(upload_dir, file.filename)
            with open(file_location, "wb") as file_object:
                file_object.write(file_content)

        # Create database entry
        db_file = models.File(
            filename=file.filename,
            file_path=file_location,
            content_type=file.content_type,
            size=file_size,
            owner_id=user.id
        )
        db.add(db_file)
        db.commit()
        db.refresh(db_file)

        # Create temporary file with the same filename as the original
        temp_dir = tempfile.mkdtemp()
        temp_file_path = os.path.join(temp_dir, file.filename)

        try:
            with open(temp_file_path, 'wb') as temp_file:
                temp_file.write(file_content)

            # Chroma DB ingestion
            ingestion = Ingestion()
            ingestion.get_or_create_collection('embeddings')
            await ingestion.add_file(temp_file_path, user.id)
        finally:
            # Clean up temporary file and directory
            os.remove(temp_file_path)
            os.rmdir(temp_dir)

        return db_file
    except Exception as e:
        print(f"Error in upload_file: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/file", response_model=List[FileUploadResponse])
def read_user_files(
    user = Depends(auth_dependency),
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    files = db.query(models.File).filter(models.File.owner_id == user.id).offset(skip).limit(limit).all()
    return files

@router.delete("/file/{file_id}")
def delete_file(
    file_id: int,
    user = Depends(auth_dependency),
    db: Session = Depends(get_db)
):
    file = db.query(models.File).filter(models.File.owner_id == user.id, models.File.id == file_id).first()
    if not file:
        raise HTTPException(status_code=404, detail="File not found")

    # Delete from storage
    if not DEBUG:
        try:
            s3_object_name = file.file_path.split(f"s3://{S3_BUCKET}/")[1]
            s3_client.delete_object(Bucket=S3_BUCKET, Key=s3_object_name)
        except ClientError as e:
            print(f"Error deleting file from S3: {e}")
            raise HTTPException(status_code=500, detail="Failed to delete file from S3")
    else:
        if os.path.exists(file.file_path):
            os.remove(file.file_path)

    # Delete from vector DB
    ingestion = Ingestion()
    ingestion.delete_document(file.filename, user.id)

    # Delete from database
    db.delete(file)
    db.commit()

    return {'message': 'File deleted'}
