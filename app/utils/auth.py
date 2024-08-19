from app.utils.jwtman import jwt_manager
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import Depends, HTTPException
from sqlalchemy.orm import Session
from app.models.models import User
from app.database import get_db


security = HTTPBearer()

def get_user(db: Session, jwt_id: int):
    return db.query(User).filter(User.id == jwt_id).first()

async def auth_dependency(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
):
    jwt = jwt_manager.validate_token(credentials.credentials)
    if not jwt:
        raise HTTPException(status_code=401, detail="Invalid token")
    user = get_user(db, jwt['sub'])
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user
