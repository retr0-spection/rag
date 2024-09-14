from app.utils.auth import auth_dependency
from app.utils.encrypt import hash_password, verify_password
from app.utils.jwtman import jwt_manager
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from app.models.models import User
from app.database import db_dependency


router = APIRouter(tags=["user"])

class UserInputBase(BaseModel):
    email: str
    password: str


class LoginBase(BaseModel):
    message: str
    access_token: str


@router.post('/signup')
def create_user(user: UserInputBase, db: db_dependency):
    email = user.email
    hashed_password = hash_password(user.password)
    payload = {'email':email, 'hashed_password':hashed_password}
    user = User(**payload)
    print(type(db))
    print(db)
    db.add(user)
    db.commit()

@router.post('/login', responses={200:{'description':'Authenticated'}, 403:{'description': 'Incorrect user/password'}}, response_model=LoginBase)
def authenticate_user(user: UserInputBase, db: db_dependency):
    email = user.email
    _user = db.query(User).filter(User.email == email).first()
    try:
        if _user:
            if verify_password(user.password, _user.hashed_password):
                token = jwt_manager.issue_token(user_id=_user.id)
                return {'message':'Authenticated', 'access_token':token}
            else:
                raise HTTPException(status_code=403, detail='incorrect user/password')
        else:
            raise HTTPException(status_code=403, detail='incorrect user/password')
    except Exception as e:
        raise HTTPException(status_code=403, detail='incorrect user/password')

@router.post("/renew")
def renew(token: str):
    new_token = jwt_manager.renew_token(token=token)
    return {"access_token": new_token}
