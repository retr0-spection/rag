from fastapi import APIRouter

router = APIRouter(tags=["users"])


@router.get("/users/")
async def read_users():
    return [{"username": "Rick"}, {"username": "Morty"}]


@router.get("/users/me")
async def read_user_me():
    return {"username": "fakecurrentuser"}


@router.get("/users/{username}")
async def read_user(username: str):
    return {"username": username}
