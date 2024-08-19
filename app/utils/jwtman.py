import jwt
import time
from typing import Dict, Optional
from fastapi import HTTPException

class JWTManager:
    def __init__(self, secret_key: str, algorithm: str = "HS256", expiration_minutes: int = 30):
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.expiration_minutes = expiration_minutes
        self.blacklist = set()

    def issue_token(self, user_id: str, additional_claims: Optional[Dict] = None) -> str:
        payload = {
            "sub": user_id,
            "exp": time.time() + self.expiration_minutes * 60,
            "iat": time.time(),
        }
        if additional_claims:
            payload.update(additional_claims)

        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        return token

    def renew_token(self, token: str) -> str:
        try:
            decoded = jwt.decode(token, self.secret_key, algorithms=[self.algorithm], options={"verify_exp": False})
            if token in self.blacklist:
                raise HTTPException(status_code=403, detail="Token has been blacklisted")

            decoded["exp"] = time.time() + self.expiration_minutes * 60
            new_token = jwt.encode(decoded, self.secret_key, algorithm=self.algorithm)
            return new_token
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token has expired")
        except jwt.InvalidTokenError:
            raise HTTPException(status_code=401, detail="Invalid token")

    def blacklist_token(self, token: str):
        self.blacklist.add(token)

    def validate_token(self, token: str) -> Dict:
        try:
            decoded = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            if token in self.blacklist:
                raise HTTPException(status_code=403, detail="Token has been blacklisted")
            return decoded
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token has expired")
        except jwt.InvalidTokenError:
            raise HTTPException(status_code=401, detail="Invalid token")

    def is_blacklisted(self, token: str) -> bool:
        return token in self.blacklist

jwt_manager = JWTManager(secret_key="your_secret_key")
