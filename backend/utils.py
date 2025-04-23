from passlib.context import CryptContext
from bson.objectid import ObjectId
from db import db

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

async def get_user_by_email(email: str):
    return await db.users.find_one({"email": email})

async def create_user(data: dict):
    data["password"] = hash_password(data["password"])
    result = await db.users.insert_one(data)
    return str(result.inserted_id)
