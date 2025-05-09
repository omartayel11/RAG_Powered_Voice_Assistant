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

async def add_recipe_to_favourites(user_email: str, recipe_title: str, recipe_content: str):
    user = await get_user_by_email(user_email)
    if not user:
        return {"status": "error", "message": "User not found."}

    favourites = user.get("favorite_recipes", [])
    for recipe in favourites:
        if recipe["title"] == recipe_title:
            return {"status": "exists", "message": "Recipe already in favourites."}

    new_entry = {"title": recipe_title, "recipe": recipe_content}
    await db.users.update_one(
        {"email": user_email},
        {"$push": {"favorite_recipes": new_entry}}
    )
    return {"status": "success", "message": "Recipe added to favourites."}
