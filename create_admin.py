import asyncio
from datetime import datetime
from core.database import users_collection
from core.auth import get_password_hash

async def create_admin():
    email = "admin@gmail.com"
    password = "12345678"
    name = "Admin User"

    print(f"Checking for existing user with email: {email}")
    existing_user = await users_collection.find_one({"email": email})
    
    if existing_user:
        print(f"User {email} found. Deleting...")
        await users_collection.delete_one({"email": email})
        print(f"Deleted existing user.")

    print(f"Creating new admin user: {email}")
    hashed_password = get_password_hash(password)
    
    new_user = {
        "email": email,
        "name": name,
        "password": hashed_password,
        "is_admin": True,
        "created_at": datetime.now(),
        "updated_at": datetime.now()
    }
    
    result = await users_collection.insert_one(new_user)
    print(f"Admin user created successfully with ID: {result.inserted_id}")

if __name__ == "__main__":
    asyncio.run(create_admin())
