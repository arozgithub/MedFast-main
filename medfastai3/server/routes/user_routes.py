from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from database import get_db
from models import User

router = APIRouter()

@router.get("/")
async def get_users(db: Session = Depends(get_db)):
    # Implement logic to retrieve users
    users = db.query(User).all()
    return {"users": [{"id": user.id, "username": user.username, "email": user.email} for user in users]}

@router.get("/{user_id}")
async def get_user(user_id: int, db: Session = Depends(get_db)):
    # Implement logic to retrieve a specific user
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        return {"message": "User not found"}
    return {"user": {"id": user.id, "username": user.username, "email": user.email}}
