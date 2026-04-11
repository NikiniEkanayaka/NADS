from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from pydantic import BaseModel

from backend.database import SessionLocal
from backend.models.db_models import User
from backend.core.security import create_access_token, hash_password, verify_password
from fastapi import Depends
from backend.core.deps import require_role
from pydantic import BaseModel

class SignupInput(BaseModel):
    name: str
    username: str
    password: str
    role: str

class UpdateProfileInput(BaseModel):
    name: str | None = None
    password: str | None = None

router = APIRouter(prefix="/auth", tags=["Auth"])


class LoginInput(BaseModel):
    username: str
    password: str


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.post("/login")
def login(data: LoginInput, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == data.username).first()

    if not user:
        raise HTTPException(status_code=401, detail="Invalid username")

    if not verify_password(data.password, user.password):
        raise HTTPException(status_code=401, detail="Invalid password")

    token = create_access_token({
        "sub": user.username,
        "role": user.role
    })

    return {
        "access_token": token,
        "token_type": "bearer",
        "role": user.role
    }



@router.post("/signup")
def signup(
    data: SignupInput,
    db: Session = Depends(get_db),
    current_user=Depends(require_role(["admin"]))  # Only admin can add users
):
    # Check if username already exists
    existing_user = db.query(User).filter(User.username == data.username).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Username already exists")

    # Only allow "admin" or "analyst"
    if data.role not in ["admin", "analyst"]:
        raise HTTPException(status_code=400, detail="Invalid role")

    # Hash password before storing
    from backend.core.security import hash_password
    hashed_password = hash_password(data.password)

    new_user = User(
        name=data.name,
        username=data.username,
        password=hashed_password,
        role=data.role
    )

    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    return {
        "message": f"User '{new_user.username}' created successfully",
        "role": new_user.role
    }



@router.put("/profile")
def update_profile(
    data: UpdateProfileInput,
    db: Session = Depends(get_db),
    current_user=Depends(require_role(["admin", "analyst"]))  # both roles allowed
):
    # 🔍 Get current logged-in user
    user = db.query(User).filter(User.username == current_user["sub"]).first()

    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # ✏️ Update name (if provided)
    if data.name:
        user.name = data.name

    # 🔐 Update password (if provided)
    if data.password:
        user.password = hash_password(data.password)

    db.commit()
    db.refresh(user)

    return {
        "message": "Profile updated successfully",
        "username": user.username,
        "name": user.name
    }