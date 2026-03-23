from fastapi import FastAPI
from backend.database import engine, Base
from backend.routes.predict import router as predict_router
from backend.routes.auth import router as auth_router

app = FastAPI(title="NADS Backend API")

# Create tables
Base.metadata.create_all(bind=engine)

# Register routes
app.include_router(predict_router, tags=["NADS"])
app.include_router(auth_router, tags=["NADS"])

@app.get("/")
def root():
    return {"message": "NADS Backend Running 🚀"}

