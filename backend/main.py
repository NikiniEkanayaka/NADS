from fastapi import FastAPI
from database import engine
from models.db_models import Base

app = FastAPI(title="NADS Backend API")

# create table if they do not exist
Base.metadata.create_all(bind=engine)


@app.get("/")
def root():
    return {"message": "NADS Backend Running 🚀"}
