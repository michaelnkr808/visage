from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import scan
from services.database import init_db
import uvicorn
import os

# Initialize database
init_db()

app = FastAPI(title="Visage Face Recognition API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes from scan.py
app.include_router(scan.app, prefix="/api")

@app.get("/")
def root():
    return {"status": "Visage API is running"}

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)