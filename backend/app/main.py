from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    workplace: str | None = None
    context: str | None = None
    details: str | None = None

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/people")
def create_person(person: Person):
    print(f"Would save: {person.name}, {person.workplace}, {person.context}, {person.details}")
    return {"Created": person.name}

@app.get("/")
def read_root():
    return {"status": "Visage API running"}