from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from services.database import save_person_info

class Person(BaseModel):
    name: str | None = None
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
    try:
        person_id = save_person_info(
            face_id=None,  # No face yet
            name=person.name,
            conversation_context=f"Workplace: {person.workplace} | Context: {person.context} | Details: {person.details}"
        )
        print(f"✅ Saved person #{person_id}: {person.name}")
        return {"created": person.name, "id": person_id}
    except Exception as e:
        print(f"❌ Failed to save: {e}")
        return {"error": str(e)}, 500

@app.get("/")
def read_root():
    return {"status": "Visage API running"}