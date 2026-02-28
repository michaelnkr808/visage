from fastapi import APIRouter, HTTPException, Form, Header
from pydantic import BaseModel
from services.database import (
    save_photo,
    save_detected_face,
    save_face_encoding,
    link_encoding_to_person,
    save_person_info,
    save_transcript,
    find_matching_face,
    get_person_info_by_name,
    update_person_last_seen,
    delete_person_by_name
)
from services.face_detection import detect_and_encode_face
import base64
from config import config

# ==================== PYDANTIC MODELS ====================

class TranscriptData(BaseModel):
    raw_text: str
    extracted_name: str | None = None
    context: str | None = None

app = APIRouter()

# ==================== AUTH MIDDLEWARE ====================

def verify_auth_token(authorization: str | None = Header(None)):
    """Verify the authorization token matches the backend secret"""
    if not authorization or authorization != f"Bearer {config.BACKEND_AUTH_TOKEN}":
        raise HTTPException(status_code=401, detail="Unauthorized: Invalid or missing auth token")

# ==================== ROUTES ====================

@app.get("/")
def read_root():
    return {
        "status": "Visage API running",
        "version": "1.0.0",
        "endpoints": {
            "POST /workflow1/first-meeting": "Capture photo + name for first meeting",
            "POST /workflow2/recognize": "Recognize person from photo",
            "GET /people/search": "Search for person by name",
            "POST /transcript": "Save conversation transcript"
        }
    }

# ==================== WORKFLOW 1: FIRST MEETING ====================

@app.post("/workflow1/first-meeting")
async def first_meeting(
    image_data: str = Form(...),  # Base64-encoded image from glasses
    user_id: str = Form(...),
    name: str = Form(""),
    conversation_context: str = Form(""),
    authorization: str | None = Header(None)
):
    """
    Workflow 1: First time meeting someone
    - Receives base64-encoded image from MentraLive glasses
    - Optionally provide name from voice transcription
    - Detect face, generate encoding, store in database
    """
    verify_auth_token(authorization)
    try:
        # Convert base64 to bytes
        image_bytes = base64.b64decode(image_data)
        
        # Convert empty strings to None
        name = name if name else None
        conversation_context = conversation_context if conversation_context else None
        
        # Save photo to database
        photo_id = save_photo(
            filename="glasses_capture.jpg",
            image_data=image_bytes,
            user_id=user_id
        )
        print(f"✅ Saved photo #{photo_id}")
        
        # Detect face and generate encoding using InsightFace
        face_result = detect_and_encode_face(image_bytes)
        
        if not face_result:
            raise HTTPException(status_code=400, detail="No face detected in image")
        
        # Save detected face
        face_id = save_detected_face(
            photo_id=photo_id,
            x=face_result['bbox']['x'],
            y=face_result['bbox']['y'],
            width=face_result['bbox']['w'],
            height=face_result['bbox']['h'],
            face_image_data=face_result.get('cropped_face'),
            confidence=face_result.get('confidence')
        )
        print(f"✅ Saved detected face #{face_id}")
        
        # Save face encoding (512-d vector)
        encoding_id = save_face_encoding(
            face_id=face_id,
            encoding=face_result['encoding'],
            model_name="InsightFace"
        )
        print(f"✅ Saved face encoding #{encoding_id}")
        
        # Save person info
        person_info_id = save_person_info(
            face_id=face_id,
            user_id=user_id,
            name=name,
            conversation_context=conversation_context
        )
        print(f"✅ Saved person info #{person_info_id}: {name}")

        # Link encoding to person so find_matching_face can group by person
        link_encoding_to_person(encoding_id, person_info_id)
        print(f"✅ Linked encoding #{encoding_id} to person #{person_info_id}")

        return {
            "success": True,
            "message": f"Successfully registered {name or 'unknown person'}",
            "data": {
                "photo_id": photo_id,
                "face_id": face_id,
                "person_info_id": person_info_id,
                "name": name
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error in first_meeting: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== WORKFLOW 2: RECOGNIZE PERSON ====================

@app.post("/workflow2/recognize")
async def recognize_person(
    image_data: str = Form(...),  # Base64-encoded image
    user_id: str = Form(...),
    authorization: str | None = Header(None)
):
    """
    Workflow 2: Recognize someone you've met before
    - Receives base64-encoded image from MentraLive glasses
    - Match face against stored encodings
    - Return person's info if match found
    """
    verify_auth_token(authorization)
    try:
        # Convert base64 to bytes
        image_bytes = base64.b64decode(image_data)
        
        # Detect face and generate encoding
        face_result = detect_and_encode_face(image_bytes)
        
        if not face_result:
            raise HTTPException(status_code=400, detail="No face detected in image")
        
        query_encoding = face_result['encoding']

        # Find matching person — groups by person_info_id across all their stored encodings
        person_info, distance = find_matching_face(query_encoding, user_id=user_id, threshold=config.FACE_MATCH_THRESHOLD)

        if not person_info:
            return {
                "success": True,
                "recognized": False,
                "message": "Haven't met this person before",
                "distance": float(distance) if distance else None
            }

        # Update last seen
        update_person_last_seen(person_info.id)
        
        return {
            "success": True,
            "recognized": True,
            "distance": float(distance),
            "person": {
                "name": person_info.name,
                "conversation_context": person_info.conversation_context,
                "first_met_at": person_info.first_met_at.isoformat(),
                "last_seen_at": person_info.last_seen_at.isoformat(),
                "times_met": person_info.times_met
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error in recognize_person: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== WORKFLOW 3: QUERY PERSON BY NAME ====================

@app.get("/people/search")
async def search_person_by_name(
    name: str,
    user_id: str,
    authorization: str | None = Header(None)
):
    """
    Workflow 3: Query person information by name
    - Search for person by name (case-insensitive, partial match)
    - Return person's info if found
    """
    verify_auth_token(authorization)
    try:
        if not name or name.strip() == "":
            raise HTTPException(status_code=400, detail="Name parameter is required")
        
        # Search for person by name
        person_info = get_person_info_by_name(name.strip(), user_id=user_id)
        
        if not person_info:
            raise HTTPException(status_code=404, detail=f"No person found with name: {name}")
        
        return {
            "name": person_info.name,
            "conversation_context": person_info.conversation_context,
            "first_met_at": person_info.first_met_at.isoformat() if person_info.first_met_at else None,
            "last_seen_at": person_info.last_seen_at.isoformat() if person_info.last_seen_at else None,
            "times_met": person_info.times_met
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error in search_person_by_name: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== WORKFLOW 4: DELETE PERSON BY NAME ====================

@app.delete("/people/delete")
async def delete_person(
    name: str,
    user_id: str,
    authorization: str | None = Header(None)
):
    """
    Workflow 4: Delete person by name
    - Remove person from database by name
    - Cascades to delete associated face encodings, detected faces, and photos
    """
    verify_auth_token(authorization)
    try:
        if not name or name.strip() == "":
            raise HTTPException(status_code=400, detail="Name parameter is required")
        
        success = delete_person_by_name(name.strip(), user_id=user_id)
        
        if not success:
            raise HTTPException(status_code=404, detail=f"No person found with name: {name}")
        
        return {
            "success": True,
            "message": f"Successfully deleted {name}"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error in delete_person: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== TRANSCRIPT ENDPOINT ====================

@app.post("/transcript")
def save_conversation_transcript(photo_id: int, transcript: TranscriptData):
    """
    Save transcription data from MentraLive glasses
    Used to extract names and conversation context
    """
    try:
        transcript_id = save_transcript(
            photo_id=photo_id,
            raw_text=transcript.raw_text,
            extracted_name=transcript.extracted_name,
            context=transcript.context
        )
        
        return {
            "success": True,
            "transcript_id": transcript_id,
            "message": "Transcript saved"
        }
        
    except Exception as e:
        print(f"❌ Error saving transcript: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== HEALTH CHECK ====================

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "visage-api"}