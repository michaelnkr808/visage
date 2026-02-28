from dotenv import load_dotenv
from datetime import datetime, timezone
from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker
from models.face_scan import Base, Photo, Transcript, DetectedFace, FaceEncoding, PersonInfo
from config import config

load_dotenv()

engine = create_engine(config.DATABASE_URL)

SessionLocal = sessionmaker(bind=engine)

#create tables
def init_db():
    """Initialize database tables"""
    Base.metadata.create_all(engine)

def get_db():
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Photo helper functions ----------------------------------------

def save_photo(filename: str, image_data: bytes, user_id: str) -> int:
    with SessionLocal() as session:
        try:
            photo = Photo(filename=filename, image_data=image_data, user_id=user_id)
            session.add(photo)
            session.commit()
            return photo.id
        except Exception:
            session.rollback()
            raise

def get_photo_by_id(photo_id: int) -> Photo | None:
    with SessionLocal() as session:
        return session.query(Photo).filter(Photo.id == photo_id).first()

def get_most_recent_photo() -> Photo | None:
    with SessionLocal() as session:
        return session.query(Photo).order_by(Photo.created_at.desc()).first()

# Face helper functions ------------------------------------------

def save_detected_face(photo_id: int, x: int, y:int, width: int, height: int,
                       face_image_data: bytes = None, confidence: float = None) -> int:
    """Save a detected face to the database"""
    with SessionLocal() as session:
        try:
            face = DetectedFace(
                photo_id=photo_id,
                x=x, y=y, width=width, height=height,
                face_image_data=face_image_data,
                confidence=confidence
            )
            session.add(face)
            session.commit()
            session.refresh(face)
            return face.id
        except Exception as e:
            session.rollback()
            raise e

def save_face_encoding(face_id: int, encoding: list, model_name: str = "InsightFace") -> int:
    """
    Save a face encoding (512-d vector).
    Note: Encoding should already be normalized by detect_and_encode_face().
    person_info_id is set separately via link_encoding_to_person() after PersonInfo is created.
    """
    with SessionLocal() as session:
        try:
            face_encoding = FaceEncoding(
                face_id=face_id,
                encoding=encoding,
                model_name=model_name
            )
            session.add(face_encoding)
            session.commit()
            return face_encoding.id
        except Exception as e:
            session.rollback()
            raise e

def link_encoding_to_person(encoding_id: int, person_info_id: int) -> None:
    """
    Link a FaceEncoding to a PersonInfo after both have been created.
    This is what enables multiple encodings per person — any subsequent
    enrollments can also be linked to the same person_info_id.
    """
    with SessionLocal() as session:
        try:
            encoding = session.query(FaceEncoding).filter(FaceEncoding.id == encoding_id).first()
            if encoding:
                encoding.person_info_id = person_info_id
                session.commit()
        except Exception as e:
            session.rollback()
            raise e

def find_matching_face(query_encoding: list, user_id: str, threshold: float = None) -> tuple[PersonInfo | None, float | None]:
    """
    Find the best matching person using pgvector similarity search.
    Groups by person_info_id and takes the minimum distance across all
    stored encodings for each person — so a person enrolled from multiple
    angles is matched against all their embeddings.

    Returns (PersonInfo, distance) if a match is found below threshold, else (None, distance).
    """
    if threshold is None:
        threshold = config.FACE_MATCH_THRESHOLD

    with SessionLocal() as session:
        # Find minimum L2 distance per person, filtered by user_id
        result = session.query(
            FaceEncoding.person_info_id,
            func.min(FaceEncoding.encoding.l2_distance(query_encoding)).label('distance')
        ).join(DetectedFace, FaceEncoding.face_id == DetectedFace.id
        ).join(Photo, DetectedFace.photo_id == Photo.id
        ).filter(
            Photo.user_id == user_id,
            FaceEncoding.person_info_id.isnot(None)
        ).group_by(FaceEncoding.person_info_id
        ).order_by('distance').first()

        if not result:
            print(f"ℹ️  No faces found in database for user {user_id}")
            return None, None

        person_info_id, distance = result.person_info_id, result.distance
        print(f"🔍 Closest match distance: {distance:.4f} (threshold: {threshold})")

        if distance >= threshold:
            print(f"❌ No match (distance {distance:.4f} >= {threshold})")
            return None, distance

        print(f"✅ Match found (distance {distance:.4f} < {threshold})")
        person_info = session.query(PersonInfo).filter(PersonInfo.id == person_info_id).first()
        return person_info, distance

# Person info helper functions ---------------------------------------

def save_person_info(face_id: int, user_id: str, name: str = None, conversation_context: str = None) -> int:
    """Save person information"""
    with SessionLocal() as session:
        try:
            person_info = PersonInfo(
                face_id=face_id,
                user_id=user_id,
                name=name,
                conversation_context=conversation_context
            )
            session.add(person_info)
            session.commit()
            return person_info.id
        except Exception as e:
            session.rollback()
            raise e

def get_person_info_by_face_id(face_id:int) -> PersonInfo | None:
    """Get person info for a face"""
    with SessionLocal() as session:
        return session.query(PersonInfo).filter(PersonInfo.face_id == face_id).first()

def get_person_info_by_name(name: str, user_id: str) -> PersonInfo | None:
    """Get person info by name (case-insensitive partial match)"""
    with SessionLocal() as session:
        return session.query(PersonInfo).filter(
            PersonInfo.name.ilike(f"%{name}%"),
            PersonInfo.user_id == user_id
        ).first()

def update_person_last_seen(person_info_id: int) -> None:
    """Update last seen timestamp and increment times_met"""
    with SessionLocal() as session:
        try:
            person_info = session.query(PersonInfo).filter(PersonInfo.id == person_info_id).first()
            if person_info:
                person_info.last_seen_at = datetime.now(timezone.utc)
                person_info.times_met += 1
                session.commit()
        except Exception as e:
            session.rollback()
            raise e

def delete_person_by_name(name: str, user_id: str) -> bool:
    """Delete person and all associated data by name"""
    with SessionLocal() as session:
        try:
            person_info = session.query(PersonInfo).filter(
                PersonInfo.name.ilike(f"%{name}%"),
                PersonInfo.user_id == user_id
            ).first()

            if not person_info:
                return False

            session.delete(person_info)
            session.commit()
            return True
        except Exception as e:
            session.rollback()
            raise e

# Transcript helper functions ----------------------------------------

def save_transcript(photo_id: int, raw_text: str = None, extracted_name: str = None, context: str = None) -> int:
    """Save transcript data"""
    with SessionLocal() as session:
        try:
            transcript = Transcript(
                photo_id=photo_id,
                raw_text=raw_text,
                extracted_name=extracted_name,
                context=context
            )
            session.add(transcript)
            session.commit()
            return transcript.id
        except Exception as e:
            session.rollback()
            raise e
