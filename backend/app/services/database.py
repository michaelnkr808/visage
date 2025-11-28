from models.face_scan import Photo, SearchResult, engine
from sqlalchemy.orm import sessionmaker

SessionLocal = sessionmaker(bind = engine)

def save_photo(filename: str, image_data: bytes):
    with SessionLocal() as session:
        try:
            photo = Photo(filename=filename, image_data=image_data)
            session.add(photo)
            session.commit()
            return photo.id
        except:
            session.rollback()
            raise

def save_result(photo_id: int, name: str, platform: str, profile_url: str, confidence: int):
    with SessionLocal() as session:
        try:
            result = SearchResult(
                photo_id = photo_id,
                name = name,
                platform = platform,
                profile_url = profile_url,
                confidence = confidence,
            )
            session.add(result)
            session.commit()
        except:
            session.rollback()
            raise

def get_photo_by_id(photo_id: int):
    with SessionLocal() as session:
        return session.query(Photo).filter(Photo.id == photo_id).first()

def get_results_for_photo(photo_id:int):
    with SessionLocal() as session:
        return session.query(SearchResult).filter(SearchResult.photo_id == photo_id).all()