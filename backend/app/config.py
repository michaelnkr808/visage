import os 
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Configuration settings for the face recognition backend"""
    
    # Database
    DATABASE_URL = os.getenv("DATABASE_URL")
    
    # InsightFace Settings
    FACE_MATCH_THRESHOLD = float(os.getenv("FACE_MATCH_THRESHOLD", "0.4"))
    """
    Distance threshold for face matching (L2 distance on normalized embeddings)
    - buffalo_l normalized embeddings: typical range 0.35-0.45
    - Lower values = stricter matching (fewer false positives)
    - Higher values = looser matching (more false positives)
    """
    
    FACE_CONFIDENCE_MIN = float(os.getenv("FACE_CONFIDENCE_MIN", "0.75"))
    """
    Minimum confidence score for face detection
    - Range: 0.0 - 1.0
    - Mentra Live at size='small' typically produces 0.74-0.84
    """
    
    # API Settings
    MENTRAOS_API_KEY = os.getenv("MENTRAOS_API_KEY")
    BACKEND_PORT = int(os.getenv("BACKEND_PORT", "8000"))
    BACKEND_AUTH_TOKEN = os.getenv("BACKEND_AUTH_TOKEN")
    
    if not BACKEND_AUTH_TOKEN:
        raise ValueError("BACKEND_AUTH_TOKEN must be set in environment variables")
    
    # Image Processing
    MAX_IMAGE_SIZE_MB = int(os.getenv("MAX_IMAGE_SIZE_MB", "10"))
    ALLOWED_IMAGE_FORMATS = ["jpg", "jpeg", "png", "webp"]
    
    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Create a singleton instance
config = Config()