import os 
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Configuration settings for the face recognition backend"""
    
    # Database
    DATABASE_URL = os.getenv("DATABASE_URL")
    
    # InsightFace Settings
    FACE_MATCH_THRESHOLD = float(os.getenv("FACE_MATCH_THRESHOLD", "0.6"))
    """
    Distance threshold for face matching (Euclidean distance)
    - Lower values = stricter matching (fewer false positives)
    - Higher values = looser matching (more false positives)
    - Default: 0.35 (good balance for normalized embeddings)
    """
    
    FACE_CONFIDENCE_MIN = float(os.getenv("FACE_CONFIDENCE_MIN", "0.9"))
    """
    Minimum confidence score for face detection
    - Range: 0.0 - 1.0
    - Default: 0.9 (90% confidence)
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