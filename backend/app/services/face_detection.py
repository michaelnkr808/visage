import numpy as np
import io
from deepface import DeepFace
from PIL import Image
import cv2 as cv
from typing import Dict, Optional, List
from config import config

def detect_and_encode_face(image_data: bytes) -> Optional[Dict]:
    """
    Detect face and generate 128-d encoding using DeepFace + RetinaFace
    
    Args:
        image_data: Raw image bytes from MentraLive glasses or database
        
    Returns:
        Dictionary with face data:
        {
            'encoding': list,  # 128-d face embedding
            'bbox': dict,      # {x, y, w, h} bounding box
            'confidence': float,
            'cropped_face': bytes  # Optional cropped face image
        }
        Returns None if no face detected
    """
    try:
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv.imdecode(nparr, cv.IMREAD_COLOR)
        
        if img is None:
            print("❌ Failed to decode image")
            return None
        
        results = DeepFace.represent(
            img_path=img,
            model_name="Facenet",         
            detector_backend="retinaface", 
            enforce_detection=True,
            align=True                     
        )
        
        # If multiple faces detected, pick the largest one (closest person)
        if len(results) > 1:
            print(f"⚠️  Detected {len(results)} faces, using largest one")
            results = [max(results, key=lambda x: x['facial_area']['w'] * x['facial_area']['h'])]
        
        face_data = results[0]
        
        # Extract bounding box
        bbox = face_data['facial_area']

        confidence = face_data.get('face_confidence', 0.99)
        if confidence < config.FACE_CONFIDENCE_MIN:
            print(f"⚠️  Face confidence too low: {confidence}")
            return None
        
        # Crop the face from the image (optional, for storage)
        x, y, w, h = bbox['x'], bbox['y'], bbox['w'], bbox['h']
        cropped_face = img[y:y+h, x:x+w]
        
        # Convert cropped face to bytes
        _, buffer = cv.imencode('.jpg', cropped_face)
        cropped_face_bytes = buffer.tobytes()
        
        # Normalize the embedding vector
        embedding = np.array(face_data['embedding'])
        norm = np.linalg.norm(embedding)
        if norm == 0:
            print("⚠️  Warning: Zero norm embedding detected")
            normalized_embedding = embedding
        else:
            normalized_embedding = embedding / norm
        
        return {
            'encoding': normalized_embedding.tolist(),  # 128-d normalized vector
            'bbox': bbox,                        # {x, y, w, h}
            'confidence': confidence,
            'cropped_face': cropped_face_bytes
        }
        
    except ValueError as e:
        # No face detected
        print(f"❌ No face detected: {e}")
        return None
    except Exception as e:
        print(f"❌ Error in face detection: {e}")
        return None


def detect_multiple_faces(image_data: bytes) -> List[Dict]:
    """
    Detect ALL faces in an image (for group photos)
    
    Args:
        image_data: Raw image bytes
        
    Returns:
        List of face dictionaries, one per detected face
    """
    try:
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv.imdecode(nparr, cv.IMREAD_COLOR)
        
        if img is None:
            print("❌ Failed to decode image")
            return []
        
        # Detect all faces
        results = DeepFace.represent(
            img_path=img,
            model_name="Facenet",
            detector_backend="retinaface",
            enforce_detection=False,  # Don't throw error if no faces
            align=True
        )
        
        faces = []
        for face_data in results:
            bbox = face_data['facial_area']
            x, y, w, h = bbox['x'], bbox['y'], bbox['w'], bbox['h']
            
            # Crop face
            cropped_face = img[y:y+h, x:x+w]
            _, buffer = cv.imencode('.jpg', cropped_face)
            cropped_face_bytes = buffer.tobytes()
            
            # Normalize the embedding
            embedding = np.array(face_data['embedding'])
            norm = np.linalg.norm(embedding)
            if norm == 0:
                normalized_embedding = embedding
            else:
                normalized_embedding = embedding / norm
            
            faces.append({
                'encoding': normalized_embedding.tolist(),
                'bbox': bbox,
                'confidence': face_data.get('face_confidence', 0.99),
                'cropped_face': cropped_face_bytes
            })
        
        print(f"✅ Detected {len(faces)} face(s)")
        return faces
        
    except Exception as e:
        print(f"❌ Error detecting multiple faces: {e}")
        return []


def get_face_from_center(faces: List[Dict]) -> Optional[Dict]:
    """
    From multiple detected faces, return the one closest to image center
    Useful when you want the person directly in front of the glasses
    
    Args:
        faces: List of face dictionaries from detect_multiple_faces()
        
    Returns:
        Face dictionary closest to center, or None if no faces
    """
    if not faces:
        return None
    
    if len(faces) == 1:
        return faces[0]
    
    # Assume image dimensions based on first face's bbox
    # (This is approximate - in production you'd want actual image dimensions)
    image_center_x = 640  # Typical image width / 2
    image_center_y = 480  # Typical image height / 2
    
    def distance_from_center(face):
        bbox = face['bbox']
        face_center_x = bbox['x'] + bbox['w'] / 2
        face_center_y = bbox['y'] + bbox['h'] / 2
        
        return np.sqrt(
            (face_center_x - image_center_x) ** 2 + 
            (face_center_y - image_center_y) ** 2
        )
    
    # Return face closest to center
    return min(faces, key=distance_from_center)


# ==================== TESTING FUNCTIONS ====================

def test_detection_on_photo_id(photo_id: int):
    """
    Test face detection on a photo from the database
    For debugging purposes
    """
    from services.database import get_photo_by_id
    
    photo = get_photo_by_id(photo_id)
    if not photo:
        print(f"❌ Photo #{photo_id} not found")
        return
    
    print(f"🔍 Testing face detection on photo #{photo_id}")
    result = detect_and_encode_face(photo.image_data)
    
    if result:
        print(f"✅ Face detected!")
        print(f"   Bounding box: {result['bbox']}")
        print(f"   Confidence: {result['confidence']:.2f}")
        print(f"   Encoding length: {len(result['encoding'])}")
    else:
        print(f"❌ No face detected")


def test_detection_on_latest_photo():
    """
    Test face detection on the most recent photo in database
    """
    from services.database import get_most_recent_photo
    
    photo = get_most_recent_photo()
    if not photo:
        print("❌ No photos in database")
        return
    
    print(f"🔍 Testing face detection on most recent photo (#{photo.id})")
    result = detect_and_encode_face(photo.image_data)
    
    if result:
        print(f"✅ Face detected!")
        print(f"   Bounding box: {result['bbox']}")
        print(f"   Confidence: {result['confidence']:.2f}")
        print(f"   Encoding length: {len(result['encoding'])}")
    else:
        print(f"❌ No face detected")


# For command-line testing
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test face detection')
    parser.add_argument('--photo_id', type=int, help='ID of photo to test')
    args = parser.parse_args()
    
    if args.photo_id:
        test_detection_on_photo_id(args.photo_id)
    else:
        test_detection_on_latest_photo()