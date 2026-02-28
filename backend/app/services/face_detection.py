import numpy as np
import cv2 as cv
from insightface.app import FaceAnalysis
from typing import Dict, Optional, List
from config import config

# Initialize InsightFace model (singleton)
# buffalo_l: best quality model with strongest angle/distance robustness
# det_size=(1024, 1024): larger input preserves more detail for small/distant faces
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(1024, 1024))

def _preprocess_image(img: np.ndarray) -> np.ndarray:
    """
    Upscale small images before detection.
    With size='small' from the Mentra Live, images can be quite small.
    Upscaling helps InsightFace detect distant/small faces.
    """
    h, w = img.shape[:2]
    if w < 640 or h < 640:
        scale = max(640 / w, 640 / h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv.resize(img, (new_w, new_h), interpolation=cv.INTER_CUBIC)
        print(f"📐 Upscaled image from {w}x{h} to {new_w}x{new_h}")
    return img

def detect_and_encode_face(image_data: bytes) -> Optional[Dict]:
    """
    Detect face and generate 512-d encoding using InsightFace

    Args:
        image_data: Raw image bytes from MentraLive glasses or database

    Returns:
        Dictionary with face data:
        {
            'encoding': list,  # 512-d face embedding
            'bbox': dict,      # {x, y, w, h} bounding box
            'confidence': float,
            'cropped_face': bytes  # Cropped face image
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

        img = _preprocess_image(img)

        # Detect faces
        faces = app.get(img)

        if not faces or len(faces) == 0:
            print("❌ No faces detected")
            return None

        # If multiple faces, pick the largest one (closest person)
        if len(faces) > 1:
            print(f"⚠️  Detected {len(faces)} faces, using largest one")
            face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        else:
            face = faces[0]

        # Get detection confidence and apply minimum threshold
        confidence = float(face.det_score)
        print(f"ℹ️  Face detected with confidence: {confidence:.4f}")

        if confidence < config.FACE_CONFIDENCE_MIN:
            print(f"⚠️  Confidence {confidence:.4f} below minimum {config.FACE_CONFIDENCE_MIN} — rejecting")
            return None

        # Extract bbox [x1, y1, x2, y2] -> convert to [x, y, w, h]
        x1, y1, x2, y2 = face.bbox.astype(int)
        x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)

        # Validate bbox
        img_height, img_width = img.shape[:2]
        face_area = w * h
        img_area = img_width * img_height
        coverage = face_area / img_area

        print(f"📏 Detected face bbox: {w}x{h} at ({x}, {y}) | Image: {img_width}x{img_height} | Coverage: {coverage*100:.1f}%")

        # Reject if bbox at origin (0,0) AND covers >98% - detector failed
        if x <= 5 and y <= 5 and coverage > 0.98:
            print(f"⚠️  Full-image detection at origin - likely false positive, rejecting")
            return None

        if w < 30 or h < 30:
            print(f"⚠️  Detected face too small ({w}x{h}) - likely false positive")
            return None

        # Crop the face from the image
        cropped_face = img[y:y+h, x:x+w]

        # Convert cropped face to bytes
        _, buffer = cv.imencode('.jpg', cropped_face)
        cropped_face_bytes = buffer.tobytes()

        # Get the 512-d embedding (already normalized by InsightFace)
        embedding = face.normed_embedding

        bbox_dict = {'x': x, 'y': y, 'w': w, 'h': h}

        return {
            'encoding': embedding.tolist(),  # 512-d normalized vector
            'bbox': bbox_dict,
            'confidence': confidence,
            'cropped_face': cropped_face_bytes
        }

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

        img = _preprocess_image(img)

        # Detect all faces
        faces = app.get(img)

        if not faces:
            print("❌ No faces detected")
            return []

        result_faces = []
        for face in faces:
            confidence = float(face.det_score)
            if confidence < config.FACE_CONFIDENCE_MIN:
                print(f"⚠️  Skipping face with confidence {confidence:.4f} below minimum")
                continue

            # Extract bbox
            x1, y1, x2, y2 = face.bbox.astype(int)
            x, y, w, h = x1, y1, x2 - x1, y2 - y1

            # Crop face
            cropped_face = img[y:y+h, x:x+w]
            _, buffer = cv.imencode('.jpg', cropped_face)
            cropped_face_bytes = buffer.tobytes()

            # Get embedding
            embedding = face.normed_embedding

            bbox_dict = {'x': x, 'y': y, 'w': w, 'h': h}

            result_faces.append({
                'encoding': embedding.tolist(),
                'bbox': bbox_dict,
                'confidence': confidence,
                'cropped_face': cropped_face_bytes
            })

        print(f"✅ Detected {len(result_faces)} face(s)")
        return result_faces

    except Exception as e:
        print(f"❌ Error detecting multiple faces: {e}")
        return []
