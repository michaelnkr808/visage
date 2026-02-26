"""
Manual script to add a person to the database
Usage: python add_person.py <image_path> <person_name> <user_id> [context]
"""
import sys
import os
sys.path.append(os.path.dirname(__file__))

from services.face_detection import detect_and_encode_face
from services.database import save_photo, save_detected_face, save_face_encoding, save_person_info

def add_person_to_db(image_path: str, name: str, user_id: str, context: str = None):
    """Add a person to the database from an image file"""
    
    # Read image file
    with open(image_path, 'rb') as f:
        image_data = f.read()
    
    print(f"📸 Processing image: {image_path}")
    
    # Detect face and generate encoding
    face_result = detect_and_encode_face(image_data)
    
    if not face_result:
        print("❌ No face detected in image")
        return False
    
    print(f"✅ Face detected with confidence: {face_result.get('confidence', 0.0):.4f}")
    
    # Save photo
    filename = os.path.basename(image_path)
    photo_id = save_photo(filename, image_data, user_id)
    print(f"✅ Saved photo #{photo_id}")
    
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
        conversation_context=context
    )
    print(f"✅ Saved person info #{person_info_id}: {name}")
    
    return True

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python add_person.py <image_path> <person_name> <user_id> [context]")
        print("Example: python add_person.py roommate.jpg 'John Doe' 'michaelnkr808@gmail.com' 'My roommate'")
        sys.exit(1)
    
    image_path = sys.argv[1]
    name = sys.argv[2]
    user_id = sys.argv[3]
    context = sys.argv[4] if len(sys.argv) > 4 else None
    
    if not os.path.exists(image_path):
        print(f"❌ Image file not found: {image_path}")
        sys.exit(1)
    
    success = add_person_to_db(image_path, name, user_id, context)
    
    if success:
        print(f"\n🎉 Successfully added {name} to database!")
    else:
        print("\n❌ Failed to add person to database")
        sys.exit(1)
