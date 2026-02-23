# Add this to test_distance.py
from services.face_detection import detect_and_encode_face
import numpy as np

# Test first photo
with open('../../me1.JPEG', 'rb') as f:
    result1 = detect_and_encode_face(f.read())

with open('../../me2.jpg', 'rb') as f:
    result2 = detect_and_encode_face(f.read())

if result1 and result2:
    enc1 = np.array(result1['encoding'])
    enc2 = np.array(result2['encoding'])
    
    # Check if normalized
    norm1 = np.linalg.norm(enc1)
    norm2 = np.linalg.norm(enc2)
    
    print(f"\n🔍 Debugging:")
    print(f"Encoding 1 norm (should be ~1.0): {norm1:.4f}")
    print(f"Encoding 2 norm (should be ~1.0): {norm2:.4f}")
    
    # Try different distance metrics
    l2_dist = np.linalg.norm(enc1 - enc2)
    cosine_sim = np.dot(enc1, enc2) / (norm1 * norm2)
    cosine_dist = 1 - cosine_sim
    
    print(f"\n📏 Distance Metrics:")
    print(f"L2 Distance: {l2_dist:.4f}")
    print(f"Cosine Distance: {cosine_dist:.4f}")
    print(f"Cosine Similarity: {cosine_sim:.4f}")
    
    # Normalize and recalculate
    enc1_norm = enc1 / norm1
    enc2_norm = enc2 / norm2
    l2_dist_normalized = np.linalg.norm(enc1_norm - enc2_norm)
    
    print(f"\n✅ Normalized L2 Distance: {l2_dist_normalized:.4f}")
    print(f"   (This should be 0.3-0.6 for same person)")