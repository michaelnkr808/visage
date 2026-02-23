#!/usr/bin/env python3
"""
Test script for Visage face recognition workflow
Tests workflow1, workflow2, and workflow3 (query by name)
"""

import requests
import base64
import sys
from pathlib import Path

# Configuration
BACKEND_URL = "http://localhost:8000"
API_BASE = f"{BACKEND_URL}/api"

def test_first_meeting(image_path: str, name: str, context: str = ""):
    """Test saving a new person"""
    print(f"\n{'='*60}")
    print(f"TEST 1: First Meeting - Saving '{name}'")
    print(f"{'='*60}")
    
    # Read and encode image
    with open(image_path, 'rb') as f:
        image_bytes = f.read()
    
    image_base64 = base64.b64encode(image_bytes).decode('utf-8')
    
    # Make request
    response = requests.post(
        f"{API_BASE}/workflow1/first-meeting",
        data={
            "image_data": image_base64,
            "name": name,
            "conversation_context": context
        }
    )
    
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Success!")
        print(f"   Photo ID: {data['data']['photo_id']}")
        print(f"   Face ID: {data['data']['face_id']}")
        print(f"   Person Info ID: {data['data']['person_info_id']}")
        print(f"   Name: {data['data']['name']}")
        return data['data']
    else:
        print(f"❌ Failed: {response.text}")
        return None

def test_recognize(image_path: str):
    """Test recognizing an existing person"""
    print(f"\n{'='*60}")
    print(f"TEST 2: Recognition - Identifying person from photo")
    print(f"{'='*60}")
    
    # Read and encode image
    with open(image_path, 'rb') as f:
        image_bytes = f.read()
    
    image_base64 = base64.b64encode(image_bytes).decode('utf-8')
    
    # Make request
    response = requests.post(
        f"{API_BASE}/workflow2/recognize",
        data={
            "image_data": image_base64
        }
    )
    
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        
        if data['recognized']:
            print(f"✅ Person Recognized!")
            print(f"   Name: {data['person']['name']}")
            print(f"   Context: {data['person']['conversation_context']}")
            print(f"   Distance: {data['distance']:.4f}")
            print(f"   Times Met: {data['person']['times_met']}")
            print(f"   First Met: {data['person']['first_met_at']}")
            print(f"   Last Seen: {data['person']['last_seen_at']}")
        else:
            print(f"⚠️  Person not recognized")
            print(f"   Message: {data['message']}")
            if data.get('distance'):
                print(f"   Distance: {data['distance']:.4f}")
        
        return data
    else:
        print(f"❌ Failed: {response.text}")
        return None

def test_query_by_name(name: str):
    """Test querying a person by name (Workflow 3)"""
    print(f"\n{'='*60}")
    print(f"TEST 3: Query by Name - Searching for '{name}'")
    print(f"{'='*60}")
    
    # Make request
    response = requests.get(
        f"{API_BASE}/people/search",
        params={"name": name}
    )
    
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Person Found!")
        print(f"   Name: {data['name']}")
        print(f"   Context: {data.get('conversation_context', 'N/A')}")
        print(f"   Times Met: {data.get('times_met', 0)}")
        if data.get('first_met_at'):
            print(f"   First Met: {data['first_met_at']}")
        if data.get('last_seen_at'):
            print(f"   Last Seen: {data['last_seen_at']}")
        return data
    elif response.status_code == 404:
        print(f"⚠️  Person not found")
        print(f"   Message: {response.json().get('detail', 'Unknown error')}")
        return None
    else:
        print(f"❌ Failed: {response.text}")
        return None

def main():
    """Run the test workflow"""
    print("\n" + "="*60)
    print("VISAGE FACE RECOGNITION TEST")
    print("="*60)
    
    # Check if image path provided
    if len(sys.argv) < 2:
        print("\n❌ Usage: python test_face_recognition.py <path_to_image> [name] [context]")
        print("\nExample:")
        print("  python test_face_recognition.py photo.jpg 'John Doe' 'Works at Apple'")
        sys.exit(1)
    
    image_path = sys.argv[1]
    name = sys.argv[2] if len(sys.argv) > 2 else "Test Person"
    context = sys.argv[3] if len(sys.argv) > 3 else "Test context"
    
    # Verify image exists
    if not Path(image_path).exists():
        print(f"\n❌ Error: Image file not found: {image_path}")
        sys.exit(1)
    
    print(f"\nUsing image: {image_path}")
    print(f"Name: {name}")
    print(f"Context: {context}")
    
    # Test 1: Save person
    result1 = test_first_meeting(image_path, name, context)
    
    if not result1:
        print("\n❌ First meeting test failed. Stopping.")
        sys.exit(1)
    
    # Test 2: Recognize same person (Workflow 2)
    print("\n⏳ Now testing recognition with the same image...")
    result2 = test_recognize(image_path)
    
    # Test 3: Query by name (Workflow 3)
    print("\n⏳ Now testing query by name...")
    result3 = test_query_by_name(name)
    
    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    
    workflow2_passed = result2 and result2.get('recognized')
    workflow3_passed = result3 is not None
    
    if result1 and workflow2_passed and workflow3_passed:
        print("✅ All tests passed!")
        print(f"   ✓ Workflow 1: Person saved successfully")
        print(f"   ✓ Workflow 2: Person recognized from photo")
        print(f"   ✓ Workflow 3: Person found by name query")
    elif result1:
        print("⚠️  Tests partially passed")
        print(f"   ✓ Workflow 1: Person saved successfully")
        if not workflow2_passed:
            print(f"   ✗ Workflow 2: Recognition failed (might be threshold issue)")
            print(f"      💡 Try adjusting FACE_MATCH_THRESHOLD in backend/.env")
        else:
            print(f"   ✓ Workflow 2: Person recognized from photo")
        if not workflow3_passed:
            print(f"   ✗ Workflow 3: Query by name failed")
        else:
            print(f"   ✓ Workflow 3: Person found by name query")
    else:
        print("❌ Tests failed")
    
    print()

if __name__ == "__main__":
    main()