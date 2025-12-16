import __future__ as print_function
import argparse
import numpy as np
import io
from deepface import DeepFace
import cv2 as cv
from PIL import Image
from models.face_scan import Photo
from services.database import get_photo_by_id, get_most_recent_photo

parser = argparse.ArgumentParser(description='Code for Cascade Classifier tutorial.')
parser.add_argument('--face_cascade', help='Path to face cascade.', default='data/haarcascades/haarcascade_frontalface_alt.xml')
parser.add_argument('--eyes_cascade', help='Path to eyes cascade.', default='data/haarcascades/haarcascade_eye_tree_eyeglasses.xml')
parser.add_argument('--photo_id', help='ID of photo to process.', type=int, default=None)
args = parser.parse_args()

face_cascade_name = args.face_cascade
eyes_cascade_name = args.eyes_cascade
face_cascade = cv.CascadeClassifier()
eyes_cascade = cv.CascadeClassifier()

#-- 1. Load the cascades
if not face_cascade.load(cv.samples.findFile(face_cascade_name)):
    print('--(!)Error loading face cascade')
    exit(0)
if not eyes_cascade.load(cv.samples.findFile(eyes_cascade_name)):
    print('--(!)Error loading eyes cascade')
    exit(0)

if args.photo_id is not None:
    photo = get_photo_by_id(args.photo_id)
else:
    photo = get_most_recent_photo()

image_bytes = photo.image_data

buf = np.frombuffer(image_bytes, dtype=np.uint8)
photo_array = cv.imdecode(buf, cv.IMREAD_GRAYSCALE)

if photo_array is None:
    print(f"Error: could not load photo from {Photo}")
else:
    detectAndDisplay(photo_array)



