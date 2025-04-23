import cv2
import os

# Load Haar cascade once (efficient reuse)
face_cascade = cv2.CascadeClassifier('face_haar/haarcascade_frontalface_alt.xml')

def detect_faces(img, face_padding_ratio=0.10):
    """
    Detects faces in an image using Haar cascades and returns bounding boxes.
    Each box is in the format: (x1, y1, x2, y2)
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detections = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    height, width = img.shape[:2]
    face_boxes = []

    for (x, y, w, h) in detections:
        pad_w = int(w * face_padding_ratio)
        pad_h = int(h * face_padding_ratio)
        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(x + w + pad_w, width - 1)
        y2 = min(y + h + pad_h, height - 1)
        face_boxes.append((x1, y1, x2, y2))

    return face_boxes
