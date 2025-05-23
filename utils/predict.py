import os
import cv2
import numpy as np
import hashlib
from models.Model_net import Age_net, Gender_net
from facenet_pytorch import MTCNN
import torch


# --- Config ---
face_padding_ratio = 0.10
face_size = 64
stage_num = [3, 3, 3]
lambda_local = 1
lambda_d = 1
DEBUG = True  # <<< Set to False to disable logging

# --- Load Haar Cascade ---
face_cascade = cv2.CascadeClassifier('face_haar/lbpcascade_frontalface_improved.xml')
# -- Load MTCNN
mtcnn = MTCNN(keep_all=True, device='cuda' if torch.cuda.is_available() else 'cpu')

def debug_print(msg):
    if DEBUG:
        print(msg)


# --- Load Models ---
def load_models():
    debug_print("Loading models...")
    age_model = Age_net(face_size, stage_num, lambda_local, lambda_d)()
    age_model.load_weights('models/age_model1.h5')
    print("Age model summary:")
    age_model.summary()

    gender_model = Gender_net(face_size, stage_num, lambda_local, lambda_d)()
    gender_model.load_weights('models/gender_model1.h5')
    print("Gender model summary:")
    gender_model.summary()

    debug_print("Models loaded successfully.")
    return age_model, gender_model


# # --- Face Detection 1 ---
# def detect_faces(img, face_padding_ratio=face_padding_ratio):
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     detections = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

#     height, width = img.shape[:2]
#     face_boxes = []

#     debug_print(f"Detected {len(detections)} face(s).")

#     for (x, y, w, h) in detections:
#         pad_w = int(w * face_padding_ratio)
#         pad_h = int(h * face_padding_ratio)
#         x1 = max(0, x - pad_w)
#         y1 = max(0, y - pad_h)
#         x2 = min(x + w + pad_w, width - 1)
#         y2 = min(y + h + pad_h, height - 1)
#         face_boxes.append((x1, y1, x2, y2))

#         debug_print(f"Face box: ({x1}, {y1}), ({x2}, {y2})")

#     return face_boxes


# # --- Face Detection 2 ---

def detect_faces(img, face_padding_ratio=face_padding_ratio):
    boxes, _ = mtcnn.detect(img)

    face_boxes = []
    height, width = img.shape[:2]

    if boxes is None:
        debug_print("No faces detected by MTCNN.")
        return face_boxes

    debug_print(f"Detected {len(boxes)} face(s) with MTCNN.")

    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        pad_w = int((x2 - x1) * face_padding_ratio)
        pad_h = int((y2 - y1) * face_padding_ratio)
        x1 = max(0, x1 - pad_w)
        y1 = max(0, y1 - pad_h)
        x2 = min(width - 1, x2 + pad_w)
        y2 = min(height - 1, y2 + pad_h)
        face_boxes.append((x1, y1, x2, y2))

    return face_boxes





# --- Face Preprocessing ---
def preprocess_faces(faces_bgr):
    blob = np.empty((len(faces_bgr), face_size, face_size, 3), dtype='float32')
    for i, face in enumerate(faces_bgr):
        resized = cv2.resize(face, (face_size, face_size))
        normalized = cv2.normalize(resized, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        blob[i] = normalized
        checksum = hashlib.md5(blob[i].tobytes()).hexdigest()
        debug_print(f"[Face {i}] Shape: {resized.shape}, Checksum: {checksum}")
    return blob


# --- Main Prediction ---
def predict_age_gender(image_bgr, age_model, gender_model):
    face_boxes = detect_faces(image_bgr)
    labels = []

    if not face_boxes:
        debug_print("No faces detected.")
        return labels, image_bgr

    faces_bgr = [image_bgr[y1:y2, x1:x2] for (x1, y1, x2, y2) in face_boxes]
    valid_faces = [face for face in faces_bgr if face.size > 0]

    debug_print(f"Valid faces for prediction: {len(valid_faces)}")

    if not valid_faces:
        debug_print("All detected face crops were invalid.")
        return labels, image_bgr

    face_inputs = preprocess_faces(valid_faces)

    genders = gender_model.predict(face_inputs)
    ages = age_model.predict(face_inputs)

    for i, ((x1, y1, x2, y2), gender, age) in enumerate(zip(face_boxes, genders, ages)):
        gender_label = "Male" if gender[0] >= 0.5 else "Female"
        age_label = int(round(age[0]))
        label = f"{gender_label}, {age_label}"
        labels.append(label)

        debug_print(f"[Face {i}] Gender: {gender[0]:.2f}, Age: {age[0]:.2f}, Label: {label}")

        # Draw bounding box and label on image
        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image_bgr, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 2)

    # Optional: frame hash for debug
    frame_hash = hashlib.md5(image_bgr.tobytes()).hexdigest()
    debug_print(f"Frame checksum: {frame_hash}")

    return labels, image_bgr
