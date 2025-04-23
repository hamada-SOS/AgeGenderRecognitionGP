import cv2
import numpy as np

def preprocess_face(face, target_size=(64, 64)):
    face = cv2.resize(face, target_size)
    face = face.astype('float32') / 255.0  # Normalize to [0, 1]
    return np.expand_dims(face, axis=0)
