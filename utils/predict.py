# import cv2
# import numpy as np
# from .detect import detect_faces

# def load_models():
#     from age_gender_ssrnet.SSRNET_model import SSR_net, SSR_net_general
#     age_model = SSR_net(64, [3, 3, 3], 1, 1)()
#     age_model.load_weights('age_gender_ssrnet/ssrnet_age_3_3_3_64_1.0_1.0.h5')

#     gender_model = SSR_net_general(64, [3, 3, 3], 1, 1)()
#     gender_model.load_weights('age_gender_ssrnet/ssrnet_gender_3_3_3_64_1.0_1.0.h5')

#     return age_model, gender_model

# def predict_age_gender(img, age_model, gender_model):
#     face_boxes = detect_faces(img)
#     results = []

#     for (x1, y1, x2, y2) in face_boxes:
#         face = img[y1:y2, x1:x2]
#         face_resized = cv2.resize(face, (64, 64))
#         face_input = np.expand_dims(face_resized.astype('float32'), axis=0)

#         # Normalize
#         face_input /= 255.0

#         gender = gender_model.predict(face_input)[0][0]
#         age = age_model.predict(face_input)[0][0]

#         label = f"{'Male' if gender > 0.5 else 'Female'}, {int(age)}"
#         results.append(label)

#         cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
#         cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

#     return results, img
import cv2
import numpy as np
from .detect import detect_faces

def load_models():
    from age_gender_ssrnet.SSRNET_model import SSR_net, SSR_net_general
    
    # Load age model
    age_model = SSR_net(64, [3, 3, 3], 1, 1)()
    age_model.load_weights('age_gender_ssrnet/ssrnet_age_3_3_3_64_1.0_1.0.h5')

    # Load gender model
    gender_model = SSR_net_general(64, [3, 3, 3], 1, 1)()
    gender_model.load_weights('age_gender_ssrnet/ssrnet_gender_3_3_3_64_1.0_1.0.h5')

    return age_model, gender_model


def preprocess_face(face_bgr, target_size=(64, 64)):
    """
    Resize and normalize face image for SSR-Net models.
    """
    face_resized = cv2.resize(face_bgr, target_size)
    face_resized = face_resized.astype('float32') / 255.0  # Normalize to [0, 1]
    face_input = np.expand_dims(face_resized, axis=0)      # Add batch dimension
    return face_input


def predict_age_gender(img_bgr, age_model, gender_model):
    """
    Detects faces, runs age & gender prediction, and returns:
    - list of labels
    - annotated image
    """
    face_boxes = detect_faces(img_bgr)
    labels = []

    for i, (x1, y1, x2, y2) in enumerate(face_boxes):
        face_bgr = img_bgr[y1:y2, x1:x2]
        
        # Validate face area
        if face_bgr.size == 0:
            continue

        face_input = preprocess_face(face_bgr)

        # Predictions
        gender_pred = gender_model.predict(face_input)[0][0]
        age_pred = age_model.predict(face_input)[0][0]

        # Debug print (optional)
        print(f"[Face {i}] Gender raw: {gender_pred:.2f}, Age raw: {age_pred:.2f}")

        gender_label = "Male" if gender_pred >= 0.5 else "Female"
        age_label = int(round(age_pred))
        label = f"{gender_label}, {age_label}"
        labels.append(label)

        # Draw bounding box and label on the image
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_bgr, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 2)

    return labels, img_bgr
