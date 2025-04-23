import cv2
from utils.predict import load_models, predict_age_gender

age_model, gender_model = load_models()

img = cv2.imread('static/uploads/images.jpeg')  # Use a real face image
labels, annotated = predict_age_gender(img, age_model, gender_model)
print("Predicted:", labels)
cv2.imwrite("annotated_result.jpg", annotated)
