from flask import Flask, render_template, request, jsonify
from utils.predict import predict_age_gender, load_models
import cv2
import numpy as np
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

age_model, gender_model = load_models()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)
    image = cv2.imread(path)
    results, annotated = predict_age_gender(image, age_model, gender_model)
    out_path = path.replace(".", "_out.")
    cv2.imwrite(out_path, annotated)
    return jsonify({'results': results, 'annotated_image': out_path})

@app.route('/predict-frame', methods=['POST'])
def predict_frame():
    # Receives base64 webcam image
    import base64
    data_url = request.form['frame']
    encoded_data = data_url.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    results, annotated = predict_age_gender(image, age_model, gender_model)

    _, buffer = cv2.imencode('.jpg', annotated)
    annotated_base64 = base64.b64encode(buffer).decode('utf-8')
    return jsonify({'results': results, 'annotated': annotated_base64})

if __name__ == '__main__':
    app.run(debug=True)
