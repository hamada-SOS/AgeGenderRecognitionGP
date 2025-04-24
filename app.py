from flask import Flask, render_template, request, jsonify
from utils.predict import predict_age_gender, load_models
import cv2
import numpy as np
import os
import base64
from werkzeug.middleware.proxy_fix import ProxyFix

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024
app.wsgi_app = ProxyFix(app.wsgi_app)
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
    try:
        data_url = request.form['frame']
        encoded_data = data_url.split(',')[1]
        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Empty image buffer")
        results, annotated = predict_age_gender(image, age_model, gender_model)
        _, buffer = cv2.imencode('.jpg', annotated)
        annotated_base64 = base64.b64encode(buffer).decode('utf-8')
        return jsonify({'results': results, 'annotated': annotated_base64})
    except Exception as e:
        print("predict-frame error:", e)
        return jsonify({'error': 'Could not process frame'}), 500

@app.route('/process-video', methods=['POST'])
def process_video():
    file = request.files['video']
    filename = file.filename
    input_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(input_path)

    base_name = os.path.splitext(filename)[0]
    output_path = os.path.join(UPLOAD_FOLDER, f"{base_name}_labeled.mp4")

    cap = cv2.VideoCapture(input_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps else 0

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_skip = 5  # Process every 5th frame
    frame_count = 0
    last_annotated = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_skip == 0:
            _, last_annotated = predict_age_gender(frame, age_model, gender_model)

        # Use the last annotated frame if available
        out.write(last_annotated if last_annotated is not None else frame)
        frame_count += 1

    cap.release()
    out.release()

    return jsonify({
        'video_url': '/' + output_path.replace('\\', '/'),
        'total_frames': total_frames,
        'fps': fps,
        'duration': duration
    })

if __name__ == '__main__':
    app.run(debug=True)
