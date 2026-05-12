import os
import cv2
import numpy as np
import tensorflow as tf
import base64
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # React frontend se connection ke liye zaroori hai

# 1. CNN Model aur Face Cascade Load karein
MODEL_PATH = 'models/emotion_cnn_model.h5'

if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
    print("--- CNN Model Loaded Successfully ---")
else:
    print("ERROR: Model file not found! Please train the CNN first.")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Emotions ki list (Sequence wahi rakhni hai jo training mein thi)
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Frontend se image data lena
        data = request.json.get('image')
        if not data:
            return jsonify({"emotion": "No Data Received"})

        # Base64 string ko image mein convert karna
        header, encoded = data.split(",", 1)
        img_data = base64.b64decode(encoded)
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Grayscale conversion (CNN grayscale images par train hua hai)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Face Detection
        faces = face_cascade.detectMultiScale(gray_frame, 1.1, 6, minSize=(48, 48))
        
        if len(faces) == 0:
            return jsonify({"emotion": "No Face Detected"})

        # Sirf pehle (primary) face ko process karein
        (x, y, w, h) = faces[0]
        roi_gray = gray_frame[y:y+h, x:x+w]
        roi_resized = cv2.resize(roi_gray, (48, 48))
        
        # Preprocessing: Shape (1, 48, 48, 1) aur Normalize (/255)
        roi_final = roi_resized.reshape(1, 48, 48, 1).astype('float32') / 255.0
        
        # Model Prediction
        prediction = model.predict(roi_final, verbose=0) # verbose=0 taake terminal mein extra logs na aayein
        max_index = int(np.argmax(prediction[0]))
        predicted_emotion = EMOTIONS[max_index]

        return jsonify({"emotion": predicted_emotion})

    except Exception as e:
        print(f"Backend Error: {e}")
        return jsonify({"emotion": "Server Error"})

if __name__ == '__main__':
    # Debug mode on rakhein taake koi bhi error terminal mein foran nazar aaye
    app.run(debug=True, port=5000)