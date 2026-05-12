# import os
# import cv2
# import numpy as np
# import joblib
# from sklearn.metrics import accuracy_score, classification_report
# from tqdm import tqdm

# # 1. Paths set karein
# TEST_DIR = 'data/test/'
# CATEGORIES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# def load_test_data():
#     data = []
#     labels = []
#     print("--- Reading Images from TEST Dataset ---")
    
#     for category in CATEGORIES:
#         path = os.path.join(TEST_DIR, category)
#         if not os.path.exists(path): continue
        
#         label = CATEGORIES.index(category)
        
#         for img_name in tqdm(os.listdir(path), desc=f"Testing {category}"):
#             try:
#                 img_path = os.path.join(path, img_name)
#                 image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                
#                 if image is not None:
#                     image = cv2.resize(image, (48, 48))
#                     data.append(image.flatten())
#                     labels.append(label)
#             except:
#                 continue
                
#     return np.array(data) / 255.0, np.array(labels)

# # Step 1: Load Test Data
# X_test, y_test = load_test_data()

# # Step 2: Load the Saved Model
# print("\nLoading Trained Model...")
# model = joblib.load('models/emotion_model.pkl')

# # Step 3: Predictions lena
# print("Predicting emotions for test data...")
# y_pred = model.predict(X_test)

# # Step 4: Results dikhana
# accuracy = accuracy_score(y_test, y_pred)
# print("\n" + "="*40)
# print(f"OVERALL ACCURACY: {accuracy * 100:.2f}%")
# print("="*40)

# # Detailed Report (Kaunsa emotion kitna sahi detect hua)
# print("\nDetailed Report per Emotion:")

# print(classification_report(y_test, y_pred, target_names=CATEGORIES))

import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

# 1. Categories aur Paths
CATEGORIES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
TEST_DIR = 'data/test/' # Make sure aapka test data folder sahi hai

# 2. Naya CNN Model Load karein
print("--- Loading CNN Model (.h5) ---")
model = tf.keras.models.load_model('models/emotion_cnn_model.h5')

def load_test_data():
    X_test = []
    y_true = []
    print("--- Loading Test Images ---")
    
    for category in CATEGORIES:
        path = os.path.join(TEST_DIR, category)
        label = CATEGORIES.index(category)
        
        if not os.path.exists(path):
            continue
            
        for img_name in tqdm(os.listdir(path), desc=f"Testing {category}"):
            try:
                img_path = os.path.join(path, img_name)
                image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                image = cv2.resize(image, (48, 48))
                
                X_test.append(image)
                y_true.append(label)
            except:
                continue
                
    return np.array(X_test).reshape(-1, 48, 48, 1) / 255.0, np.array(y_true)

# Data load karein
X_test, y_true = load_test_data()

print("\n--- Predicting emotions for test data... ---")
# Model se predictions lein
predictions = model.predict(X_test)
y_pred = np.argmax(predictions, axis=1) # Sabse zyada probability wala index uthana

# 3. Final Report
print("\n" + "="*45)
print(f"OVERALL CNN ACCURACY: {accuracy_score(y_true, y_pred) * 100:.2f}%")
print("="*45 + "\n")

print("Detailed Report per Emotion:")
print(classification_report(y_true, y_pred, target_names=CATEGORIES))