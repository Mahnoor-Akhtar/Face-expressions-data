import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.utils import class_weight
from tqdm import tqdm

# Setup
CATEGORIES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
TRAIN_DIR = 'data/train/'

def load_data():
    X, y = [], []
    for category in CATEGORIES:
        path = os.path.join(TRAIN_DIR, category)
        label = CATEGORIES.index(category)
        for img_name in tqdm(os.listdir(path), desc=f"Loading {category}"):
            try:
                img = cv2.imread(os.path.join(path, img_name), cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (48, 48))
                X.append(img)
                y.append(label)
            except: continue
    return np.array(X).reshape(-1, 48, 48, 1) / 255.0, np.array(y)

X_train, y_train = load_data()

# --- SOLUTION 1: Class Weights ---
# Model ko batayenge ke 'Surprise' aur 'Disgust' important hain
weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights_dict = dict(enumerate(weights))

# Building a Stronger CNN
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    layers.BatchNormalization(), # Learning ko stable karne ke liye
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5), 
    layers.Dense(7, activation='softmax')
])

# --- SOLUTION 2: Lower Learning Rate ---
# Taake model barikiyaan (like surprise patterns) miss na kare
opt = tf.keras.optimizers.Adam(learning_rate=0.0005)

model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print("\n--- Training Improved CNN (Focusing on Weak Classes) ---")
model.fit(X_train, y_train, epochs=30, batch_size=64, 
          validation_split=0.2, class_weight=class_weights_dict)

model.save('models/emotion_cnn_model.h5')
print("\nSUCCESS: Improved CNN Model Saved!")