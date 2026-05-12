import os
import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
import joblib
from tqdm import tqdm

# 1. Paths set karein
TRAIN_DIR = 'data/train/'
CATEGORIES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

def load_data():
    data = []
    labels = []
    print("--- Reading ALL Images for Better Training ---")
    
    for category in CATEGORIES:
        path = os.path.join(TRAIN_DIR, category)
        label = CATEGORIES.index(category)
        
        for img_name in tqdm(os.listdir(path), desc=f"Loading {category}"):
            try:
                img_path = os.path.join(path, img_name)
                image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                
                if image is not None:
                    image = cv2.resize(image, (48, 48))
                    # Pixels ko flat kar ke normalize karna (0-1 range)
                    data.append(image.flatten())
                    labels.append(label)
            except:
                continue
                
    return np.array(data) / 255.0, np.array(labels)

# Data load karein
X_train, y_train = load_data()

print(f"\n--- Training on {len(X_train)} images with SAGA solver ---")

# Humne 'class_weight' balanced rakha hai taake kam images wali categories (like disgust) bhi sahi se train hon
model = LogisticRegression(
    max_iter=5000, 
    solver='saga', 
    class_weight='balanced', 
    verbose=1, 
    n_jobs=-1  # Yeh computer ke saare cores use karega taake kaam jaldi ho
)

# Model train karein
model.fit(X_train, y_train)

# Save the brain
if not os.path.exists('models'):
    os.makedirs('models')

joblib.dump(model, 'models/emotion_model.pkl')
print("\n" + "="*40)
print("SUCCESS: Model Updated & Saved!")
print("="*40)