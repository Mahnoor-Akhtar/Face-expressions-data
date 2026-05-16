import os
import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import json
from datetime import datetime

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Define paths and classes
emotions = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
model_path = Path("models/emotion_cnn_model.h5")
test_data_dir = Path("data/test")

print("=" * 80)
print("EMOTION RECOGNITION MODEL - PERFORMANCE COMPARISON")
print("=" * 80)
print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# Check if model exists
if not model_path.exists():
    print(f"❌ Model not found at {model_path}")
    print("Please ensure emotion_cnn_model.h5 exists in the models directory")
    exit(1)

# Load the model
print("Loading model...")
try:
    model = load_model(model_path)
    print("✓ Model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit(1)

print()
print("=" * 80)
print("MODEL ARCHITECTURE")
print("=" * 80)
model.summary()

print()
print("=" * 80)
print("TESTING MODEL ON EMOTION CLASSES")
print("=" * 80)
print()

# Test on each emotion class
results = {}
correct_predictions = 0
total_predictions = 0

for emotion_idx, emotion in enumerate(emotions):
    emotion_path = test_data_dir / emotion
    
    if not emotion_path.exists():
        print(f"⚠ No test data found for {emotion}")
        continue
    
    # Get test images
    test_images = list(emotion_path.glob("*.jpg"))[:50]  # Test on first 50 images per class
    
    if len(test_images) == 0:
        print(f"⚠ No images found in {emotion}")
        continue
    
    emotion_correct = 0
    emotion_predictions = []
    
    for img_path in test_images:
        try:
            # Load and preprocess image
            img = image.load_img(img_path, target_size=(48, 48), color_mode='grayscale')
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0
            
            # Make prediction
            prediction = model.predict(img_array, verbose=0)
            predicted_emotion_idx = np.argmax(prediction)
            predicted_emotion = emotions[predicted_emotion_idx]
            confidence = float(prediction[0][predicted_emotion_idx])
            
            emotion_predictions.append({
                'predicted': predicted_emotion,
                'confidence': confidence
            })
            
            # Check if correct
            if predicted_emotion == emotion:
                emotion_correct += 1
                correct_predictions += 1
            
            total_predictions += 1
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue
    
    accuracy = (emotion_correct / len(test_images) * 100) if len(test_images) > 0 else 0
    results[emotion] = {
        'correct': emotion_correct,
        'total': len(test_images),
        'accuracy': accuracy,
        'predictions': emotion_predictions
    }
    
    status = "✓" if accuracy >= 70 else "⚠" if accuracy >= 50 else "❌"
    print(f"{status} {emotion.upper():<12} - Accuracy: {accuracy:>6.2f}% ({emotion_correct}/{len(test_images)})")

print()
print("=" * 80)
print("OVERALL RESULTS")
print("=" * 80)

overall_accuracy = (correct_predictions / total_predictions * 100) if total_predictions > 0 else 0
print(f"Overall Accuracy: {overall_accuracy:.2f}% ({correct_predictions}/{total_predictions})")
print()

# Suggestions for improvement
print("=" * 80)
print("SUGGESTIONS FOR MODEL IMPROVEMENT")
print("=" * 80)
print()
print("1. ADDRESS CLASS IMBALANCE:")
print("   - Use class weights: classes with fewer samples get higher weight")
print("   - Data augmentation: rotate, flip, zoom images for minority classes")
print("   - Oversampling: duplicate samples from minority classes")
print()
print("2. MODEL ARCHITECTURES TO COMPARE:")
print("   - ResNet50 (pre-trained): Better feature extraction")
print("   - VGG16 (pre-trained): Proven on FER tasks")
print("   - MobileNet: Faster, lighter model")
print("   - EfficientNet: Better accuracy with fewer parameters")
print()
print("3. TRAINING IMPROVEMENTS:")
print("   - Use transfer learning from pre-trained ImageNet models")
print("   - Implement data augmentation in training pipeline")
print("   - Use callbacks: EarlyStopping, ReduceLROnPlateau")
print("   - Implement dropout and regularization")
print()
print("4. TESTING & VALIDATION:")
print("   - Cross-validation to ensure model generalization")
print("   - Confusion matrix analysis")
print("   - Per-class precision, recall, F1-score metrics")
print()
print("=" * 80)
