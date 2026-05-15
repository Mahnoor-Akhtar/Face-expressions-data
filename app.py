import os
import hashlib
from datetime import datetime, timezone
import cv2
import numpy as np
import tensorflow as tf
import base64
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient  # type: ignore[reportMissingImports]
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

app = Flask(__name__)
CORS(app)  # React frontend se connection ke liye zaroori hai

MONGO_URI = os.getenv("MONGO_URI", "mongodb://127.0.0.1:27017")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "emosense_ai")
MONGO_COLLECTION_NAME = os.getenv("MONGO_COLLECTION_NAME", "model_metrics")

metrics_collection = None
try:
    mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=1500)
    mongo_client.admin.command("ping")
    metrics_collection = mongo_client[MONGO_DB_NAME][MONGO_COLLECTION_NAME]
    metrics_collection.create_index("cache_key", unique=True)
    metrics_collection.create_index("updated_at")
    print("--- MongoDB connected for metrics cache ---")
except Exception as exc:
    metrics_collection = None
    print(f"--- MongoDB cache disabled: {exc} ---")

# Predictions collection (stores each prediction request/result)
predictions_collection = None
try:
    if mongo_client is not None:
        predictions_collection = mongo_client[MONGO_DB_NAME]["predictions"]
        predictions_collection.create_index("timestamp")
        predictions_collection.create_index("image_hash")
        print("--- MongoDB connected for predictions storage ---")
except Exception as exc:
    predictions_collection = None
    print(f"--- MongoDB predictions disabled: {exc} ---")

# 1. CNN Model aur Face Cascade Load karein
MODEL_PATH = 'models/emotion_cnn_model.h5'

if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
    print("--- CNN Model Loaded Successfully ---")
else:
    print("ERROR: Model file not found! Please train the CNN first.")

# Try to load alternative models (logistic variants) if present
LOGISTIC_PATH = 'modelss/emotion_model.pkl'
LOGISTIC_BALANCED_PATH = 'modelsss/emotion_model.pkl'
logistic_model = None
logistic_balanced_model = None
try:
    if os.path.exists(LOGISTIC_PATH):
        logistic_model = joblib.load(LOGISTIC_PATH)
        print("--- Logistic model loaded ---")
    if os.path.exists(LOGISTIC_BALANCED_PATH):
        logistic_balanced_model = joblib.load(LOGISTIC_BALANCED_PATH)
        print("--- Balanced logistic model loaded ---")
except Exception as _:
    logistic_model = None
    logistic_balanced_model = None

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Emotions ki list (Sequence wahi rakhni hai jo training mein thi)
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]


def get_file_signature(file_path):
    if not os.path.exists(file_path):
        return "missing"
    stat = os.stat(file_path)
    payload = f"{os.path.abspath(file_path)}:{stat.st_mtime_ns}:{stat.st_size}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_cache_key():
    source_bits = [
        "v3",
        get_file_signature('models/emotion_cnn_model.h5'),
        get_file_signature('modelss/emotion_model.pkl'),
        get_file_signature('modelsss/emotion_model.pkl'),
        get_file_signature('data/test'),
    ]
    return hashlib.sha256("|".join(source_bits).encode("utf-8")).hexdigest()


def load_metrics_cache(cache_key):
    if metrics_collection is None:
        return None
    doc = metrics_collection.find_one({"cache_key": cache_key}, {"_id": 0})
    if not doc:
        return None
    results = doc.get("results") or {}
    results["cache_key"] = doc.get("cache_key")
    results["cached"] = True
    updated_at = doc.get("updated_at")
    results["cached_at"] = updated_at.isoformat() if updated_at else None
    return results


def cache_is_complete(results):
    required_models = ["cnn", "logistic", "logistic_balanced"]
    required_summary_keys = ["accuracy", "macro_avg", "weighted_avg", "per_class", "confusion_matrix"]
    for model_key in required_models:
        model = results.get(model_key)
        if not model:
            return False
        for field in required_summary_keys:
            if field not in model:
                return False
        if not model.get("per_class"):
            return False
    return True


def store_metrics_cache(cache_key, results):
    if metrics_collection is None:
        return
    metrics_collection.update_one(
        {"cache_key": cache_key},
        {
            "$set": {
                "cache_key": cache_key,
                "results": results,
                "updated_at": datetime.now(timezone.utc),
            }
        },
        upsert=True,
    )


def compute_model_metrics():
    CATEGORIES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
    TEST_DIR = 'data/test/'

    X_test_cnn = []
    X_test_logistic = []
    y_true = []
    for category in CATEGORIES:
        path = os.path.join(TEST_DIR, category)
        label = CATEGORIES.index(category)
        if not os.path.exists(path):
            continue
        for img_name in os.listdir(path):
            try:
                img_path = os.path.join(path, img_name)
                image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                image = cv2.resize(image, (48, 48))
                X_test_cnn.append(image)
                X_test_logistic.append(image.flatten())
                y_true.append(label)
            except:
                continue

    if len(y_true) == 0:
        return None, {"error": "No test data found"}

    X_test_cnn = np.array(X_test_cnn).reshape(-1, 48, 48, 1) / 255.0
    X_test_logistic = np.array(X_test_logistic) / 255.0
    y_true = np.array(y_true)

    results = {}

    cnn_path = 'models/emotion_cnn_model.h5'
    if os.path.exists(cnn_path):
        cnn_model = tf.keras.models.load_model(cnn_path)
        preds = cnn_model.predict(X_test_cnn, verbose=0)
        y_pred_cnn = np.argmax(preds, axis=1)

        acc = float(accuracy_score(y_true, y_pred_cnn))
        report = classification_report(y_true, y_pred_cnn, target_names=CATEGORIES, output_dict=True, zero_division=0)
        prec, rec, f1, sup = precision_recall_fscore_support(y_true, y_pred_cnn, zero_division=0)
        cm = confusion_matrix(y_true, y_pred_cnn).tolist()

        per_class = {}
        for i, cat in enumerate(CATEGORIES):
            per_class[cat] = {
                'precision': float(prec[i]),
                'recall': float(rec[i]),
                'f1': float(f1[i]),
                'support': int(sup[i])
            }

        results['cnn'] = {
            'accuracy': acc,
            'macro_avg': {
                'precision': float(report['macro avg']['precision']),
                'recall': float(report['macro avg']['recall']),
                'f1': float(report['macro avg']['f1-score']),
            },
            'weighted_avg': {
                'precision': float(report['weighted avg']['precision']),
                'recall': float(report['weighted avg']['recall']),
                'f1': float(report['weighted avg']['f1-score']),
            },
            'per_class': per_class,
            'confusion_matrix': cm
        }

    for model_key, model_path, label_name in [
        ('logistic', 'modelss/emotion_model.pkl', 'Logistic Regression'),
        ('logistic_balanced', 'modelsss/emotion_model.pkl', 'Balanced Logistic Regression'),
    ]:
        if os.path.exists(model_path):
            log_model = joblib.load(model_path)
            y_pred_log = log_model.predict(X_test_logistic)

            acc = float(accuracy_score(y_true, y_pred_log))
            report = classification_report(y_true, y_pred_log, target_names=CATEGORIES, output_dict=True, zero_division=0)
            prec, rec, f1, sup = precision_recall_fscore_support(y_true, y_pred_log, zero_division=0)
            cm = confusion_matrix(y_true, y_pred_log).tolist()

            per_class = {}
            for i, cat in enumerate(CATEGORIES):
                per_class[cat] = {
                    'precision': float(prec[i]),
                    'recall': float(rec[i]),
                    'f1': float(f1[i]),
                    'support': int(sup[i])
                }

            results[model_key] = {
                'label': label_name,
                'accuracy': acc,
                'macro_avg': {
                    'precision': float(report['macro avg']['precision']),
                    'recall': float(report['macro avg']['recall']),
                    'f1': float(report['macro avg']['f1-score']),
                },
                'weighted_avg': {
                    'precision': float(report['weighted avg']['precision']),
                    'recall': float(report['weighted avg']['recall']),
                    'f1': float(report['weighted avg']['f1-score']),
                },
                'per_class': per_class,
                'confusion_matrix': cm
            }

    if not results:
        return None, {"error": "No models found"}

    return results, None

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
        
        # Model selection (default to CNN)
        model_choice = request.json.get('model', 'cnn')
        predicted_emotion = 'Neutral'
        probs = None

        if model_choice == 'cnn' or model is None:
            # CNN prediction (default)
            prediction = model.predict(roi_final, verbose=0)
            probs = [float(x) for x in prediction[0].tolist()]
            max_index = int(np.argmax(prediction[0]))
            predicted_emotion = EMOTIONS[max_index]
        elif model_choice == 'logistic' and logistic_model is not None:
            roi_flat = roi_resized.flatten().reshape(1, -1).astype('float32') / 255.0
            try:
                probs_arr = logistic_model.predict_proba(roi_flat)
                probs = [float(x) for x in probs_arr[0].tolist()]
            except Exception:
                # fallback: predict only
                pred = logistic_model.predict(roi_flat)
                probs = [0.0] * len(EMOTIONS)
                probs[int(pred[0])] = 1.0
            max_index = int(np.argmax(probs))
            predicted_emotion = EMOTIONS[max_index]
        elif model_choice == 'logistic_balanced' and logistic_balanced_model is not None:
            roi_flat = roi_resized.flatten().reshape(1, -1).astype('float32') / 255.0
            try:
                probs_arr = logistic_balanced_model.predict_proba(roi_flat)
                probs = [float(x) for x in probs_arr[0].tolist()]
            except Exception:
                pred = logistic_balanced_model.predict(roi_flat)
                probs = [0.0] * len(EMOTIONS)
                probs[int(pred[0])] = 1.0
            max_index = int(np.argmax(probs))
            predicted_emotion = EMOTIONS[max_index]
        else:
            # If requested model not available, fall back to CNN if possible
            if model is not None:
                prediction = model.predict(roi_final, verbose=0)
                probs = [float(x) for x in prediction[0].tolist()]
                max_index = int(np.argmax(prediction[0]))
                predicted_emotion = EMOTIONS[max_index]
            else:
                return jsonify({"emotion": "Model Not Available"}), 400

        # Attempt to store prediction record in MongoDB (if available)
        try:
            if predictions_collection is not None:
                # Hash image bytes to avoid storing raw images
                image_hash = hashlib.sha256(img_data).hexdigest()
                record = {
                    "predicted_emotion": predicted_emotion,
                    "probabilities": probs if probs is not None else [],
                    "timestamp": datetime.now(timezone.utc),
                    "image_hash": image_hash,
                    "model_used": model_choice if 'model_choice' in locals() else 'cnn',
                }
                predictions_collection.insert_one(record)
        except Exception as _:
            # Fail silently to avoid changing API behavior
            pass

        # Keep response identical to previous behavior
        return jsonify({"emotion": predicted_emotion})

    except Exception as e:
        print(f"Backend Error: {e}")
        return jsonify({"emotion": "Server Error"})


@app.route('/api/models/metrics', methods=['GET'])
def models_metrics():
    try:
        refresh = request.args.get("refresh", "0") == "1"
        cache_key = build_cache_key()

        if not refresh:
            cached = load_metrics_cache(cache_key)
            if cached and cache_is_complete(cached):
                return jsonify(cached)

        results, error = compute_model_metrics()
        if error:
            return jsonify(error), 400

        payload = dict(results)
        payload["cache_key"] = cache_key
        payload["cached"] = False
        payload["generated_at"] = datetime.now(timezone.utc).isoformat()

        store_metrics_cache(cache_key, results)
        return jsonify(payload)
    except Exception as e:
        print(f"Metrics Error: {e}")
        return jsonify({"error": "Server Error"}), 500


@app.route('/api/reports/generate', methods=['GET'])
def generate_report():
    """Return a report of all stored predictions.

    Response format: JSON list of records with timestamp, predicted_emotion,
    probabilities, and image_hash. If MongoDB is not configured, returns 503.
    """
    try:
        if predictions_collection is None:
            return jsonify({"error": "Predictions storage not configured"}), 503

        # Return full documents so frontend can display all stored fields
        docs = list(predictions_collection.find({}).sort("timestamp", 1))
        out = []
        for d in docs:
            doc = dict(d)
            _id = doc.get("_id")
            if _id is not None:
                try:
                    doc["_id"] = str(_id)
                except Exception:
                    doc["_id"] = repr(_id)
            ts = doc.get("timestamp")
            if isinstance(ts, datetime):
                doc["timestamp"] = ts.isoformat()
            out.append(doc)

        return jsonify({"count": len(out), "predictions": out})
    except Exception as e:
        print(f"Report Error: {e}")
        return jsonify({"error": "Server Error"}), 500

if __name__ == '__main__':
    # Debug mode on rakhein taake koi bhi error terminal mein foran nazar aaye
    app.run(debug=True, port=5000)