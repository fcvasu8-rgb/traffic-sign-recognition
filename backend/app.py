import os
import json
import numpy as np
from PIL import Image
from io import BytesIO

from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf


# ============================================================
# SETTINGS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "traffic_sign_model.keras"
)

CLASS_NAMES_PATH = os.path.join(
    BASE_DIR,
    "model",
    "class_names.json"
)

IMG_SIZE = 64


# ============================================================
# TENSORFLOW CPU SETTINGS
# ============================================================

# Render free instance has very limited CPU.
# Limiting TensorFlow threads prevents unnecessary CPU usage.

try:
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
except Exception:
    pass


# ============================================================
# FLASK
# ============================================================

app = Flask(__name__)
CORS(app)


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 60)
print("TRAFFIC SIGN RECOGNITION - FLASK SERVER")
print("=" * 60)

print("Loading traffic sign model...")

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("Model loaded successfully!")


# ============================================================
# LOAD CLASS NAMES
# ============================================================

with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
    class_names = json.load(f)

print("Class names loaded successfully!")


# ============================================================
# MODEL WARM-UP
# ============================================================

print("Warming up TensorFlow model...")

dummy_image = np.zeros(
    (1, IMG_SIZE, IMG_SIZE, 3),
    dtype=np.float32
)

# Run one prediction when the server starts.
# This initializes TensorFlow before real users send requests.
_ = model(dummy_image, training=False).numpy()

print("Model warm-up completed!")
print("=" * 60)


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def prepare_image(image_bytes):

    image = Image.open(
        BytesIO(image_bytes)
    ).convert("RGB")

    image = image.resize(
        (IMG_SIZE, IMG_SIZE)
    )

    image_array = np.array(
        image,
        dtype=np.float32
    )

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array


# ============================================================
# HOME
# ============================================================

@app.route("/", methods=["GET"])
def home():

    return jsonify({
        "message": "Traffic Sign Recognition API is running!",
        "status": "success"
    })


# ============================================================
# IMAGE PREDICTION
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        if "image" not in request.files:

            return jsonify({
                "error": "No image uploaded"
            }), 400

        file = request.files["image"]

        image_bytes = file.read()

        input_image = prepare_image(
            image_bytes
        )

        # Direct model call is faster than model.predict()
        predictions = model(
            input_image,
            training=False
        ).numpy()[0]

        class_id = int(
            np.argmax(predictions)
        )

        confidence = float(
            predictions[class_id] * 100
        )

        prediction = class_names[class_id]

        print(
            f"Prediction: {prediction} | "
            f"Confidence: {confidence:.2f}%"
        )

        return jsonify({
            "prediction": prediction,
            "confidence": round(confidence, 2),
            "class_id": class_id
        })

    except Exception as e:

        print("Prediction error:", str(e))

        return jsonify({
            "error": str(e)
        }), 500


# ============================================================
# CAMERA PREDICTION
# ============================================================

@app.route("/predict-camera", methods=["POST"])
def predict_camera():

    try:

        if "image" not in request.files:

            return jsonify({
                "error": "No camera image received"
            }), 400

        file = request.files["image"]

        image_bytes = file.read()

        input_image = prepare_image(
            image_bytes
        )

        predictions = model(
            input_image,
            training=False
        ).numpy()[0]

        class_id = int(
            np.argmax(predictions)
        )

        confidence = float(
            predictions[class_id] * 100
        )

        prediction = class_names[class_id]

        print(
            f"Camera prediction: {prediction} | "
            f"Confidence: {confidence:.2f}%"
        )

        return jsonify({
            "prediction": prediction,
            "confidence": round(confidence, 2),
            "class_id": class_id
        })

    except Exception as e:

        print("Camera prediction error:", str(e))

        return jsonify({
            "error": str(e)
        }), 500


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )