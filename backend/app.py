from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import tensorflow as tf
import numpy as np
import json
import os
import io

app = Flask(__name__)
CORS(app)

# =========================================================
# PATHS
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

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

# =========================================================
# LOAD MODEL
# =========================================================

print("=" * 60)
print("TRAFFIC SIGN RECOGNITION - FLASK SERVER")
print("=" * 60)

print("Loading traffic sign model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")

# =========================================================
# LOAD CLASS NAMES
# =========================================================

with open(
    CLASS_NAMES_PATH,
    "r",
    encoding="utf-8"
) as file:

    class_names = json.load(file)

print("Class names loaded successfully!")

# =========================================================
# HOME API
# =========================================================

@app.route("/", methods=["GET"])
def home():

    return jsonify({
        "message": "Traffic Sign Recognition API is running!",
        "status": "success"
    })


# =========================================================
# IMAGE PREDICTION
# =========================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        # Check image
        if "image" not in request.files:

            return jsonify({
                "error": "No image uploaded"
            }), 400

        file = request.files["image"]

        # Check filename
        if file.filename == "":

            return jsonify({
                "error": "No image selected"
            }), 400

        # Open image
        image = Image.open(file)

        # Convert image to RGB
        image = image.convert("RGB")

        # Resize image
        image = image.resize((64, 64))

        # Convert image to NumPy array
        image_array = np.array(
            image,
            dtype=np.float32
        )

        # Add batch dimension
        image_array = np.expand_dims(
            image_array,
            axis=0
        )

        # Predict
        predictions = model.predict(
            image_array,
            verbose=0
        )

        # Find highest probability class
        predicted_class = int(
            np.argmax(predictions[0])
        )

        # Calculate confidence
        confidence = float(
            predictions[0][predicted_class] * 100
        )

        # Get traffic sign name
        prediction_name = class_names.get(
            str(predicted_class),
            f"Class {predicted_class}"
        )

        # Display result in terminal
        print(
            f"Prediction: {prediction_name}"
        )

        print(
            f"Confidence: {confidence:.2f}%"
        )

        # Send result to React
        return jsonify({

            "prediction": prediction_name,

            "confidence": round(
                confidence,
                2
            ),

            "class_id": predicted_class

        })

    except Exception as e:

        print(
            "Prediction error:",
            str(e)
        )

        return jsonify({

            "error": "Failed to process image",

            "details": str(e)

        }), 500


# =========================================================
# CAMERA PREDICTION
# =========================================================

@app.route(
    "/predict-camera",
    methods=["POST"]
)
def predict_camera():

    try:

        # Check camera image
        if "image" not in request.files:

            return jsonify({
                "error": "No camera frame received"
            }), 400

        file = request.files["image"]

        # Read image data
        image_data = file.read()

        # Open image from memory
        image = Image.open(
            io.BytesIO(image_data)
        )

        # Convert to RGB
        image = image.convert("RGB")

        # Resize
        image = image.resize((64, 64))

        # Convert to NumPy array
        image_array = np.array(
            image,
            dtype=np.float32
        )

        # Add batch dimension
        image_array = np.expand_dims(
            image_array,
            axis=0
        )

        # Predict
        predictions = model.predict(
            image_array,
            verbose=0
        )

        # Get predicted class
        predicted_class = int(
            np.argmax(predictions[0])
        )

        # Get confidence
        confidence = float(
            predictions[0][predicted_class] * 100
        )

        # Get traffic sign name
        prediction_name = class_names.get(
            str(predicted_class),
            f"Class {predicted_class}"
        )

        # Display camera result
        print(
            f"Camera Prediction: "
            f"{prediction_name}"
        )

        print(
            f"Camera Confidence: "
            f"{confidence:.2f}%"
        )

        # Send result to React
        return jsonify({

            "prediction": prediction_name,

            "confidence": round(
                confidence,
                2
            ),

            "class_id": predicted_class

        })

    except Exception as e:

        print(
            "Camera prediction error:",
            str(e)
        )

        return jsonify({

            "error": "Failed to process camera frame",

            "details": str(e)

        }), 500


# =========================================================
# START SERVER
# =========================================================

if __name__ == "__main__":

    print("=" * 60)

    print(
        "Server starting..."
    )

    print(
        "URL: http://localhost:5000"
    )

    print("=" * 60)

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )