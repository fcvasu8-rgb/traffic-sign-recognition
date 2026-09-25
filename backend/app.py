import os
import json
from io import BytesIO

import numpy as np
from PIL import Image

from flask import Flask, request, jsonify
from flask_cors import CORS

import tensorflow as tf


# ============================================================
# SETTINGS
# ============================================================

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

IMG_SIZE = 64


# ============================================================
# TENSORFLOW CPU SETTINGS
# ============================================================

# Render Free has limited CPU resources.
# Limiting TensorFlow threads helps prevent excessive CPU usage.

try:
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
except Exception:
    pass


# ============================================================
# FLASK APP
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

try:

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    print("Model loaded successfully!")

except Exception as e:

    print("ERROR loading model:")
    print(str(e))

    raise


# ============================================================
# LOAD CLASS NAMES
# ============================================================

print("Loading class names...")

try:

    with open(
        CLASS_NAMES_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        class_names = json.load(f)

    print("Class names loaded successfully!")

    print(
        "Class names type:",
        type(class_names).__name__
    )

except Exception as e:

    print("ERROR loading class names:")
    print(str(e))

    raise


# ============================================================
# GET CLASS NAME
# ============================================================

def get_class_name(class_id):

    """
    Supports both formats:

    List:
    [
        "Speed limit (20km/h)",
        "Speed limit (30km/h)",
        ...
    ]

    Dictionary:
    {
        "0": "Speed limit (20km/h)",
        "1": "Speed limit (30km/h)",
        ...
    }
    """

    # If class_names is a dictionary
    if isinstance(class_names, dict):

        # JSON dictionary keys are normally strings
        key = str(class_id)

        if key in class_names:

            return class_names[key]

        # Extra fallback in case keys are integers
        if class_id in class_names:

            return class_names[class_id]

        raise KeyError(
            f"Class ID {class_id} not found in class_names.json"
        )

    # If class_names is a list
    elif isinstance(class_names, list):

        if 0 <= class_id < len(class_names):

            return class_names[class_id]

        raise IndexError(
            f"Class ID {class_id} is outside class_names list"
        )

    else:

        raise TypeError(
            "class_names.json must contain either "
            "a list or dictionary"
        )


# ============================================================
# MODEL WARM-UP
# ============================================================

print("Warming up TensorFlow model...")

try:

    dummy_image = np.zeros(
        (
            1,
            IMG_SIZE,
            IMG_SIZE,
            3
        ),
        dtype=np.float32
    )

    # Run one prediction when the server starts.
    # This initializes TensorFlow before the first real request.

    _ = model(
        dummy_image,
        training=False
    ).numpy()

    print("Model warm-up completed!")

except Exception as e:

    print("WARNING: Model warm-up failed:")
    print(str(e))


print("=" * 60)
print("SERVER READY")
print("=" * 60)


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def prepare_image(image_bytes):

    """
    Converts uploaded image bytes into the format
    expected by the CNN model.
    """

    image = Image.open(
        BytesIO(image_bytes)
    ).convert("RGB")

    # Resize to model input size
    image = image.resize(
        (IMG_SIZE, IMG_SIZE)
    )

    # Convert to NumPy float32
    image_array = np.array(
        image,
        dtype=np.float32
    )

    # Add batch dimension
    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # IMPORTANT:
    # Do NOT divide by 255 here.
    #
    # Your Keras model already contains:
    # Rescaling(1/255)
    #
    # So the model performs normalization internally.

    return image_array


# ============================================================
# HOME ROUTE
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

        # Check whether image exists
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

        # Read image
        image_bytes = file.read()

        if not image_bytes:

            return jsonify({
                "error": "Uploaded image is empty"
            }), 400

        # Prepare image
        input_image = prepare_image(
            image_bytes
        )

        # ====================================================
        # CNN PREDICTION
        # ====================================================

        predictions = model(
            input_image,
            training=False
        ).numpy()[0]

        # Get predicted class
        class_id = int(
            np.argmax(predictions)
        )

        # Get confidence
        confidence = float(
            predictions[class_id] * 100
        )

        # Get human-readable class name
        prediction = get_class_name(
            class_id
        )

        print(
            f"Prediction: {prediction} | "
            f"Class ID: {class_id} | "
            f"Confidence: {confidence:.2f}%"
        )

        # Return result
        return jsonify({

            "prediction": prediction,

            "confidence": round(
                confidence,
                2
            ),

            "class_id": class_id

        })

    except Exception as e:

        print(
            "Prediction error:",
            repr(e)
        )

        return jsonify({

            "error": str(e)

        }), 500


# ============================================================
# CAMERA PREDICTION
# ============================================================

@app.route("/predict-camera", methods=["POST"])
def predict_camera():

    try:

        # Check image
        if "image" not in request.files:

            return jsonify({
                "error": "No camera image received"
            }), 400

        file = request.files["image"]

        # Read image
        image_bytes = file.read()

        if not image_bytes:

            return jsonify({
                "error": "Camera image is empty"
            }), 400

        # Prepare image
        input_image = prepare_image(
            image_bytes
        )

        # ====================================================
        # CNN PREDICTION
        # ====================================================

        predictions = model(
            input_image,
            training=False
        ).numpy()[0]

        # Predicted class
        class_id = int(
            np.argmax(predictions)
        )

        # Confidence
        confidence = float(
            predictions[class_id] * 100
        )

        # Class name
        prediction = get_class_name(
            class_id
        )

        print(
            f"Camera prediction: {prediction} | "
            f"Class ID: {class_id} | "
            f"Confidence: {confidence:.2f}%"
        )

        return jsonify({

            "prediction": prediction,

            "confidence": round(
                confidence,
                2
            ),

            "class_id": class_id

        })

    except Exception as e:

        print(
            "Camera prediction error:",
            repr(e)
        )

        return jsonify({

            "error": str(e)

        }), 500


# ============================================================
# RUN SERVER LOCALLY
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("Starting Flask server...")
    print("Local URL: http://localhost:5000")
    print("=" * 60)

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )