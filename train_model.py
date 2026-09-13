import os
import json
import numpy as np
from PIL import Image
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models

# ==============================
# CONFIGURATION
# ==============================

DATASET_PATH = r"dataset\GTSRB\Final_Training\Images"
MODEL_PATH = r"model\traffic_sign_model.keras"
CLASS_NAMES_PATH = r"model\class_names.json"

IMG_SIZE = 64
BATCH_SIZE = 64
EPOCHS = 15

# ==============================
# TRAFFIC SIGN CLASS NAMES
# ==============================

CLASS_NAMES = {
    0: "Speed limit (20km/h)",
    1: "Speed limit (30km/h)",
    2: "Speed limit (50km/h)",
    3: "Speed limit (60km/h)",
    4: "Speed limit (70km/h)",
    5: "Speed limit (80km/h)",
    6: "End of speed limit (80km/h)",
    7: "Speed limit (100km/h)",
    8: "Speed limit (120km/h)",
    9: "No passing",
    10: "No passing for vehicles over 3.5 tons",
    11: "Right-of-way at intersection",
    12: "Priority road",
    13: "Yield",
    14: "Stop",
    15: "No vehicles",
    16: "Vehicles over 3.5 tons prohibited",
    17: "No entry",
    18: "General caution",
    19: "Dangerous curve left",
    20: "Dangerous curve right",
    21: "Double curve",
    22: "Bumpy road",
    23: "Slippery road",
    24: "Road narrows on the right",
    25: "Road work",
    26: "Traffic signals",
    27: "Pedestrians",
    28: "Children crossing",
    29: "Bicycles crossing",
    30: "Beware of ice/snow",
    31: "Wild animals crossing",
    32: "End of all speed and passing limits",
    33: "Turn right ahead",
    34: "Turn left ahead",
    35: "Ahead only",
    36: "Go straight or right",
    37: "Go straight or left",
    38: "Keep right",
    39: "Keep left",
    40: "Roundabout mandatory",
    41: "End of no passing",
    42: "End of no passing by vehicles over 3.5 tons"
}

# ==============================
# CHECK DATASET
# ==============================

if not os.path.exists(DATASET_PATH):
    print("ERROR: Dataset folder not found!")
    print(DATASET_PATH)
    exit()

print("=" * 60)
print("TRAFFIC SIGN RECOGNITION - CNN TRAINING")
print("=" * 60)

# ==============================
# LOAD IMAGES
# ==============================

images = []
labels = []

print("\nLoading dataset...")
print("This may take a few minutes...\n")

for class_id in range(43):

    class_folder = os.path.join(DATASET_PATH, f"{class_id:05d}")

    if not os.path.exists(class_folder):
        print(f"WARNING: Class folder missing: {class_folder}")
        continue

    image_files = [
        file for file in os.listdir(class_folder)
        if file.lower().endswith((".ppm", ".png", ".jpg", ".jpeg"))
    ]

    print(
        f"Class {class_id:02d}: "
        f"{len(image_files)} images - "
        f"{CLASS_NAMES[class_id]}"
    )

    for image_file in image_files:

        image_path = os.path.join(class_folder, image_file)

        try:
            image = Image.open(image_path)
            image = image.convert("RGB")
            image = image.resize((IMG_SIZE, IMG_SIZE))

            images.append(np.array(image, dtype=np.uint8))
            labels.append(class_id)

        except Exception as e:
            print(f"Could not load: {image_path}")

# ==============================
# CONVERT DATA
# ==============================

print("\nPreparing data...")

X = np.array(images, dtype=np.uint8)
y = np.array(labels, dtype=np.int64)

print(f"Total images: {len(X)}")
print(f"Image shape: {X.shape}")
print(f"Number of classes: {len(np.unique(y))}")

# ==============================
# TRAIN / VALIDATION SPLIT
# ==============================

print("\nSplitting dataset...")

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"Training images: {len(X_train)}")
print(f"Validation images: {len(X_val)}")

# ==============================
# CREATE TF DATASETS
# ==============================

train_dataset = tf.data.Dataset.from_tensor_slices(
    (X_train, y_train)
)

val_dataset = tf.data.Dataset.from_tensor_slices(
    (X_val, y_val)
)

# ==============================
# DATA AUGMENTATION
# ==============================

data_augmentation = tf.keras.Sequential([
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.10),
    layers.RandomTranslation(0.10, 0.10)
])

# ==============================
# CNN MODEL
# ==============================

model = models.Sequential([

    layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),

    # Normalize pixels
    layers.Rescaling(1.0 / 255),

    # Data augmentation
    data_augmentation,

    # CNN Layer 1
    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    # CNN Layer 2
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    # CNN Layer 3
    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    # CNN Layer 4
    layers.Conv2D(256, (3, 3), activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    # Flatten
    layers.Flatten(),

    # Fully connected layers
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.5),

    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),

    # 43 traffic sign classes
    layers.Dense(43, activation="softmax")
])

# ==============================
# COMPILE MODEL
# ==============================

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\n" + "=" * 60)
print("CNN MODEL")
print("=" * 60)

model.summary()

# ==============================
# PREPARE DATASETS
# ==============================

train_dataset = (
    train_dataset
    .shuffle(10000)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

val_dataset = (
    val_dataset
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

# ==============================
# CALLBACKS
# ==============================

callbacks = [

    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=4,
        restore_best_weights=True
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=0.00001
    )
]

# ==============================
# TRAIN MODEL
# ==============================

print("\n" + "=" * 60)
print("STARTING CNN TRAINING")
print("=" * 60)

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=callbacks
)

# ==============================
# EVALUATE MODEL
# ==============================

print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

loss, accuracy = model.evaluate(val_dataset)

print(f"\nValidation Loss: {loss:.4f}")
print(f"Validation Accuracy: {accuracy * 100:.2f}%")

# ==============================
# CREATE MODEL DIRECTORY
# ==============================

os.makedirs("model", exist_ok=True)

# ==============================
# SAVE MODEL
# ==============================

model.save(MODEL_PATH)

print("\nModel saved successfully:")
print(MODEL_PATH)

# ==============================
# SAVE CLASS NAMES
# ==============================

with open(CLASS_NAMES_PATH, "w", encoding="utf-8") as file:
    json.dump(CLASS_NAMES, file, indent=4)

print("Class names saved successfully:")
print(CLASS_NAMES_PATH)

# ==============================
# FINISHED
# ==============================

print("\n" + "=" * 60)
print("TRAINING COMPLETED SUCCESSFULLY!")
print("=" * 60)

print(f"\nFinal validation accuracy: {accuracy * 100:.2f}%")

print("\nFiles created:")
print("1. model\\traffic_sign_model.keras")
print("2. model\\class_names.json")

print("\nNext step:")
print("Connect this trained model to the Flask backend.")