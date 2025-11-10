import os
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

# ---------------- Config ----------------
SEED       = 42
IMG_SIZE   = (224, 224)
BATCH_SIZE = 32
EPOCHS     = 20
LR         = 1e-3

BASE_DIR = "spectrogram_dataset_windowed_splits/gtzan"  # NEW dataset from splits
OUT_DIR  = "models"
os.makedirs(OUT_DIR, exist_ok=True)

# Reproducibility
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ---------------- Paths ----------------
train_dir = os.path.join(BASE_DIR, "train")
val_dir   = os.path.join(BASE_DIR, "val")
test_dir  = os.path.join(BASE_DIR, "test")

if not os.path.isdir(train_dir):
    raise FileNotFoundError(f"Train dir not found: {train_dir}")
if not os.path.isdir(val_dir):
    raise FileNotFoundError(f"Val dir not found: {val_dir}")
if not os.path.isdir(test_dir):
    raise FileNotFoundError(f"Test dir not found: {test_dir}")

# ---------------- Data Generators ----------------
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.05,
    height_shift_range=0.05,
    horizontal_flip=True,
    zoom_range=0.05,
    fill_mode="nearest"
)

plain_gen = ImageDataGenerator(rescale=1./255)

# Train generator defines class_indices
train_flow = train_gen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    color_mode="rgb",
    class_mode="categorical",
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED
)

# Use same class order for val & test
class_names = list(train_flow.class_indices.keys())

val_flow = plain_gen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    color_mode="rgb",
    class_mode="categorical",
    batch_size=BATCH_SIZE,
    shuffle=False,
    seed=SEED,
    classes=class_names
)

test_flow = plain_gen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    color_mode="rgb",
    class_mode="categorical",
    batch_size=BATCH_SIZE,
    shuffle=False,
    seed=SEED,
    classes=class_names
)

print("\n---- CLASS INDICES ----")
print(train_flow.class_indices)

num_classes = len(class_names)

# Sanity batch
xb, yb = next(train_flow)
print("\nSanity check:")
print("Batch x shape:", xb.shape)
print("Batch y shape:", yb.shape)
print("One-hot sum example:", float(yb[0].sum()))

# ---------------- Model ----------------
def build_cnn(input_shape=(224, 224, 3), num_classes=10):
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(32, (3,3), padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64, (3,3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(128, (3,3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(256, (3,3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    return model

model = build_cnn((*IMG_SIZE, 3), num_classes)
model.compile(
    optimizer=optimizers.Adam(LR),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
model.summary()

# ---------------- Callbacks ----------------
ckpt_path = os.path.join(OUT_DIR, "cnn_gtzan_mel_windowed_splits_best.h5")

cbs = [
    callbacks.ModelCheckpoint(
        ckpt_path,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    ),
    callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    ),
    callbacks.EarlyStopping(
        monitor="val_loss",
        patience=6,
        restore_best_weights=True,
        verbose=1
    ),
]

# ---------------- Train ----------------
history = model.fit(
    train_flow,
    validation_data=val_flow,
    epochs=EPOCHS,
    callbacks=cbs,
    verbose=1
)

# ---------------- Evaluate (TEST) ----------------
probs = model.predict(test_flow, verbose=1)
y_pred = probs.argmax(axis=1)
y_true = test_flow.classes

report_str = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    digits=4
)
print("\nClassification Report (CNN, Mel, GTZAN, windowed, fixed splits):\n")
print(report_str)

# Metrics summary
report_dict = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    digits=4,
    output_dict=True
)
acc  = report_dict["accuracy"]
prec = report_dict["weighted avg"]["precision"]
rec  = report_dict["weighted avg"]["recall"]
f1   = report_dict["weighted avg"]["f1-score"]

print(f"\n✅ Accuracy: {acc:.4f} | W-Precision: {prec:.4f} | W-Recall: {rec:.4f} | W-F1: {f1:.4f}")

# Save report
rep_path = os.path.join(OUT_DIR, "report_cnn_gtzan_mel_windowed_splits.txt")
with open(rep_path, "w", encoding="utf-8") as f:
    f.write(report_str)
print(f"💾 Saved report to {rep_path}")

# Confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
plt.figure(figsize=(8, 7))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)
plt.title("Confusion Matrix (CNN, Mel, GTZAN, windowed, fixed splits)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()

cm_path = os.path.join(OUT_DIR, "cm_cnn_gtzan_mel_windowed_splits.png")
plt.savefig(cm_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"💾 Saved confusion matrix to {cm_path}")

# Save model
model_path = os.path.join(OUT_DIR, "cnn_gtzan_mel_windowed_splits.keras")
model.save(model_path)
print(f"💾 Saved model to {model_path}")
