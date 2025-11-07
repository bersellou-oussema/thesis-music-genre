# cnn_melspectrogram_gtzan_trackwise.py
# CNN (from scratch) on GTZAN Mel-spectrogram images with TRACK-WISE split + full diagnostics.
# Outputs: classification report (printed & .txt), final one-line summary, confusion matrix .png, saved model.

import os, glob, pathlib, random
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix

# -------------------- Config --------------------
SEED          = 42
IMG_SIZE      = (224, 224)
BATCH_SIZE    = 32
EPOCHS        = 20
LR            = 1e-3
VAL_FRACTION  = 0.20
TEST_FRACTION = 0.20

# Use your MEL spectrograms (the ones you generated earlier)
DATA_DIR = "spectrogram_dataset_windowed/gtzan"   # <-- mel-spectrogram PNGs grouped by genre subfolders
OUT_DIR  = "models"
os.makedirs(OUT_DIR, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# -------------------- Gather files -> DataFrame --------------------
paths, labels, tracks = [], [], []
genres = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
if not genres:
    raise FileNotFoundError(f"No class folders found under {DATA_DIR}")

for g in genres:
    gdir = os.path.join(DATA_DIR, g)
    for p in glob.glob(os.path.join(gdir, "*.png")):
        paths.append(p)
        labels.append(g)
        # infer track id from filename: "genre.00012_12345.png" -> "genre.00012"
        stem = pathlib.Path(p).stem
        track_id = stem.split("_")[0]  # stable per song
        tracks.append(track_id)

df = pd.DataFrame({"path": paths, "class": labels, "track": tracks})
if df.empty:
    raise RuntimeError(f"No PNG files found under {DATA_DIR}")

df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
print(f"Total images: {len(df)}")
print("Classes:", genres)
print("Per-class counts:", Counter(df["class"]))

# -------------------- Track-wise split (no leakage) --------------------
gss = GroupShuffleSplit(n_splits=1, test_size=TEST_FRACTION, random_state=SEED)
train_val_idx, test_idx = next(gss.split(df, groups=df["track"]))
df_train_val = df.iloc[train_val_idx].reset_index(drop=True)
df_test      = df.iloc[test_idx].reset_index(drop=True)

gss2 = GroupShuffleSplit(n_splits=1, test_size=VAL_FRACTION, random_state=SEED)
train_idx, val_idx = next(gss2.split(df_train_val, groups=df_train_val["track"]))
df_train = df_train_val.iloc[train_idx].reset_index(drop=True)
df_val   = df_train_val.iloc[val_idx].reset_index(drop=True)

# Sanity: no track overlap
assert set(df_train["track"]).isdisjoint(set(df_val["track"]))
assert set(df_train["track"]).isdisjoint(set(df_test["track"]))
assert set(df_val["track"]).isdisjoint(set(df_test["track"]))

# -------------------- Generators --------------------
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

def make_flow(gen, df, shuffle):
    return gen.flow_from_dataframe(
        dataframe=df,
        x_col="path",
        y_col="class",
        target_size=IMG_SIZE,
        color_mode="rgb",
        class_mode="categorical",
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        seed=SEED
    )

train_flow = make_flow(train_gen, df_train, shuffle=True)
val_flow   = make_flow(plain_gen, df_val,   shuffle=False)
test_flow  = make_flow(plain_gen, df_test,  shuffle=False)

# -------------------- Diagnostics --------------------
print("\n>>> TensorFlow:", tf.__version__)
try:
    build = tf.sysconfig.get_build_info() if hasattr(tf.sysconfig, "get_build_info") else {}
    print(">>> TF build:", {k: build.get(k) for k in ("cuda_version", "cudnn_version")})
except Exception as e:
    print(">>> TF build info unavailable:", e)

print("---- DATAFRAME SIZES ----")
print("df_train:", len(df_train), " df_val:", len(df_val), " df_test:", len(df_test))
print("---- GENERATOR SIZES ----")
print("train_flow.samples:", train_flow.samples)
print("val_flow.samples  :", val_flow.samples)
print("test_flow.samples :", test_flow.samples)
print("\n---- CLASS INDICES ----")
print(train_flow.class_indices)

num_classes = len(train_flow.class_indices)

xb, yb = next(train_flow)
print("\n---- ONE BATCH SANITY ----")
print("num_classes:", num_classes)
print("Batch shapes (x, y):", xb.shape, yb.shape)
print("One-hot sum (yb[0]):", float(yb[0].sum()))
print("Argmax label example:", int(yb[0].argmax()))
assert val_flow.samples > 0, "Validation generator is empty!"

# -------------------- Model (compact CNN) --------------------
def build_cnn(input_shape=(224,224,3), num_classes=10):
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
model.compile(optimizer=optimizers.Adam(LR), loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

# -------------------- Train --------------------
ckpt_path = os.path.join(OUT_DIR, "cnn_gtzan_mel_trackwise_best.h5")
cbs = [
    callbacks.ModelCheckpoint(ckpt_path, monitor="val_accuracy", save_best_only=True, verbose=1),
    callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1),
    callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True, verbose=1),
]

history = model.fit(
    train_flow,
    validation_data=val_flow,
    epochs=EPOCHS,
    callbacks=cbs,
    verbose=1
)

# -------------------- Evaluate on TEST --------------------
probs = model.predict(test_flow, verbose=1)
y_pred = probs.argmax(axis=1)
y_true = test_flow.classes
labels_order = list(train_flow.class_indices.keys())

report_str = classification_report(y_true, y_pred, target_names=labels_order, digits=4)
print("\n🎯 Classification Report (CNN, Mel-spectrograms, GTZAN, track-wise):\n")
print(report_str)

# One-line summary
rep = classification_report(y_true, y_pred, target_names=labels_order, digits=4, output_dict=True)
acc  = rep["accuracy"]
prec = rep["weighted avg"]["precision"]
rec  = rep["weighted avg"]["recall"]
f1   = rep["weighted avg"]["f1-score"]
print(f"\n✅ CNN (Track-wise, Mel) — Accuracy: {acc:.4f} | Weighted Precision: {prec:.4f} | Weighted Recall: {rec:.4f} | Weighted F1: {f1:.4f}")

# Save report
rep_path = os.path.join(OUT_DIR, "report_cnn_gtzan_mel_trackwise.txt")
with open(rep_path, "w", encoding="utf-8") as f:
    f.write(report_str)
print(f"💾 Saved report to {rep_path}")

# Confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
plt.figure(figsize=(8, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels_order, yticklabels=labels_order)
plt.title("Confusion Matrix (CNN, Mel-spectrograms, GTZAN, Track-wise)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
cm_path = os.path.join(OUT_DIR, "cm_cnn_gtzan_mel_trackwise.png")
plt.savefig(cm_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"💾 Saved confusion matrix to {cm_path}")

# Save model
model_path = os.path.join(OUT_DIR, "cnn_gtzan_mel_trackwise.keras")
model.save(model_path)
print(f"💾 Saved model to {model_path}")
