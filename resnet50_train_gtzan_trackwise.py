# resnet50_train_gtzan_trackwise.py
# ResNet50 on GTZAN spectrogram images with *track-wise* split
# - builds splits by track id (so all segments from a song stay together)
# - uses class_indices to determine num_classes (no .num_classes attribute)
# - prints dataset sanity info
# - saves confusion matrix & report

import os
import glob
import random
import pathlib
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras import layers, models, optimizers, callbacks

# -----------------------
# CONFIG
# -----------------------
DATA_DIR   = "spectrogram_dataset_windowed/gtzan"  # spectrograms per-genre folders
OUT_DIR    = "models"
IMG_SIZE   = (224, 224)
BATCH_SIZE = 32
SEED       = 42
VAL_FRACTION  = 0.20
TEST_FRACTION = 0.20
EPOCHS        = 15
LR            = 1e-4

os.makedirs(OUT_DIR, exist_ok=True)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# -----------------------
# Gather files → DataFrame
# -----------------------
paths = []
labels = []
tracks = []  # group key

genres = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
for g in genres:
    gdir = os.path.join(DATA_DIR, g)
    for p in glob.glob(os.path.join(gdir, "*.png")):
        paths.append(p)
        labels.append(g)
        # infer track id from filename "genre.00012_12345.png" → "genre.00012"
        stem = pathlib.Path(p).stem  # e.g., "pop.00012_3456"
        track_id = stem.split("_")[0]  # before first "_"
        tracks.append(track_id)

df = pd.DataFrame({"path": paths, "class": labels, "track": tracks})
df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

print(f"Total spectrograms: {len(df)}")
print("Classes:", genres)
print("Per-class counts:", Counter(df["class"]))

# -----------------------
# Track-wise stratified split using GroupShuffleSplit
# -----------------------
gss = GroupShuffleSplit(n_splits=1, test_size=TEST_FRACTION, random_state=SEED)
train_val_idx, test_idx = next(gss.split(df, groups=df["track"]))
df_train_val = df.iloc[train_val_idx].reset_index(drop=True)
df_test      = df.iloc[test_idx].reset_index(drop=True)

# split train_val into train/val again by groups
gss2 = GroupShuffleSplit(n_splits=1, test_size=VAL_FRACTION, random_state=SEED)
train_idx, val_idx = next(gss2.split(df_train_val, groups=df_train_val["track"]))
df_train = df_train_val.iloc[train_idx].reset_index(drop=True)
df_val   = df_train_val.iloc[val_idx].reset_index(drop=True)

# -----------------------
# Generators (use preprocess_input for ResNet50)
# -----------------------
train_gen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=10,
    width_shift_range=0.05,
    height_shift_range=0.05,
    horizontal_flip=True,
    fill_mode="nearest",
)

plain_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

def make_flow(gen, df, shuffle, subset_name):
    return gen.flow_from_dataframe(
        df,
        x_col="path",
        y_col="class",
        target_size=IMG_SIZE,
        color_mode="rgb",
        class_mode="categorical",
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        seed=SEED
    )

train_flow = make_flow(train_gen, df_train, shuffle=True,  subset_name="train")
val_flow   = make_flow(plain_gen, df_val,   shuffle=False, subset_name="val")
test_flow  = make_flow(plain_gen, df_test,  shuffle=False, subset_name="test")

# -----------------------
# Diagnostics (safe prints)
# -----------------------
print("\n>>> Python :", tf.__version__)
print(">>> TensorFlow:", tf.__version__)
try:
    build = tf.sysconfig.get_build_info() if hasattr(tf.sysconfig, "get_build_info") else {}
    print(">>> TF build:", {k: build.get(k) for k in ("cuda_version", "cudnn_version")})
except Exception as e:
    print(">>> TF build info unavailable:", e)

print("---- DATAFRAME SIZES ----")
print("df_train:", len(df_train), " df_val:", len(df_val), " df_test:", len(df_test))
print("Unique classes in train:", sorted(df_train["class"].unique()))
print("Unique classes in test :", sorted(df_test["class"].unique()))

print("\n---- GENERATOR SIZES ----")
print("train_flow.samples:", train_flow.samples)
print("val_flow.samples  :", val_flow.samples)
print("test_flow.samples :", test_flow.samples)

print("\n---- CLASS INDICES (order used by Keras) ----")
print(train_flow.class_indices)

# Determine number of classes from mapping (portable across Keras versions)
num_classes = len(train_flow.class_indices)

# One batch sanity check
xb, yb = next(train_flow)
print("\n---- ONE BATCH SANITY ----")
print("num_classes:", num_classes)
print("Batch shapes (x, y):", xb.shape, yb.shape)
print("One-hot sum (yb[0]):", float(yb[0].sum()))
print("Argmax label example:", int(yb[0].argmax()))

assert val_flow.samples > 0, "Validation generator is empty — check group split!"

# -----------------------
# Model
# -----------------------
base = ResNet50(
    include_top=False,
    weights="imagenet",
    input_shape=(*IMG_SIZE, 3)
)
base.trainable = False  # fine-tune later if needed

inp = layers.Input(shape=(*IMG_SIZE, 3))
x = base(inp, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
out = layers.Dense(num_classes, activation="softmax")(x)
model = models.Model(inp, out)

model.compile(
    optimizer=optimizers.Adam(LR),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -----------------------
# Callbacks
# -----------------------
ckpt_path = os.path.join(OUT_DIR, "resnet50_gtzan_trackwise_best.h5")
cbs = [
    callbacks.ModelCheckpoint(ckpt_path, monitor="val_accuracy", save_best_only=True, verbose=1),
    callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1),
    callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True, verbose=1),
]

steps_per_epoch   = train_flow.samples // BATCH_SIZE
validation_steps  = max(1, val_flow.samples // BATCH_SIZE)

history = model.fit(
    train_flow,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_flow,
    validation_steps=validation_steps,
    callbacks=cbs,
    verbose=1
)

# -----------------------
# Evaluation on TEST
# -----------------------
test_steps = int(np.ceil(test_flow.samples / BATCH_SIZE))
probs = model.predict(test_flow, steps=test_steps, verbose=1)
y_pred = probs.argmax(axis=1)
y_true = test_flow.classes
labels_order = list(train_flow.class_indices.keys())

print("\n🎯 Classification Report (GTZAN, track-wise):")
report_txt = classification_report(y_true, y_pred, target_names=labels_order, digits=4)
print(report_txt)

# Save report
rep_path = os.path.join(OUT_DIR, "report_resnet50_gtzan_trackwise.txt")
with open(rep_path, "w", encoding="utf-8") as f:
    f.write(report_txt)

# Confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
plt.figure(figsize=(8, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels_order, yticklabels=labels_order)
plt.title("Confusion Matrix (GTZAN, ResNet50 track-wise)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
cm_path = os.path.join(OUT_DIR, "cm_resnet50_gtzan_trackwise.png")
plt.savefig(cm_path, dpi=150, bbox_inches="tight")
plt.close()

# Quick summary line
acc = (y_true == y_pred).mean()
print(f"\n✅ GTZAN (Track-wise) — Accuracy: {acc:.4f}")
print(f"💾 Saved: {ckpt_path}, {cm_path}, {rep_path}")

# Optionally unfreeze and fine-tune:
# base.trainable = True
# model.compile(optimizer=optimizers.Adam(LR/10), loss="categorical_crossentropy", metrics=["accuracy"])
# model.fit(...)
