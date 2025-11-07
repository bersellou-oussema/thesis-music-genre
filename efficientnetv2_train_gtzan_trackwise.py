# efficientnetv2_train_gtzan_trackwise.py
# Train EfficientNetV2B0 as a CNN on GTZAN spectrogram images with a track-wise split.
# Saves: model (.keras), classification report (.txt), confusion matrix (.png)

import os
import re
import json
import math
import random
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2B0, preprocess_input
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# ----------------------------
# Paths & constants (edit here)
# ----------------------------
DATA_DIR = "spectrogram_dataset_windowed/gtzan"  # <-- adjust if your path differs
OUT_DIR  = "models"
IMG_SIZE = (224, 224)
BATCH    = 32
EPOCHS   = 15
SEED     = 42

os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------
# Utilities: robust track-id extraction from filename
# ---------------------------------------------------
def extract_track_id(path: Path) -> str:
    """
    Try to recover the original track ID from a spectrogram slice filename.
    This function is deliberately tolerant to various naming patterns, e.g.:
      - blues.00001_0003.png
      - pop-00045-seg-07.png
      - jazz_00012_win03.png
      - rock.00001.png (no windowing)
    Strategy:
      1) take stem
      2) split on common window markers
      3) keep the first part and remove trailing slice indices like '-07' '_07'
    """
    stem = path.stem

    # split at common window markers
    for token in ["_chunk", "_win", "-win", "_seg", "-seg", "__", "_slice"]:
        if token in stem:
            stem = stem.split(token)[0]

    # remove trailing separators + digits (e.g., "-07", "_12")
    stem = re.sub(r"[-_]\d+$", "", stem)

    # if pattern like genre.00001[_anything], keep genre.00001
    m = re.match(r"([a-zA-Z]+[._-]\d+)", stem)
    if m:
        return m.group(1)

    return stem

# -------------------------------------------
# Build a dataframe of all images & track ids
# -------------------------------------------
def build_dataframe(data_dir: str) -> pd.DataFrame:
    p = Path(data_dir)
    rows = []
    classes = sorted([d.name for d in p.iterdir() if d.is_dir()])
    for cls in classes:
        for img in (p / cls).glob("*.png"):
            tid = extract_track_id(img)
            rows.append({"path": str(img), "class": cls, "track": tid})
    df = pd.DataFrame(rows)
    return df, classes

# -----------------------------
# Make a strict track-wise split
# -----------------------------
def trackwise_split(df: pd.DataFrame, test_size=0.20, val_size=0.20, seed=SEED):
    # one row per track with its label (assume all slices from a track have same class)
    tracks = df.groupby("track")["class"].agg(lambda x: Counter(x).most_common(1)[0][0]).reset_index()

    # First: test split (stratified by class at track level)
    tr_tracks, te_tracks = train_test_split(
        tracks, test_size=test_size, random_state=seed, stratify=tracks["class"]
    )

    # Second: validation split from the remaining training tracks
    tr2, va_tracks = train_test_split(
        tr_tracks, test_size=val_size, random_state=seed, stratify=tr_tracks["class"]
    )

    # Map back to the slices (images)
    df_train = df[df["track"].isin(tr2["track"])]
    df_val   = df[df["track"].isin(va_tracks["track"])]
    df_test  = df[df["track"].isin(te_tracks["track"])]

    return df_train.reset_index(drop=True), df_val.reset_index(drop=True), df_test.reset_index(drop=True)

# -------------------------
# Image generators (Keras)
# -------------------------
def make_gens(df_train, df_val, df_test, classes):
    # Use preprocess_input from EfficientNetV2
    train_aug = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=5,
        width_shift_range=0.05,
        height_shift_range=0.05,
        horizontal_flip=False,  # spectrograms: flipping may change temporal order
        fill_mode="nearest",
    )
    plain = ImageDataGenerator(preprocessing_function=preprocess_input)

    def flow(df, shuffle):
        return train_aug.flow_from_dataframe(
            df,
            x_col="path",
            y_col="class",
            classes=classes,
            target_size=IMG_SIZE,
            color_mode="rgb",
            class_mode="categorical",
            batch_size=BATCH,
            shuffle=shuffle,
            seed=SEED
        )

    train_gen = flow(df_train, shuffle=True)
    val_gen   = plain.flow_from_dataframe(
        df_val, x_col="path", y_col="class", classes=classes,
        target_size=IMG_SIZE, color_mode="rgb",
        class_mode="categorical", batch_size=BATCH, shuffle=False, seed=SEED
    )
    test_gen  = plain.flow_from_dataframe(
        df_test, x_col="path", y_col="class", classes=classes,
        target_size=IMG_SIZE, color_mode="rgb",
        class_mode="categorical", batch_size=BATCH, shuffle=False, seed=SEED
    )
    return train_gen, val_gen, test_gen

# -------------
# Build model
# -------------
def build_model(num_classes: int):
    base = EfficientNetV2B0(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
        pooling="avg",
    )
    x = layers.Dropout(0.25)(base.output)
    out = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(base.input, out)
    opt = optimizers.Adam(learning_rate=1e-3)
    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])
    return model

# -------------------------
# Confusion matrix plotting
# -------------------------
def save_confusion(y_true, y_pred, labels, out_png, title):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

# ============
# Main script
# ============
if __name__ == "__main__":
    # Diagnostics
    print(">>> Python:", tf.sysconfig.get_build_info().get("cpu_compiler", "OK"))
    print(">>> TensorFlow:", tf.__version__)
    try:
        build = tf.sysconfig.get_build_info()
        print(">>> TF build:", {k: build.get(k) for k in ["cuda_version", "cudnn_version"]})
    except Exception:
        pass

    # Build dataframe
    df, classes = build_dataframe(DATA_DIR)
    classes = sorted(classes)
    print("\nTotal spectrogram slices:", len(df))
    print("Classes:", classes)

    per_class_counts = df.groupby("class")["track"].nunique()
    print("Unique tracks per class:", dict(per_class_counts))

    # Track-wise split
    df_train, df_val, df_test = trackwise_split(df, test_size=0.20, val_size=0.20, seed=SEED)

    print("\n---- DATAFRAME SIZES ----")
    print("df_train:", len(df_train), " df_val:", len(df_val), " df_test:", len(df_test))
    print("Tracks (train/val/test):",
          df_train['track'].nunique(), df_val['track'].nunique(), df_test['track'].nunique())

    # Generators
    train_gen, val_gen, test_gen = make_gens(df_train, df_val, df_test, classes)

    print("\n---- GENERATOR SIZES ----")
    print("train_gen.samples:", train_gen.samples)
    print("val_gen.samples  :", val_gen.samples)
    print("test_gen.samples :", test_gen.samples)

    print("\n---- CLASS INDICES ----")
    print(train_gen.class_indices)

    # Small sanity check: one batch
    xb, yb = next(train_gen)
    print("\n---- ONE BATCH SANITY ----")
    print("Batch shapes (x, y):", xb.shape, yb.shape)
    print("One-hot sum (yb[0]):", float(yb[0].sum()))
    print("Argmax label example:", int(yb[0].argmax()))

    # Model
    num_classes = len(classes)
    model = build_model(num_classes)
    model.summary()

    ckpt_path = os.path.join(OUT_DIR, "efficientnetv2_gtzan_trackwise_best.keras")
    cbs = [
        callbacks.ModelCheckpoint(ckpt_path, monitor="val_accuracy",
                                  save_best_only=True, save_weights_only=False),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1, min_lr=1e-5),
        callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True, verbose=1),
        callbacks.CSVLogger(os.path.join(OUT_DIR, "efficientnetv2_gtzan_trackwise_log.csv"))
    ]

    steps_tr = math.ceil(train_gen.samples / BATCH)
    steps_va = math.ceil(val_gen.samples   / BATCH)

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        steps_per_epoch=steps_tr,
        validation_steps=steps_va,
        callbacks=cbs,
        verbose=1
    )

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_steps = math.ceil(test_gen.samples / BATCH)
    probs = model.predict(test_gen, steps=test_steps, verbose=1)
    y_pred_idx = probs.argmax(axis=1)
    y_true_idx = test_gen.classes

    idx_to_class = {v: k for k, v in test_gen.class_indices.items()}
    y_pred = [idx_to_class[i] for i in y_pred_idx]
    y_true = [idx_to_class[i] for i in y_true_idx]

    # Classification report
    rep = classification_report(y_true, y_pred, labels=classes, digits=4, output_dict=False)
    print("\n🧪 Classification Report (EffNetV2B0, GTZAN, track-wise):\n")
    print(rep)

    # Save report
    report_path = os.path.join(OUT_DIR, "report_efficientnetv2_gtzan_trackwise.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Classification Report (EfficientNetV2B0, GTZAN, track-wise)\n\n")
        f.write(rep)
    print(f"\n💾 Saved report to {report_path}")

    # Confusion matrix
    cm_path = os.path.join(OUT_DIR, "cm_efficientnetv2_gtzan_trackwise.png")
    save_confusion(y_true, y_pred, classes, cm_path,
                   "Confusion Matrix (EfficientNetV2B0, GTZAN, Track-wise)")
    print(f"💾 Saved confusion matrix to {cm_path}")

    # Final metric line
    rep_dict = classification_report(y_true, y_pred, labels=classes, digits=4, output_dict=True)
    acc  = rep_dict["accuracy"]
    w_p  = rep_dict["weighted avg"]["precision"]
    w_r  = rep_dict["weighted avg"]["recall"]
    w_f1 = rep_dict["weighted avg"]["f1-score"]

    print(f"\n✅ EfficientNetV2B0 (GTZAN Track-wise) — "
          f"Accuracy: {acc:.4f} | Weighted Precision: {w_p:.4f} | "
          f"Weighted Recall: {w_r:.4f} | Weighted F1: {w_f1:.4f}")
    print(f"💾 Best model saved to: {ckpt_path}")
