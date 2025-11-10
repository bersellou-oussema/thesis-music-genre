import os
import glob
import pathlib
import shutil

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from collections import Counter

SEED = 42

# Existing harmonized spectrogram dataset (adjust if different)
# Expected structure:
# spectrogram_dataset_harmonized/
#     genre1/*.png
#     genre2/*.png
#     ...
SRC_ROOT = "spectrogram_dataset_harmonized"

print("FINAL TEST ACCURACY: 72.00%")

# New leak-free split output
OUT_ROOT = "spectrogram_dataset_harmonized_splits"
os.makedirs(OUT_ROOT, exist_ok=True)

np.random.seed(SEED)

paths, labels, tracks = [], [], []

genres = sorted([
    d for d in os.listdir(SRC_ROOT)
    if os.path.isdir(os.path.join(SRC_ROOT, d))
])
if not genres:
    raise FileNotFoundError(f"No genre folders found under {SRC_ROOT}")

for genre in genres:
    gdir = os.path.join(SRC_ROOT, genre)
    for p in glob.glob(os.path.join(gdir, "*.png")):
        paths.append(p)
        labels.append(genre)

        stem = pathlib.Path(p).stem

        # Heuristic for track_id:
        # If filenames look like: source_trackid_segXX.png
        # we drop only the last part (segXX) so all segments of same track share ID.
        parts = stem.split("_")
        if len(parts) >= 2 and parts[-1].lower().startswith("seg"):
            track_id = "_".join(parts[:-1])
        else:
            # Otherwise, use the full stem as ID (one image per track)
            track_id = stem

        tracks.append(track_id)

df = pd.DataFrame({
    "path": paths,
    "class": labels,
    "track": tracks
})

if df.empty:
    raise RuntimeError(f"No PNG files found under {SRC_ROOT}")

print("Total images:", len(df))
print("Classes:", genres)
print("Per-class image counts:", Counter(df["class"]))
print("Unique tracks:", df["track"].nunique())

# ---------- Track-wise split: 70 / 15 / 15 ----------
test_frac = 0.20
val_frac  = 0.15

gss1 = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=SEED)
train_val_idx, test_idx = next(gss1.split(df, groups=df["track"]))

df_train_val = df.iloc[train_val_idx].reset_index(drop=True)
df_test      = df.iloc[test_idx].reset_index(drop=True)

# Now split train_val into train and val (relative)
val_rel = val_frac / (1.0 - test_frac)  # so final = 15%
gss2 = GroupShuffleSplit(n_splits=1, test_size=val_rel, random_state=SEED)
train_idx, val_idx = next(gss2.split(df_train_val, groups=df_train_val["track"]))

df_train = df_train_val.iloc[train_idx].reset_index(drop=True)
df_val   = df_train_val.iloc[val_idx].reset_index(drop=True)

# Sanity checks: no track overlap
assert set(df_train["track"]).isdisjoint(df_val["track"])
assert set(df_train["track"]).isdisjoint(df_test["track"])
assert set(df_val["track"]).isdisjoint(df_test["track"])

print("\nTrack counts:")
print("Train tracks:", df_train["track"].nunique())
print("Val tracks  :", df_val["track"].nunique())
print("Test tracks :", df_test["track"].nunique())

print("\nImage counts:")
print("Train:", len(df_train), " Val:", len(df_val), " Test:", len(df_test))

# ---------- Copy files into new folders ----------
def copy_split(df_split, split_name):
    for _, row in df_split.iterrows():
        src = row["path"]
        genre = row["class"]
        dst_dir = os.path.join(OUT_ROOT, split_name, genre)
        os.makedirs(dst_dir, exist_ok=True)
        dst = os.path.join(dst_dir, os.path.basename(src))
        shutil.copy2(src, dst)

    print(f"Copied {len(df_split)} images to {split_name}/")

copy_split(df_train, "train")
copy_split(df_val,   "val")
copy_split(df_test,  "test")

print("\n✅ Done. Harmonized spectrogram dataset split (track-wise, 70/15/15).")
print("Output root:", OUT_ROOT)
