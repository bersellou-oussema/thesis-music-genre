import os
import random
import json
from collections import defaultdict

random.seed(42)  # for reproducibility

DATASET_ROOT = "data_wav"  # your GTZAN wav folder
OUTPUT_SPLIT_DIR = "splits_gtzan"
os.makedirs(OUTPUT_SPLIT_DIR, exist_ok=True)

# 1. Collect tracks grouped by genre
genre_to_tracks = defaultdict(list)

for genre in os.listdir(DATASET_ROOT):
    genre_path = os.path.join(DATASET_ROOT, genre)
    if not os.path.isdir(genre_path):
        continue
    for fname in os.listdir(genre_path):
        if fname.lower().endswith((".wav", ".au", ".mp3")):
            track_path = os.path.join(genre_path, fname)
            # track_id = filename without extension (ensures windows/spectrograms map to same track)
            track_id = f"{genre}/{os.path.splitext(fname)[0]}"
            genre_to_tracks[genre].append((track_id, track_path))

splits = {"train": [], "val": [], "test": []}

# 2. For each genre, split by TRACKS into train/val/test
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

for genre, tracks in genre_to_tracks.items():
    random.shuffle(tracks)
    n = len(tracks)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_tracks = tracks[:n_train]
    val_tracks = tracks[n_train:n_train + n_val]
    test_tracks = tracks[n_train + n_val:]

    splits["train"].extend(train_tracks)
    splits["val"].extend(val_tracks)
    splits["test"].extend(test_tracks)

# 3. Save as JSON + plain txt for convenience
split_json = {
    split: [
        {"track_id": track_id, "path": path}
        for (track_id, path) in items
    ]
    for split, items in splits.items()
}

with open(os.path.join(OUTPUT_SPLIT_DIR, "gtzan_splits.json"), "w") as f:
    json.dump(split_json, f, indent=2)

for split in ["train", "val", "test"]:
    with open(os.path.join(OUTPUT_SPLIT_DIR, f"{split}.txt"), "w") as f:
        for track_id, path in splits[split]:
            f.write(f"{track_id}|{path}\n")

print("Saved splits in", OUTPUT_SPLIT_DIR)
