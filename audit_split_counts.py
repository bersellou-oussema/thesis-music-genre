import os, glob
from pathlib import Path
import numpy as np

def count_by_class(root):
    if not Path(root).exists():
        print(f"[!] Missing: {root}")
        return {}
    classes = sorted([d.name for d in Path(root).iterdir() if d.is_dir()])
    counts = {}
    for c in classes:
        n = len(glob.glob(os.path.join(root, c, "*.npy")))
        counts[c] = n
    return counts

def sample_check(root, n=3):
    files = glob.glob(os.path.join(root, "*", "*.npy"))
    files = files[:n]
    for f in files:
        x = np.load(f)
        print(f"  {Path(f).parent.name}/{Path(f).name}  shape={x.shape}  min={x.min():.2f}  max={x.max():.2f}")

def main():
    train = "spectrogram_dataset_harmonized/gtzan_split/train"
    val   = "spectrogram_dataset_harmonized/gtzan_split/val"
    test  = "spectrogram_dataset_harmonized/gtzan_split/test"

    ct_train = count_by_class(train)
    ct_val   = count_by_class(val)
    ct_test  = count_by_class(test)

    # Classes
    train_classes = sorted(ct_train.keys())
    val_classes   = sorted(ct_val.keys())
    test_classes  = sorted(ct_test.keys())

    print("\nClasses (train):", train_classes)
    print("Classes (val):  ", val_classes)
    print("Classes (test): ", test_classes)
    print("\n#classes:", len(train_classes))

    # Mismatches?
    if train_classes != val_classes or train_classes != test_classes:
        print("\n[!] Class name mismatch between splits — fix your folders before training!")
        return

    # Counts per class
    print("\nCounts per class (train):")
    for k in train_classes: print(f"  {k:10s}: {ct_train.get(k,0)}")
    print("\nCounts per class (val):")
    for k in val_classes: print(f"  {k:10s}: {ct_val.get(k,0)}")
    print("\nCounts per class (test):")
    for k in test_classes: print(f"  {k:10s}: {ct_test.get(k,0)}")

    # Quick sample checks
    print("\nSample files from TRAIN:")
    sample_check(train, n=3)
    print("\nSample files from VAL:")
    sample_check(val, n=3)

if __name__ == "__main__":
    main()
