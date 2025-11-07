import os, shutil, glob, random
from pathlib import Path

# -------- settings --------
RATIO = (0.8, 0.1, 0.1)  # train/val/test when auto-splitting
SEED = 42
# --------------------------

def track_id_from_npy(path):
    """Return a stable track id without window/hash.
    Example: blues/blues.00000__abc123.npy -> 'blues/blues.00000' """
    p = Path(path)
    genre = p.parent.name
    base = p.stem
    core = base.split("__")[0]
    return f"{genre}/{core}"

def load_lists_if_exist(list_dir):
    lists = {}
    for name in ["train","val","test"]:
        f = Path(list_dir) / f"{name}.txt"
        if f.exists():
            with open(f, "r", encoding="utf-8") as fh:
                items = [line.strip() for line in fh if line.strip()]
            lists[name] = set(items)
    return lists if len(lists)==3 else None

def collect_all_track_ids(root):
    npys = glob.glob(os.path.join(root, "**", "*.npy"), recursive=True)
    track_ids = {}
    for f in npys:
        tid = track_id_from_npy(f)
        track_ids.setdefault(tid, []).append(f)
    return track_ids

def auto_split_by_track(track_ids):
    by_genre = {}
    for tid in track_ids.keys():
        genre = tid.split("/")[0]
        by_genre.setdefault(genre, []).append(tid)

    random.Random(SEED).seed(SEED)
    train, val, test = set(), set(), set()
    for g, ids in by_genre.items():
        ids = sorted(ids)
        random.shuffle(ids)
        n = len(ids)
        n_tr = int(RATIO[0]*n)
        n_v  = int(RATIO[1]*n)
        tr = ids[:n_tr]
        va = ids[n_tr:n_tr+n_v]
        te = ids[n_tr+n_v:]
        train.update(tr); val.update(va); test.update(te)
    return {"train": train, "val": val, "test": test}

def copy_split(track_ids, split_dict, out_root):
    for split_name, tids in split_dict.items():
        for tid in tids:
            files = track_ids[tid]
            genre = tid.split("/")[0]
            dst_dir = Path(out_root)/split_name/genre
            dst_dir.mkdir(parents=True, exist_ok=True)
            for src in files:
                dst = dst_dir/Path(src).name
                if not dst.exists():
                    shutil.copy2(src, dst)

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_root", required=True, help="harmonized npy root (e.g., spectrogram_dataset_harmonized/gtzan)")
    ap.add_argument("--out_root", required=True, help="destination with train/val/test subfolders")
    ap.add_argument("--lists_dir", default=None, help="folder containing train.txt/val.txt/test.txt (optional)")
    args = ap.parse_args()

    Path(args.out_root).mkdir(parents=True, exist_ok=True)
    track_ids = collect_all_track_ids(args.in_root)

    split_lists = load_lists_if_exist(args.lists_dir) if args.lists_dir else None
    if split_lists is None:
        print("No split lists found → auto-splitting 80/10/10 by track.")
        split_lists = auto_split_by_track(track_ids)

    copy_split(track_ids, split_lists, args.out_root)
    print("✅ Done. Data copied to:", args.out_root)

if __name__ == "__main__":
    main()
