import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Constants
SR = 22050
SEGMENT_DURATION = 1.5  # in seconds
HOP_DURATION = 0.75     # 50% overlap
N_MELS = 128
IMG_SIZE = (2.24, 2.24)  # for 224x224 pixels

# Output directory
OUTPUT_BASE = "spectrogram_dataset_windowed"

def trim_and_normalize(y):
    y_trimmed, _ = librosa.effects.trim(y)
    y_norm = librosa.util.normalize(y_trimmed)
    return y_norm

def save_mel_spectrogram(y, sr, output_path):
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    plt.figure(figsize=IMG_SIZE)
    librosa.display.specshow(mel_db, sr=sr)
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def process_audio_file(audio_path, output_dir, dataset_name, genre, file_id):
    try:
        y, sr = librosa.load(audio_path, sr=SR)
        y = trim_and_normalize(y)

        segment_samples = int(SEGMENT_DURATION * sr)
        hop_samples = int(HOP_DURATION * sr)

        for i in range(0, len(y) - segment_samples, hop_samples):
            segment = y[i:i + segment_samples]
            if len(segment) < segment_samples:
                continue

            segment_id = f"{file_id}_{i}"
            genre_dir = os.path.join(output_dir, dataset_name, genre)
            os.makedirs(genre_dir, exist_ok=True)
            output_path = os.path.join(genre_dir, f"{segment_id}.png")
            save_mel_spectrogram(segment, sr, output_path)

    except Exception as e:
        print(f"⚠️ Failed to process {audio_path}: {e}")

def process_dataset(dataset_path, dataset_name, output_dir):
    for genre in os.listdir(dataset_path):
        genre_path = os.path.join(dataset_path, genre)
        if not os.path.isdir(genre_path):
            continue
        print(f"\n🎧 Processing genre: {genre} ({dataset_name})")

        for filename in tqdm(os.listdir(genre_path)):
            if not filename.lower().endswith(".wav"):
                continue
            audio_path = os.path.join(genre_path, filename)
            file_id = os.path.splitext(filename)[0]
            process_audio_file(audio_path, output_dir, dataset_name, genre, file_id)

if __name__ == "__main__":
    print("🚀 Generating Mel spectrograms with trim+normalize+windowing...")
    process_dataset("data_wav", "gtzan", OUTPUT_BASE)
    process_dataset("fma_small", "fma", OUTPUT_BASE)
    print("\n✅ All spectrograms saved to:", OUTPUT_BASE)
