import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

REAL_DIR = "../../data/alignedFrames/originals-fifty-dataset/original_sequences/youtube/c40/videos"
FAKE_DIR = "../../data/alignedFrames/manipulated-fifty-dataset/manipulated_sequences/FaceSwap/c40/videos" 
OUTPUT_DIR = "ablation_results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_embedding(path):
    arr = np.load(path).astype(np.float32)
    return arr.mean(axis=0) if arr.ndim > 1 else arr

true_distances = []

for fname in tqdm(os.listdir(FAKE_DIR), desc="Evaluating Paired Distances"):
    if "_" not in fname:
        continue

    try:
        base_id, swapped_id = fname.split("_")
    except ValueError:
        continue

    fake_path = os.path.join(FAKE_DIR, fname, "features.npy")
    real_path = os.path.join(REAL_DIR, swapped_id, "features.npy")

    if not (os.path.exists(fake_path) and os.path.exists(real_path)):
        continue

    fake_emb = load_embedding(fake_path)
    real_emb = load_embedding(real_path)

    distance = np.linalg.norm(fake_emb - real_emb)
    true_distances.append(distance)

np.save(os.path.join(OUTPUT_DIR, "true_distances.npy"), np.array(true_distances))

plt.figure(figsize=(8, 4))
plt.hist(true_distances, bins=20, alpha=0.8, color='lightcoral', edgecolor='black')
plt.title("Paired Evaluation: Real vs. Fake (True Identity)")
plt.xlabel("L2 Distance")
plt.ylabel("Number of Videos")
plt.grid(True)
plt.tight_layout()
plt.show()