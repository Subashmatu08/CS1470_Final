import os
import numpy as np
import random
from tqdm import tqdm
import matplotlib.pyplot as plt

REAL_DIR = "../../data/alignedFrames/originals-fifty-dataset/original_sequences/youtube/c40/videos"
FAKE_DIR = "../../data/alignedFrames/manipulated-fifty-dataset/manipulated_sequences/FaceSwap/c40/videos" 
OUTPUT_DIR = "ablation_results"
REFERENCE_SIZES = [5, 10, 15, 20]
SEED = 42

random.seed(SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_embedding(path):
    arr = np.load(path).astype(np.float32)
    return arr.mean(axis=0) if arr.ndim > 1 else arr

all_real_ids = {
    name for name in os.listdir(REAL_DIR)
    if os.path.isdir(os.path.join(REAL_DIR, name))
}

fake_paths = []
for fname in os.listdir(FAKE_DIR):
    if "_" in fname and os.path.isdir(os.path.join(FAKE_DIR, fname)):
        try:
            base_id, swapped_id = fname.split("_")
            if swapped_id in all_real_ids:
                fake_paths.append((fname, swapped_id))
        except ValueError:
            continue

distance_distributions = {k: [] for k in REFERENCE_SIZES}

for k in REFERENCE_SIZES:
    print(f"\n🔍 Running ablation for reference size = {k}")
    for fake_folder, identity in tqdm(fake_paths, desc=f"Ref Size {k}"):
        fake_path = os.path.join(FAKE_DIR, fake_folder, "features.npy")
        if not os.path.exists(fake_path):
            continue

        fake_emb = load_embedding(fake_path)

        unrelated_ids = list(all_real_ids - {identity})
        if len(unrelated_ids) < k:
            continue 

        sampled_ids = random.sample(unrelated_ids, k)

        ref_embs = []
        for rid in sampled_ids:
            ref_path = os.path.join(REAL_DIR, rid, "features.npy")
            if os.path.exists(ref_path):
                ref_embs.append(load_embedding(ref_path))

        if not ref_embs:
            continue

        ref_matrix = np.stack(ref_embs)
        dists = np.linalg.norm(ref_matrix - fake_emb, axis=1)
        min_dist = np.min(dists)

        distance_distributions[k].append(min_dist)

true_distances_path = os.path.join(OUTPUT_DIR, "true_distances.npy")
if os.path.exists(true_distances_path):
    true_distances = np.load(true_distances_path)
else:
    true_distances = None

if true_distances is not None:
    fig, axes = plt.subplots(len(REFERENCE_SIZES) + 1, 1, figsize=(8, 4 * (len(REFERENCE_SIZES) + 1)), sharex=True)
    baseline_ax = axes[0]
    baseline_ax.hist(true_distances, bins=20, alpha=0.8, color='lightcoral', edgecolor='black')
    baseline_ax.set_title("True Distances (Real vs. Paired Fake)")
    baseline_ax.set_ylabel("Number of Videos")
    baseline_ax.grid(True)
    axes_to_plot = axes[1:]
else:
    fig, axes = plt.subplots(len(REFERENCE_SIZES), 1, figsize=(8, 4 * len(REFERENCE_SIZES)), sharex=True)
    axes_to_plot = axes

for ax, k in zip(axes_to_plot, REFERENCE_SIZES):
    data = distance_distributions[k]
    ax.hist(data, bins=20, alpha=0.8, color='skyblue', edgecolor='black')
    ax.set_title(f"Distance Distribution – k={k} Unpaired References")
    ax.set_ylabel("Number of Videos")
    ax.grid(True)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}'))

axes[-1].set_xlabel("Minimum L2 Distance to Reference Set")
plt.suptitle("Unpaired Reference Ablation Study", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(os.path.join(OUTPUT_DIR, "ablation_study_plot.png"))
plt.show()

for k in REFERENCE_SIZES:
    np.save(os.path.join(OUTPUT_DIR, f"distances_k{k}.npy"), np.array(distance_distributions[k]))