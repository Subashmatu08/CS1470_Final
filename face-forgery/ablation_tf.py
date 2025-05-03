import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random
from sklearn.metrics import roc_auc_score, average_precision_score
import argparse

parser = argparse.ArgumentParser(description="Ablation study for reference set size")
parser.add_argument('--real_root', required=True, help='Path to the real video folders (each contains features.npy)')
parser.add_argument('--test_root', required=True, help='Path to the test feature .npy file')
parser.add_argument('--labels_root', required=True, help='Path to the labels .npy file')
parser.add_argument('--ref_sizes', default="25,12,8,4,2,1", help='Comma-separated list of reference sizes (e.g. 25,12,8,4,2,1)')
args = parser.parse_args()

real_folder = args.real_root
test_features_path = args.test_root
labels_path = args.labels_root
reference_sizes = [int(s.strip()) for s in args.ref_sizes.split(',')]

test_set = np.load(test_features_path).astype(np.float32)
labels = np.load(labels_path)
real_video_folders = [os.path.join(real_folder, f) for f in os.listdir(real_folder) if os.path.isdir(os.path.join(real_folder, f))]

auc_results = []

for ref_size in reference_sizes:
    sampled_folders = random.sample(real_video_folders, ref_size)

    ref_embeddings = []
    for folder in sampled_folders:
        path = os.path.join(folder, 'features.npy')
        features = np.load(path).astype(np.float32)
        if len(features.shape) > 1:
            features = features.mean(axis=0)
        ref_embeddings.append(features)

    ref_matrix = np.stack(ref_embeddings)

    train_set = tf.convert_to_tensor(ref_matrix, dtype=tf.float32)
    test_tensor = tf.convert_to_tensor(test_set, dtype=tf.float32)

    train_exp = tf.expand_dims(train_set, axis=0)
    test_exp = tf.expand_dims(test_tensor, axis=1)

    dist_matrix = tf.reduce_sum(tf.square(test_exp - train_exp), axis=2)
    distances = tf.reduce_min(dist_matrix, axis=1).numpy()

    auc = roc_auc_score(labels, distances)
    ap = average_precision_score(labels, distances)
    auc_results.append((ref_size, auc, ap))
    print(f"Reference size: {ref_size}, AUC: {auc:.4f}, AP: {ap:.4f}")

#plot
sizes, aucs, aps = zip(*auc_results)
colors = plt.cm.viridis(np.linspace(0, 1, len(sizes)))
labels_legend = [f"{s} {'Identities' if s > 1 else 'Identity'}" for s in sizes]

plt.figure(figsize=(10, 6))
bars = plt.bar(labels_legend, [a * 100 for a in aucs], color=colors)

# for bar, label in zip(bars, labels_legend):
#     bar.set_label(label)

for i, bar in enumerate(bars):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f"ROC-AUC: {aucs[i]*100:.1f}%\nAP: {aps[i]*100:.1f}%",
             ha='center', va='bottom', fontsize=9)

# plt.legend(title='Reference Set Size', loc='lower right')
plt.title('Ablation Study: ROC-AUC vs Number of Reference Identities')
plt.xlabel('Number of Reference Identities')
plt.ylabel('ROC-AUC (%)')
plt.ylim(0, 100)
plt.grid(axis='y')
plt.tight_layout()
plt.show()