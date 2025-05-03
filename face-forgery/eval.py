import numpy as np
import faiss
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

parser = argparse.ArgumentParser(description='')
parser.add_argument('--real_root', help='path to the real data (train set)')
parser.add_argument('--test_root', help='path to the test set')
parser.add_argument('--labels_root', help='path to the test labels')
parser.add_argument('--save_roc', default=None, help='optional path to save ROC curve image')
args = parser.parse_args()

# Load datasets
train_set = np.load(args.real_root, allow_pickle=True).astype(np.float32)
test_set = np.load(args.test_root, allow_pickle=True).astype(np.float32)
labels = np.load(args.labels_root, allow_pickle=True)

# Build FAISS index
index = faiss.IndexFlatL2(train_set.shape[1])
index.add(train_set)

# Perform search
k_value = 1
D, _ = index.search(test_set, k_value)
distances = np.sum(D, axis=1)


auc = roc_auc_score(labels, distances)
ap = average_precision_score(labels, distances)
print(f'AP: {ap*100:.2f}%, AUC: {auc*100:.2f}%')


fpr, tpr, thresholds = roc_curve(labels, distances)

# Estimate EER (Equal Error Rate)
fnr = 1 - tpr
eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
print(f'EER: {eer*100:.2f}%')
# for d, label in zip(distances, labels):
#     print(f"Distance: {d:.4f}, Label: {label}")
# print(f"EER threshold: {eer_threshold:.4f}")

plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
plt.plot([0,1], [0,1], 'k--', label='Random Chance')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.grid(True)
plt.legend(loc='lower right')

if args.save_roc:
    plt.savefig(args.save_roc)
    print(f"ROC curve saved to {args.save_roc}")
else:
    plt.show()



# plt.figure(figsize=(8,6))
# plt.hist(distances[labels==1], bins=10, alpha=0.6, label='Real', color='green')
# plt.hist(distances[labels==0], bins=10, alpha=0.6, label='Fake', color='red')
# plt.axvline(eer_threshold, color='blue', linestyle='--', label=f'EER Threshold = {eer_threshold:.2f}')
# plt.title('Distance Distribution')
# plt.xlabel('Distance to Real Reference Set')
# plt.ylabel('Frequency')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()