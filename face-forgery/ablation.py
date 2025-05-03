import os, random, argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score
import faiss                                   

parser = argparse.ArgumentParser(description="Ablation study with FAISS")
parser.add_argument('--real_root',  required=True, help='Directory with real-video sub-folders')
parser.add_argument('--test_root',  required=True, help='Path to test-set *.npy file')
parser.add_argument('--labels_root',required=True, help='Path to label *.npy file')
parser.add_argument('--ref_sizes',  default="25,12,8,4,2,1",
                    help='Comma-separated list of reference-set sizes')
parser.add_argument('--seed', type=int, default=42, help='RNG seed for reproducibility')
args = parser.parse_args()

np.random.seed(args.seed)
random.seed(args.seed)

test_set = np.load(args.test_root).astype(np.float32)          
labels   = np.load(args.labels_root)

if len(test_set) != len(labels):
    raise ValueError(f"Size mismatch: {len(test_set)} test vectors vs {len(labels)} labels")

real_folders = [os.path.join(args.real_root, f)
                for f in os.listdir(args.real_root)
                if os.path.isdir(os.path.join(args.real_root, f))]

if len(real_folders) == 0:
    raise RuntimeError(f"No sub-folders found in {args.real_root}")

ref_sizes = [int(s.strip()) for s in args.ref_sizes.split(',')]
if max(ref_sizes) > len(real_folders):
    raise ValueError(f"Requested ref_size {max(ref_sizes)} exceeds "
                     f"available folders ({len(real_folders)})")

auc_results = [] 

for ref_size in ref_sizes:
    chosen = random.sample(real_folders, ref_size)
    refs   = []
    for folder in chosen:
        feat_path = os.path.join(folder, 'features.npy')
        if not os.path.isfile(feat_path):
            raise FileNotFoundError(f"{feat_path} missing")
        f = np.load(feat_path).astype(np.float32)
        if f.ndim > 1:
            f = f.mean(axis=0)              
        refs.append(f)

    ref_matrix = np.stack(refs)                

    d = ref_matrix.shape[1]
    index = faiss.IndexFlatL2(d)               
    index.add(ref_matrix)                      
    D, _ = index.search(test_set, k=1)         
    distances = D[:, 0]                        

    auc = roc_auc_score(labels, distances)
    ap  = average_precision_score(labels, distances)

    auc_results.append((ref_size, auc, ap))
    print(f"Ref={ref_size:2d} | AUC={auc*100:6.2f}% | AP={ap*100:6.2f}%")

sizes, aucs, aps = zip(*auc_results)
colors = plt.cm.viridis(np.linspace(0, 1, len(sizes)))
xticklabels = [f"{s} video{'s' if s>1 else ''}" for s in sizes]

plt.figure(figsize=(10, 6))
bars = plt.bar(xticklabels, [a*100 for a in aucs], color=colors)

for i, bar in enumerate(bars):
    plt.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 1,
             f"AUC {aucs[i]*100:.1f}%\nAP {aps[i]*100:.1f}%",
             ha='center', va='bottom', fontsize=9)

plt.title("Ablation Study: ROC-AUC vs Number of Reference Videos")
plt.xlabel("Reference-set size")
plt.ylabel("ROC-AUC (%)")
plt.ylim(0, 100)
plt.grid(axis='y')
plt.tight_layout()
plt.show()