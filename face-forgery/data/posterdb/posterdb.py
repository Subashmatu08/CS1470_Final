import os
import numpy as np
import random


real_root = './real'
test_real_path = './test/real/033/features.npy'
test_fake_path = './test/fake/033_097/features.npy'
output_dir = './main' 


os.makedirs(output_dir, exist_ok=True)

def load_feature(path):
    feat = np.load(path).astype(np.float32)
    return feat.mean(axis=0) if feat.ndim > 1 else feat


real_embeddings = []
for subfolder in os.listdir(real_root):
    folder_path = os.path.join(real_root, subfolder)
    feature_path = os.path.join(folder_path, 'features.npy')
    if os.path.isfile(feature_path):
        emb = load_feature(feature_path)
        real_embeddings.append(emb)

real_embeddings = np.stack(real_embeddings)

real_test = load_feature(test_real_path)
fake_test = load_feature(test_fake_path)
test_set = np.stack([real_test, fake_test])
labels = np.array([0, 1], dtype=np.int32)

np.save(os.path.join(output_dir, 'real.npy'), real_embeddings)
np.save(os.path.join(output_dir, 'test.npy'), test_set)
np.save(os.path.join(output_dir, 'labels.npy'), labels)

print("done")
print(f"  real.npy   shape: {real_embeddings.shape}")
print(f"  test.npy   shape: {test_set.shape}")
print(f"  labels.npy values: {labels.tolist()}")