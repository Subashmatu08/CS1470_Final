import os
import shutil
import random
import numpy as np
import argparse

def average_features(feature_path):
    features = np.load(feature_path)
    if len(features.shape) > 1:
        return np.mean(features, axis=0)
    return features

def main(real_videos_path, fake_videos_path, eval_dataset_path):
e
    real_folder = os.path.join(eval_dataset_path, 'real')
    test_folder = os.path.join(eval_dataset_path, 'test')
    main_folder = os.path.join(eval_dataset_path, 'main')

    os.makedirs(real_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)
    os.makedirs(main_folder, exist_ok=True)

    
    real_folders = [f for f in os.listdir(real_videos_path) if os.path.isdir(os.path.join(real_videos_path, f))]
    fake_folders = [f for f in os.listdir(fake_videos_path) if os.path.isdir(os.path.join(fake_videos_path, f))]

    #shuffle
    random.shuffle(real_folders)
    random.shuffle(fake_folders)

    #split
    total_real = len(real_folders)
    total_fake = len(fake_folders)

    if total_real < 4 or total_fake < 2:
        raise ValueError("Too few samples to split meaningfully.")

    num_real_train = total_real // 2
    num_real_test = total_real // 4
    num_fake_test = min(num_real_test, total_fake)

    train_real_folders = real_folders[:num_real_train]
    test_real_folders = real_folders[num_real_train:num_real_train + num_real_test]
    test_fake_folders = fake_folders[:num_fake_test]

    print(f"Reference real set: {len(train_real_folders)}")
    print(f"Test set: {len(test_real_folders)} real + {len(test_fake_folders)} fake")

   
    for folder in train_real_folders:
        shutil.copytree(os.path.join(real_videos_path, folder), os.path.join(real_folder, folder))
    for folder in test_real_folders:
        shutil.copytree(os.path.join(real_videos_path, folder), os.path.join(test_folder, folder))
    for folder in test_fake_folders:
        shutil.copytree(os.path.join(fake_videos_path, folder), os.path.join(test_folder, folder))

    
    real_features = []
    for folder in train_real_folders:
        feature_path = os.path.join(real_folder, folder, 'features.npy')
        real_features.append(average_features(feature_path))
    np.save(os.path.join(main_folder, 'real_features.npy'), np.vstack(real_features))

    
    test_features = []
    test_labels = []

    for folder in test_real_folders:
        feature_path = os.path.join(test_folder, folder, 'features.npy')
        test_features.append(average_features(feature_path))
        test_labels.append(1)

    for folder in test_fake_folders:
        feature_path = os.path.join(test_folder, folder, 'features.npy')
        test_features.append(average_features(feature_path))
        test_labels.append(0)

    np.save(os.path.join(main_folder, 'test_features.npy'), np.vstack(test_features))
    np.save(os.path.join(main_folder, 'labels.npy'), np.array(test_labels))

    print("\n Dataset preparation complete!")
    print(f"- real/: {len(train_real_folders)} reference real folders")
    print(f"- test/: {len(test_real_folders)} real + {len(test_fake_folders)} fake")
    print(f"- main/: real_features.npy, test_features.npy, labels.npy")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--real_input', type=str, required=True, help='Path to real videos folder')
    parser.add_argument('--fake_input', type=str, required=True, help='Path to fake videos folder')
    parser.add_argument('--output', type=str, required=True, help='Path to output eval_dataset folder')
    args = parser.parse_args()

    main(args.real_input, args.fake_input, args.output)