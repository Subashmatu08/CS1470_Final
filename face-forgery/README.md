# FaceSwapping Evaluation and Ablation Framework

## 1. Installation

Activate your virtual environment:

```bash
source ../venv/bin/activate
```

Clone the FaceX-Zoo repository:

```bash
git clone https://github.com/JDAI-CV/FaceX-Zoo.git
```

Move the required scripts:

```bash
mv extract_frames.py FaceX-Zoo/face_sdk/
mv detect_and_align.py FaceX-Zoo/face_sdk/
mv extract_feature.py FaceX-Zoo/test_protocol/
mv eval.py FaceX-Zoo/test_protocol/
mv eval_tf.py FaceX-Zoo/test_protocol/
mv ablation.py FaceX-Zoo/test_protocol/
mv ablation_tf.py FaceX-Zoo/test_protocol/
mv identity_eval.py FaceX-Zoo/test_protocol/
mv identity_ablation.py FaceX-Zoo/test_protocol/
mv backbone_conf.yaml FaceX-Zoo/test_protocol/
```

Download the checkpoint (`Epoch_17.pt`) from [this link](https://drive.google.com/drive/folders/1h_meJetsaVUm-37Wqo-o3ed9lyWcS8-B) and place it in:

```
FaceX-Zoo/test_protocol/
```

---

## 2. Dataset Preparation

Our dataset follows this structure:

```
.mp4 videos
  → [frames/]
  → [alignedFrames/]
  → [features.npy]
```

### 2.1 Run FaceForensics.py

This script splits the dataset and creates the initial directory layout:
The second two make a smaller subset of the data

```bash
python download.py ./locaton --server EU2 --dataset FaceSwap –dataset original --compression c40

python download.py ./location --server EU2 --dataset original --compression c40
# you can add a flag to make a smaller portion for example --num_videos 50
# change the location to align with our /data/mp4 or manually place them such that we have it like /data/mp4/original and /data/mp4/manipulated
```

### 2.2 Extract Frames

```bash
cd FaceX-Zoo/face_sdk/
python extract_frames.py --input_root PATH_TO_MP4 --out_root frames --num_frames 5
```

### 2.3 Detect and Align

```bash
cd FaceX-Zoo/face_sdk/
python detect_and_align.py --input_root frames --out_root alignedFrames
```

### 2.4 Feature Extraction

```bash
cd FaceX-Zoo/test_protocol/
python extract_feature.py --input_root alignedFrames
```

Each video folder will now contain a `features.npy` file.

---

## 3. Evaluation Dataset Generation (Non-Identity Based)

Use the following to split real/fake videos and generate `.npy` files for evaluation:

```bash
python eval_db.py \
  --real_input path_to_real \
  --fake_input path_to_fake \
  --output path_to_eval_dataset
```

This creates:

```
eval_dataset/
├── real/            # Reference real videos
├── test/            # Real + fake videos for testing
└── main/
    ├── real_features.npy
    ├── test_features.npy
    └── labels.npy
```

---

## 4. Evaluation

### 4.1 Evaluation (Non-Identity)

```bash
cd FaceX-Zoo/test_protocol/
python eval_tf.py \
  --real_root eval_dataset/main/real_features.npy \
  --test_root eval_dataset/main/test_features.npy \
  --labels_root eval_dataset/main/labels.npy
```

### 4.2 Identity-Based Evaluation

```bash
cd FaceX-Zoo/test_protocol/
python eval.py
```

This uses the aligned dataset folders directly:

```
alignedFrames/originals-fifty-dataset/
alignedFrames/manipulated-fifty-dataset/
```

---

## 5. Ablation Studies

### 5.1 Ablation

```bash
cd FaceX-Zoo/test_protocol/
python ablation_tf.py \
  --real_root eval_dataset/real \
  --test_root eval_dataset/main/test_features.npy \
  --labels_root eval_dataset/main/labels.npy \
  --ref_sizes 25,12,8,4,2,1
```

### 5.2 Identity-Based Ablation

```bash
cd FaceX-Zoo/test_protocol/
python ablation.py
```

Plots and saves histograms of paired vs unpaired distances.

## 7. Notes

- For identity-based evaluation, ensure fake folder names follow `XXX_YYY` where `YYY` is the identity being impersonated.
- All `.npy` feature arrays are mean-aggregated from the frame-level features.
- Default results (in our case histograms) are saved under `test_protocol/ablation_results/`.
