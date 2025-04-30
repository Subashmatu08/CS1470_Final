##  1. Instructions
Step 1:
```
git clone https://github.com/facebookresearch/av_hubert.git
cd av_hubert/avhubert
git submodule init
git submodule update
cd ../fairseq
pip install --editable ./
cd ../../
```

Step 2:
```
mv preprocess.py av_hubert/avhubert/
mv inference.py av_hubert/avhubert/
mv eval.py av_hubert/avhubert/
```

Step 3:
```
mkdir -p misc/
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 -O misc/shape_predictor_68_face_landmarks.dat.bz2
bzip2 -d misc/shape_predictor_68_face_landmarks.dat.bz2
wget --content-disposition https://github.com/mpc001/Lipreading_using_Temporal_Convolutional_Networks/raw/master/preprocessing/20words_mean_face.npy -O misc/20words_mean_face.npy
```

Step 4:
```
wget https://dl.fbaipublicfiles.com/avhubert/model/lrs3/vsr/large_lrs3_30h.pt -O misc/model.pt
```

##  2. Usage
### 2.1  Data download and preparation
Download the PolyGlotFake dataset from:
```
https://github.com/tobuta/PolyGlotFake?tab=readme-ov-file#quantitative-comparison
```
Extract the fake English videos into a new folder audio_visual/data/fake and the real English videos into a new folder audio_visual/data/real.

Also create new folders av_hubert/avhubert/features_fake and av_hubert/avhubert/features_real.

Comment out line 121 in av_hubert/fairseq/fairseq/models/__init__.py and line 71 in av_hubert/fairseq/fairseq/tasks/__init__.py.

### 2.2 Pre-processing

For extracting ROIs and wav files, run the following command:
```
cd av_hubert/avhubert
python preprocess.py \
  --input_root ../../audio-visual/data/real
python preprocess.py \
  --input_root ../../audio-visual/data/fake
```

### 2.3 Feature extraction
For extracting audio and video representations, run the following command:
```
cd FACTOR
PYTHONPATH=$PWD python -m av_hubert.avhubert.inference --input_root audio-visual/data/real
PYTHONPATH=$PWD python -m av_hubert.avhubert.inference --input_root audio-visual/data/fake
```

### 2.4 Evaluation
Finally, you can evaluate by running the following command:
```
cd FACTOR
PYTHONPATH=$PWD python av_hubert/avhubert/eval.py
```
### 2.5 Ablation
You can run your ablation.py file to get your ablation results:
```
python av_hubert/avhubert/plot_ablation.py --csv ablation_results.csv
```
Then, plot your results by running:
```
python plot_ablation.py --csv ablation_results.csv
```
### 2.6 MLP
First, run mlp.py like so:
```
cd FACTOR/audio-visual
python mlp.py
```
Next, run your training file:
```
python train_mlp.py \
  --real_dir features_real \
  --fake_dir features_fake \
  --output mlp.pt \
  --epochs 20 \
  --batch_size 32
```
And finally your test file:
```
python test_mlp.py
```
