#!/usr/bin/env python
"""
TensorFlow / NumPy inference pipeline for AV-HuBERT features.

1. If `avhubert.onnx` is missing it will:
      a)  load the .pt checkpoint with PyTorch
      b)  export to ONNX
      c)  save `avhubert.onnx`
2. Subsequent runs load ONNX with onnxruntime – no torch import!
"""

import os, argparse, pathlib, tempfile, subprocess, warnings, sys
import numpy as np
import librosa
from python_speech_features import logfbank
from tqdm import tqdm

import torch
import torch.nn as nn
import os
import torch

from av_hubert.avhubert.hubert import AVHubertModel

from av_hubert.avhubert.hubert import AVHubertModel
from torch import nn
import numpy as np
import torch
from tqdm import tqdm
import warnings

def subsample_frames(feat, num_frames=32):
    idx = np.linspace(0, feat.shape[0] - 1, num_frames).astype(int)
    return feat[idx]

class AVHubertOnnxWrapper(nn.Module):
    def __init__(self, hubert: AVHubertModel):
        super().__init__()
        self.hubert = hubert    

    def forward(self, audio: torch.Tensor, video: torch.Tensor):
        feat_a = self.hubert.forward_features(audio, modality="audio")
        feat_v = self.hubert.forward_features(video, modality="video")
        return feat_a, feat_v
def _find_hubert_model(mod):
    """
    Recursively look inside .model, .hubert or .encoder attributes
    until we hit an object that has feature_extractor_audio (the real AVHubertModel).
    """
    if hasattr(mod, "feature_extractor_audio") and hasattr(mod, "feature_extractor_video"):
        return mod

    for attr in ("model", "hubert", "encoder"):
        if hasattr(mod, attr):
            candidate = getattr(mod, attr)
            found = _find_hubert_model(candidate)
            if found is not None:
                return found

    return None


def export_if_needed(pt_checkpoint="misc/model.pt"):
    import os, torch
    from fairseq import checkpoint_utils

    # 1) skip if already exported
    if os.path.exists(ONNX_PATH):
        return

    print("🔥  First run: exporting PyTorch model → ONNX …")

    # 2) load
    models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
        [pt_checkpoint],
        arg_overrides={
            "task.label_dir": _DUMMY_DIR,
            "task.data":      _DUMMY_DIR,
        },
    )
    raw = models[0]

    # 3) find the real AVHubertModel inside whatever wrapper Fairseq gave us
    hubert = _find_hubert_model(raw)
    if hubert is None:
        raise RuntimeError(f"Could not locate AVHubertModel inside checkpoint wrapper {type(raw)}")

    # 4) move to CPU & eval
    hubert = hubert.cpu().eval()

    # 5) wrap it for ONNX
    wrapper = AVHubertOnnxWrapper(hubert)

    audio_proj = hubert.feature_extractor_audio.proj
    video_proj = hubert.feature_extractor_video.proj
    in_dim_audio = audio_proj.in_features
    in_dim_video = video_proj.in_features

    dummy_a = torch.randn(1, in_dim_audio, 200)         # [B, C_audio, T]
    dummy_v = torch.randn(1, in_dim_video, 88, 88, 200) # [B, C_video, H, W, T]

    # 8) export
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_a, dummy_v),
            ONNX_PATH,
            input_names=["audio", "video"],
            output_names=["feat_audio", "feat_video"],
            dynamic_axes={"audio": {2: "T"}, "video": {4: "T"}},
            opset_version=16,
        )
    print(f"✅  Exported to {ONNX_PATH}")

    
from fairseq.data import dictionary as _fsq_dict
_real_load = _fsq_dict.Dictionary.load
ONNX_PATH = "avhubert_extractor.onnx"

_DUMMY_DIR = "/Users/zoeykatzive/Desktop/CSCI 1470/FACTOR/dummy_dict"
print("⚙️  redirecting dictionaries to", _DUMMY_DIR)

def _patched_load(path, *a, **kw):
    if "/dict.km.txt" in path or "/dict.wrd.txt" in path:
        fname = "dict.km.txt" if "dict.km.txt" in path else "dict.wrd.txt"
        path = f"{_DUMMY_DIR}/{fname}"
    return _real_load(path, *a, **kw)

_fsq_dict.Dictionary.load = staticmethod(_patched_load)

###############################################################################
import torch
from torch.nn.modules.module import Module as _TorchModule

_orig_load_state_dict = _TorchModule.load_state_dict

def _load_state_dict_ignore_size(self, state_dict, strict=True, *args, **kwargs):
    own_state = self.state_dict()
    filtered = {
        k: v for k, v in state_dict.items()
        if k in own_state and v.shape == own_state[k].shape
    }
    for k, v in state_dict.items():
        if k not in filtered:
            print(f"[load_state_dict]  skip size-mismatched key: {k} {v.shape} "
                  f"-> {own_state.get(k, None)}")
    return _orig_load_state_dict(self, filtered, strict=False, *args, **kwargs)

_TorchModule.load_state_dict = _load_state_dict_ignore_size
###############################################################################




export_if_needed()
#SECTION 2
import onnxruntime as ort
import tensorflow as tf

ort_sess = ort.InferenceSession(ONNX_PATH, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

def layer_norm_np(x):
    """Re-implementation of torch.nn.functional.layer_norm in NumPy."""
    eps = 1e-5
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps)

def stacker(feats, k=4):
    T, F = feats.shape
    pad = (-T) % k
    if pad:
        feats = np.concatenate([feats, np.zeros((pad, F), feats.dtype)], 0)
    return feats.reshape(-1, k * F)

def load_audio_numpy(path):
    # 1) load waveform @16 kHz
    wav, sr = librosa.load(path, sr=16_000)
    # 2) compute log-filterbanks (float32)
    fbanks = logfbank(wav, samplerate=sr).astype(np.float32)
    # 3) stack into 4-frame blocks and layer-norm
    fbanks = stacker(fbanks, 4)            
    fbanks = layer_norm_np(fbanks)         
    # 4) transpose to (F, T) and add batch dim → (1, F, T)
    a = fbanks.T[None, :, :]
    return a.astype(np.float32)



from av_hubert.avhubert.utils import (
    Compose,
    Normalize,
    CenterCrop,
    load_video,
)




def load_video_numpy(path):
  
    frames = load_video(path).astype(np.float32)
    frames /= 255.0
    T, H, W = frames.shape
    top  = (H - 88) // 2
    left = (W - 88) // 2
    frames = frames[:, top : top+88, left : left+88]   
    frames = (frames - 0.421) / 0.165              
    v = frames.transpose(1,2,0)[None, None, :, :, :]
    return v.astype(np.float32)

#SECTION 3
def extract_features(mouth_roi_path, audio_path):
    a_np = load_audio_numpy(audio_path)         
    v_np = load_video_numpy(mouth_roi_path)     

    ort_out = ort_sess.run(None, {"audio": a_np, "video": v_np})
    feat_audio, feat_video = ort_out
    return feat_audio.squeeze(0), feat_video.squeeze(0)


def main():
    import argparse, os, pathlib, warnings, numpy as np, torch
    from tqdm import tqdm
    from av_hubert.avhubert.inference import extract_features, subsample_frames

    ap = argparse.ArgumentParser()
    ap.add_argument("--input_root", required=True)
    ap.add_argument("--output_dir")
    args = ap.parse_args()

 
    if args.output_dir is None:
        args.output_dir = (
            pathlib.Path(__file__).with_suffix("").parent
            / f"features_{pathlib.Path(args.input_root).name}"
        )
    os.makedirs(args.output_dir, exist_ok=True)

    all_audio, all_video, all_sims, paths = [], [], [], []
    N = 32  
    for root, _, files in tqdm(list(os.walk(args.input_root))):
        for f in files:
            if not f.endswith(".mp4") or f.endswith("_roi.mp4"):
                continue

            roi = os.path.join(root, f[:-4] + "_roi.mp4")
            wav = os.path.join(root, f[:-4] + ".wav")
            try:
                fa, fv = extract_features(roi, wav)  
            except Exception as e:
                warnings.warn(f"skip {f}: {e}")
                continue

            # 1) keep raw
            all_audio.append(fa)
            all_video.append(fv)
            paths.append(f)

            # 2) subsample both to N=32 frames

            fa32 = subsample_frames(fa.T, N).T   
            fv32 = subsample_frames(fv.T, N).T   

            # 3) cosine‐mean sim over those 32 frames
            a = torch.from_numpy(fa32)
            v = torch.from_numpy(fv32)
            a_n = a / a.norm(dim=0, keepdim=True)
            v_n = v / v.norm(dim=0, keepdim=True)
            cosine_per_frame = (a_n * v_n).sum(dim=0)  
            sim_scalar = cosine_per_frame.mean().item()
            all_sims.append(sim_scalar)

    audio_arr = np.empty(len(all_audio), dtype=object)
    video_arr = np.empty(len(all_video), dtype=object)
    paths_arr = np.empty(len(paths),       dtype=object)
    sims_arr  = np.array(all_sims, dtype=np.float32)

    for i, (fa, fv, p) in enumerate(zip(all_audio, all_video, paths)):
        audio_arr[i] = fa
        video_arr[i] = fv
        paths_arr[i] = p

    np.save(os.path.join(args.output_dir, "audio.npy"), audio_arr)
    np.save(os.path.join(args.output_dir, "video.npy"), video_arr)
    np.save(os.path.join(args.output_dir, "sims.npy"), sims_arr)
    np.save(os.path.join(args.output_dir, "paths.npy"))

    print(f"✅ Saved {len(paths)} items + sims → {args.output_dir}")



if __name__ == "__main__":
    main()