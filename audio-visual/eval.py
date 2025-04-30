#!/usr/bin/env python
import os
import argparse
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

def subsample_frames(feat, num_frames=32):
    """
    feat: np.ndarray of shape (T, C)
    returns exactly num_frames rows by uniform sampling along T.
    """
    T = feat.shape[0]
    if T == 0:
        raise ValueError("Empty feature sequence")
    idx = np.linspace(0, T - 1, num_frames).astype(int)
    return feat[idx]

def compute_percentile_scores(audio_list, video_list, lambda_pct, num_frames):
    scores = []
    for a, v in zip(audio_list, video_list):
        a_t = a.T                     
        v_t = v.T                     
        a_sub = subsample_frames(a_t, num_frames)
        v_sub = subsample_frames(v_t, num_frames)

        na = np.linalg.norm(a_sub, axis=1, keepdims=True)
        nv = np.linalg.norm(v_sub, axis=1, keepdims=True)
        a_n = a_sub / na
        v_n = v_sub / nv

        dots = (a_n * v_n).sum(axis=1)     
        scores.append(np.percentile(dots, lambda_pct))
    return np.array(scores, dtype=np.float32)

def compute_mean_scores(audio_list, video_list, num_frames):
    scores = []
    for a, v in zip(audio_list, video_list):
        a_t = a.T
        v_t = v.T
        a_sub = subsample_frames(a_t, num_frames)
        v_sub = subsample_frames(v_t, num_frames)

        na = np.linalg.norm(a_sub, axis=1, keepdims=True)
        nv = np.linalg.norm(v_sub, axis=1, keepdims=True)
        a_n = a_sub / na
        v_n = v_sub / nv

        dots = (a_n * v_n).sum(axis=1)
        scores.append(dots.mean())
    return np.array(scores, dtype=np.float32)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--real_dir",   required=True,
                   help="path to features_real (audio.npy + video.npy)")
    p.add_argument("--fake_dir",   required=True,
                   help="path to features_fake (audio.npy + video.npy)")
    p.add_argument("--mode",       choices=["percentile","mean"],
                   default="percentile",
                   help="‘percentile’ uses a λ-percentile; ‘mean’ uses frame-mean")
    p.add_argument("--lambda_pct", type=float, default=3,
                   help="percentile for ‘percentile’ mode")
    p.add_argument("--num_frames", type=int,   default=32,
                   help="number of frames to subsample to")
    args = p.parse_args()

    #load feature arrays

    real_a = np.load(os.path.join(args.real_dir, "audio.npy"), allow_pickle=True)
    real_v = np.load(os.path.join(args.real_dir, "video.npy"), allow_pickle=True)
    fake_a = np.load(os.path.join(args.fake_dir, "audio.npy"), allow_pickle=True)
    fake_v = np.load(os.path.join(args.fake_dir, "video.npy"), allow_pickle=True)

    # compute truth  scores
    if args.mode == "percentile":
        real_scores = compute_percentile_scores(real_a, real_v,
                                                args.lambda_pct,
                                                args.num_frames)
        fake_scores = compute_percentile_scores(fake_a, fake_v,
                                                args.lambda_pct,
                                                args.num_frames)
        mode_desc = f"λ={args.lambda_pct}"
    else:
        real_scores = compute_mean_scores(real_a, real_v, args.num_frames)
        fake_scores = compute_mean_scores(fake_a, fake_v, args.num_frames)
        mode_desc = "mean"

   
    all_scores = -np.concatenate([real_scores, fake_scores])
    labels     = np.concatenate([
        np.zeros_like(real_scores, dtype=int),
        np.ones_like (fake_scores, dtype=int),
    ])

    auc = roc_auc_score(labels, all_scores)
    ap  = average_precision_score(labels, all_scores)

    print(f"AUC @ mode={args.mode} ({mode_desc}), N={args.num_frames} → {auc*100:.2f}")
    print(f"  AP @ mode={args.mode} ({mode_desc}), N={args.num_frames} → {ap*100:.2f}")


