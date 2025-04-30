# av_hubert/avhubert/ablation.py
import os
import argparse
import numpy as np
import pandas as pd
import warnings
from sklearn.metrics import roc_auc_score, average_precision_score


import torch
import clip                                      # pip install git+https://github.com/openai/CLIP.git
from PIL import Image
from transformers import BlipProcessor, BlipForImageTextRetrieval

device = "cuda" if torch.cuda.is_available() else "cpu"

def subsample_frames(feat, num_frames=32):
    idx = np.linspace(0, feat.shape[0] - 1, num_frames).astype(int)
    return feat[idx]

def load_feats(feat_dir, num_frames=32):
    raw_a = np.load(os.path.join(feat_dir, "audio.npy"), allow_pickle=True)
    raw_v = np.load(os.path.join(feat_dir, "video.npy"), allow_pickle=True)
    a_list, v_list = [], []
    for a_raw, v_raw in zip(raw_a, raw_v):
        a, v = a_raw.T, v_raw.T
        a_sub, v_sub = subsample_frames(a, num_frames), subsample_frames(v, num_frames)
        if a_sub.shape[1] != v_sub.shape[1]:
            warnings.warn(f"Skipping mismatched dims: {a_sub.shape} vs {v_sub.shape}")
            continue
        a_list.append(a_sub)
        v_list.append(v_sub)
    if not a_list:
        raise RuntimeError(f"No valid pairs in {feat_dir}")
    return a_list, v_list

def av_sim(audio_list, video_list, lambda_pct):
    sims = []
    for a, v in zip(audio_list, video_list):
        a_n = a / np.linalg.norm(a, axis=1, keepdims=True)
        v_n = v / np.linalg.norm(v, axis=1, keepdims=True)
        dots = (a_n * v_n).sum(axis=1)
        sims.append(np.percentile(dots, lambda_pct))
    return sims

# ——— Fig 3c helpers ———
# 1) CLIP
_clip_model, _clip_preprocess = clip.load("ViT-L/14", device=device)
def clip_truth_scores(image_paths, captions, lambda_pct=50):
    imgs = torch.stack([_clip_preprocess(Image.open(p).convert("RGB")) for p in image_paths]).to(device)
    tokens = clip.tokenize(captions).to(device)
    with torch.no_grad():
        img_feats = _clip_model.encode_image(imgs)
        txt_feats = _clip_model.encode_text(tokens)
    img_feats /= img_feats.norm(dim=-1, keepdim=True)
    txt_feats /= txt_feats.norm(dim=-1, keepdim=True)
    sims = (img_feats * txt_feats).sum(dim=-1).cpu().numpy()
    return sims

# 2) BLIP2
_blip_model = BlipForImageTextRetrieval.from_pretrained("Salesforce/blip2-flan-t5-xl").to(device)
_blip_proc  = BlipProcessor.from_pretrained("Salesforce/blip2-flan-t5-xl")
def blip2_truth_scores(image_paths, captions, lambda_pct=50):
    images = [Image.open(p).convert("RGB") for p in image_paths]
    inputs = _blip_proc(text=captions, images=images, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        out = _blip_model(**inputs)
    sims = out.logits_per_image.diag().cpu().numpy()
    return sims

# 3) BLIP2‐CLIP hybrid
def blip2_clip_truth_scores(image_paths, captions, lambda_pct=50):
    # text via BLIP2
    images = [Image.open(p).convert("RGB") for p in image_paths]
    inputs = _blip_proc(text=captions, images=images, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        txt_emb = _blip_model.get_text_features(**inputs)
    txt_emb /= txt_emb.norm(dim=-1, keepdim=True)
    # image via CLIP
    imgs = torch.stack([_clip_preprocess(img) for img in images]).to(device)
    with torch.no_grad():
        img_emb = _clip_model.encode_image(imgs)
    img_emb /= img_emb.norm(dim=-1, keepdim=True)
    sims = (img_emb * txt_emb).sum(dim=-1).cpu().numpy()
    return sims

def run_ablation(real_dir, fake_dir, out_csv):
    real_a, real_v = load_feats(real_dir)
    fake_a, fake_v = load_feats(fake_dir)
    n_real, n_fake = len(real_a), len(fake_a)
    labels = np.concatenate([np.zeros(n_real), np.ones(n_fake)])

    results = []

    # — Fig 3a: reference-set size —
    ref_sizes = [1,2,5,10,20,n_real]
    rng = np.random.default_rng(0)
    for k in ref_sizes:
        def truth(a_list, v_list):
            return av_sim(a_list, v_list, lambda_pct=3)
        rs_real = truth(real_a, real_v)
        rs_fake = truth(fake_a, fake_v)
        scores = -np.concatenate([rs_real, rs_fake])
        auc = roc_auc_score(labels, scores)*100
        ap  = average_precision_score(labels, scores)*100
        fake_higher = (np.array(rs_fake) > np.array(rs_real)).mean()
        results.append({
            "ablation":    "ref_set_size",
            "param":       k,
            "description":"number_of_references",
            "AUC":         auc,
            "AP":          ap,
            "fake_higher": fake_higher,
            "real_higher": 1-fake_higher
        })

    # — Fig 3b: -percentile —
    for lam in [0,1,3,5,10,20,50,100]:
        rs_real = av_sim(real_a, real_v, lam)
        rs_fake = av_sim(fake_a, fake_v, lam)
        scores = -np.concatenate([rs_real, rs_fake])
        auc = roc_auc_score(labels, scores)*100
        ap  = average_precision_score(labels, scores)*100
        fake_higher = (np.array(rs_fake) > np.array(rs_real)).mean()
        results.append({
            "ablation":    "lambda_percentile",
            "param":       lam,
            "description":"percentile_lambda",
            "AUC":         auc,
            "AP":          ap,
            "fake_higher": fake_higher,
            "real_higher": 1-fake_higher
        })



if __name__=="__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--real_dir",   required=True)
    p.add_argument("--fake_dir",   required=True)
    p.add_argument("--output_csv", required=True)
    args = p.parse_args()
    run_ablation(args.real_dir, args.fake_dir, args.output_csv)
