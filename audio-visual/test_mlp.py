
import os, sys, argparse
import numpy as np
import torch

def subsample_frames(feat: np.ndarray, num_frames: int) -> np.ndarray:
    idx = np.linspace(0, feat.shape[0] - 1, num_frames).astype(int)
    return feat[idx]

if __name__ == "__main__":
    p = argparse.ArgumentParser("Test SimpleAVClassifier on multiple feature sets")
    p.add_argument(
        "--data_dirs", nargs="+",
        default=["features_real", "features_fake"],
        help="one or more feature folders to evaluate"
    )
    p.add_argument(
        "--model_ckpt", default="mlp.pt",
        help="path to your trained MLP checkpoint"
    )
    args = p.parse_args()

    sys.path.insert(0, os.path.join(os.getcwd(), "av_hubert", "avhubert"))
    from mlp import SimpleAVClassifier

    sample_dir = args.data_dirs[0]
    sample_audio = np.load(os.path.join(sample_dir, "audio.npy"), allow_pickle=True)[0]
    sample_video = np.load(os.path.join(sample_dir, "video.npy"), allow_pickle=True)[0]
    C_a, _ = sample_audio.shape
    C_v, _ = sample_video.shape

    mlp = SimpleAVClassifier(C_a, C_v)
    mlp.load_state_dict(torch.load(args.model_ckpt, map_location="cpu"))
    mlp.eval()

    N = 32 

    for data_dir in args.data_dirs:
        print(f"\n=== Evaluating {data_dir} ===")
        audio = np.load(os.path.join(data_dir, "audio.npy"),  allow_pickle=True)
        video = np.load(os.path.join(data_dir, "video.npy"),  allow_pickle=True)
        sims  = np.load(os.path.join(data_dir, "sims.npy"))
        paths = np.load(os.path.join(data_dir, "paths.npy"), allow_pickle=True)

        for fa, fv, sim, fn in zip(audio, video, sims, paths):
            fa32 = subsample_frames(fa.T, N).T
            fv32 = subsample_frames(fv.T, N).T

            a_t = torch.from_numpy(fa32).float().unsqueeze(0)  
            v_t = torch.from_numpy(fv32).float().unsqueeze(0)  
            s_t = torch.tensor([[sim]], dtype=torch.float32)  

            with torch.no_grad():
                logits = mlp(a_t, v_t, s_t)
                probs  = torch.softmax(logits, dim=-1)

            # [0,1] index is the "fake" class probability
            print(f"{fn:30s}   fake_prob={probs[0,1].item():.3f}")







