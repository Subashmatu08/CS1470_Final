
import os, sys
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "av_hubert/avhubert"))
from  av_hubert.avhubert.mlp import SimpleAVClassifier

class AVFeatureDataset(torch.utils.data.Dataset):
    def __init__(self, real_dir, fake_dir):
        # load real examples
        real_a = np.load(os.path.join(real_dir, "audio.npy"),  allow_pickle=True)
        real_v = np.load(os.path.join(real_dir, "video.npy"),  allow_pickle=True)
        real_s = np.load(os.path.join(real_dir, "sims.npy"))
        
        # load fake examples
        fake_a = np.load(os.path.join(fake_dir, "audio.npy"),  allow_pickle=True)
        fake_v = np.load(os.path.join(fake_dir, "video.npy"),  allow_pickle=True)
        fake_s = np.load(os.path.join(fake_dir, "sims.npy"))
        
        # concatenate into four parallel lists
        self.audios = list(real_a) + list(fake_a)
        self.videos = list(real_v) + list(fake_v)
        self.sims    = np.concatenate([real_s, fake_s])
        # 0 = real, 1 = fake
        self.labels  = [0] * len(real_a) + [1] * len(fake_a)
        
        # fixed number of frames for subsampling
        self.N = 32
        self.N = 32   
    def __len__(self):
        return len(self.audios)
    def __getitem__(self, i):
        a_np = self.audios[i]   
        v_np = self.videos[i]   
        s    = self.sims[i]
        y    = self.labels[i]

        from av_hubert.avhubert.inference import subsample_frames

        a32 = subsample_frames(a_np.T, self.N).T   
        v32 = subsample_frames(v_np.T, self.N).T   

        a_t = torch.tensor(a32, dtype=torch.float32)
        v_t = torch.tensor(v32, dtype=torch.float32)
        s_t = torch.tensor([s],  dtype=torch.float32)
        y_t = torch.tensor(y,    dtype=torch.long)

        return a_t, v_t, s_t, y_t

def train(real_dir="features_real", fake_dir="features_fake",
          epochs=10, batch_size=16, lr=1e-3, ckpt="mlp.pt"):
    ds = AVFeatureDataset(real_dir, fake_dir)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2)
    C_a = ds.audios[0].shape[0]
    C_v = ds.videos[0].shape[0]
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleAVClassifier(C_a, C_v).to(device)
    opt   = optim.Adam(model.parameters(), lr=lr)
    ce    = nn.CrossEntropyLoss()

    for ep in range(1, epochs+1):
        model.train()
        total = 0
        for a, v, s, y in dl:
            a, v, s, y = a.to(device), v.to(device), s.to(device), y.to(device)
            logits = model(a, v, s)           
            loss   = ce(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * a.size(0)
        print(f"Epoch {ep:2d}  avg loss {total/len(ds):.4f}")

    torch.save(model.state_dict(), ckpt)
    print(f"✔️  saved MLP checkpoint → {ckpt}")

if __name__=="__main__":
    train()

