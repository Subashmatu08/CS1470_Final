
import torch
import torch.nn as nn
class SimpleAVClassifier(nn.Module):
    def __init__(self, C_a, C_v, hidden_dim=128, num_classes=2):
        """
        C_a: audio‐feature dimension
        C_v: video‐feature dimension
        hidden_dim: size of the MLP’s hidden layer
        num_classes: number of output classes (real vs fake)
        """
        super().__init__()
        # input to the MLP: audio + video + similarity scalar
        self.fc1 = nn.Linear(C_a + C_v + 1, hidden_dim)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, a, v, sim_scalar):
        """
        a: Tensor of shape (B, C_a, N)   -- batch of N‐frame audio features
        v: Tensor of shape (B, C_v, N)   -- batch of N‐frame video features
        sim_scalar: Tensor of shape (B, 1)  -- precomputed cosine‐mean per sample
        """
 
        a_p = a.mean(dim=2)
        v_p = v.mean(dim=2)

        x = torch.cat([a_p, v_p, sim_scalar], dim=1)


        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

