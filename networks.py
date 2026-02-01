# networks.py – Shared Encoder, CRN (mit Uncertainty), MSPN, MCN
import torch
import torch.nn as nn
import numpy as np

class SharedEncoder(nn.Module):
    def __init__(self, input_dim=228, latent_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
            nn.ReLU()
        )
    def forward(self, x): return self.net(x)


class CRN(nn.Module):
    """Context Recognition Network: 50–100 Frames latent -> Modus-Wahrscheinlichkeiten + Uncertainty."""
    def __init__(self, latent_dim=128, hidden_dim=64, num_modes=10, dropout=0.1):
        super().__init__()
        self.num_modes = num_modes
        self.lstm = nn.LSTM(latent_dim, hidden_dim, batch_first=True, dropout=dropout if dropout else 0)
        self.fc = nn.Linear(hidden_dim, num_modes)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, return_uncertainty=False):
        # x: (batch, seq_len, latent_dim)
        self.dropout.train(self.training)
        x = self.dropout(x)
        out, _ = self.lstm(x)
        logits = self.fc(out[:, -1, :])
        probs = torch.softmax(logits, dim=-1)
        if return_uncertainty:
            entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1)
            max_entropy = np.log(self.num_modes)
            uncertainty = entropy / max_entropy  # 0 = sicher, 1 = unsicher
            return probs, uncertainty
        return probs

class MSPN(nn.Module):
    def __init__(self, latent_dim=128, action_dim=12):
        super().__init__()
        self.actor = nn.Sequential(nn.Linear(latent_dim, 64), nn.ReLU(), nn.Linear(64, action_dim), nn.Softmax(dim=-1))
        self.critic = nn.Sequential(nn.Linear(latent_dim, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(self, x): return self.actor(x), self.critic(x)

class MCN(nn.Module):
    def __init__(self, num_modes=10):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(num_modes + 5, 32), nn.ReLU(), nn.Linear(32, num_modes), nn.Softmax(dim=-1))
    def forward(self, mode_probs, meta_stats):
        return self.net(torch.cat([mode_probs, meta_stats], dim=-1))

# Classical Deep RL Architectures
class DQNet(nn.Module):
    def __init__(self, input_dim=228):
        super().__init__(); self.net = nn.Sequential(nn.Linear(input_dim, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 12))
    def forward(self, x): return self.net(x)

class ACNet(nn.Module):
    def __init__(self, input_dim=228):
        super().__init__(); self.base = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU())
        self.actor = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 12), nn.Softmax(dim=-1))
        self.critic = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(self, x): b = self.base(x); return self.actor(b), self.critic(b)

class SynergyNet(nn.Module):
    def __init__(self, input_dim=228):
        super().__init__(); self.net = nn.Sequential(nn.Linear(input_dim+12, 128), nn.ReLU(), nn.Linear(128, 12))
    def forward(self, s, a_oh): return self.net(torch.cat([s, a_oh], dim=-1))

class DuelingDQNet(nn.Module):
    def __init__(self, input_dim=228, output_dim=12):
        super().__init__()
        self.feature = nn.Sequential(nn.Linear(input_dim, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU())
        self.advantage = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, output_dim))
        self.value = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1))
    def forward(self, x):
        x = self.feature(x)
        advantage = self.advantage(x)
        value = self.value(x)
        return value + advantage - advantage.mean(dim=-1, keepdim=True)
