from __future__ import annotations
import os, json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from .model import MPSE

class TurnDataset(Dataset):
    def __init__(self, npz_path: str):
        d = np.load(npz_path, allow_pickle=True)
        self.X = d["X"].astype(np.float32)
        self.Y = d["Y"].astype(np.float32)
        self.Q = d["Q"].astype(np.float32)  # quality weights per sample
    def __len__(self):
        return len(self.X)
    def __getitem__(self, i):
        return self.X[i], self.Y[i], self.Q[i]

def nll_loss(mu, logvar, y):
    # Gaussian NLL for heteroscedastic regression
    var = torch.exp(logvar)
    return 0.5 * ((y - mu)**2 / (var + 1e-8) + logvar)

def train_mpse(npz_path: str, out_dir: str, epochs: int = 5, batch_size: int = 8, lr: float = 3e-4,
               hidden_dim: int = 256, dropout: float = 0.1, device: str = "cuda"):
    os.makedirs(out_dir, exist_ok=True)
    ds = TurnDataset(npz_path)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    in_dim = ds.X.shape[1]
    num_idx = ds.Y.shape[1]
    model = MPSE(in_dim, hidden_dim, num_idx, dropout=dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    for ep in range(epochs):
        model.train()
        pbar = tqdm(dl, desc=f"MPSE train ep {ep+1}/{epochs}")
        loss_sum = 0.0
        n = 0
        for X, Y, Q in pbar:
            X = torch.from_numpy(X).to(device)
            Y = torch.from_numpy(Y).to(device)
            Q = torch.from_numpy(Q).to(device)

            mu, sigma, alpha, logvar = model(X)

            # quality-weighted NLL
            nll = nll_loss(mu, logvar, Y).mean(dim=1)  # [B]
            loss_nll = (nll * Q).mean()

            # alpha entropy regularizer (avoid collapse)
            ent = -(alpha * torch.log(alpha + 1e-8)).sum(dim=1)
            loss_alpha = (-ent).mean() * 0.01

            loss = loss_nll + loss_alpha
            opt.zero_grad()
            loss.backward()
            opt.step()

            loss_sum += float(loss.detach().cpu())
            n += 1
            pbar.set_postfix(loss=loss_sum/max(1,n))

    ckpt = os.path.join(out_dir, "mpse.pt")
    torch.save(model.state_dict(), ckpt)
    meta = {"in_dim": in_dim, "num_idx": num_idx, "hidden_dim": hidden_dim, "dropout": dropout}
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return ckpt
