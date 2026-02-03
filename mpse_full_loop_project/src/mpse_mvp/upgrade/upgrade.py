from __future__ import annotations
import os, json, math
import numpy as np
import torch

from mpse_mvp.mpse.model import MPSE

def load_mpse(ckpt: str, meta_path: str, device: str = "cpu"):
    meta = json.load(open(meta_path, encoding="utf-8"))
    model = MPSE(meta["in_dim"], meta["hidden_dim"], meta["num_idx"], dropout=meta["dropout"])
    sd = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(sd)
    model.to(device); model.eval()
    return model, meta

@torch.no_grad()
def infer_mpse(model: MPSE, X: np.ndarray, device: str = "cpu"):
    xt = torch.from_numpy(X.astype(np.float32)).to(device)
    mu, sigma, alpha, _logvar = model(xt)
    return (mu.cpu().numpy(), sigma.cpu().numpy(), alpha.cpu().numpy())

def compute_p_ok(mu: float, sigma: float, tau: float):
    # P(S <= tau) for Normal(mu, sigma^2)
    # approximate CDF via erf
    z = (tau - mu) / (sigma + 1e-8)
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

def upgrade(turns_path: str, X: np.ndarray, ckpt: str, meta_path: str,
            idx_names: list[str], tau: dict, out_turns_path: str,
            sigma_lambda: float = 2.0, sigma_max: float = 0.12,
            inject_state_tokens: bool = True, device: str = "cpu"):

    rows = [json.loads(l) for l in open(turns_path, encoding="utf-8")]
    model, meta = load_mpse(ckpt, meta_path, device=device)
    MU, SIG, ALPHA = infer_mpse(model, X, device=device)

    out_rows = []
    for i,r in enumerate(rows):
        mu_i = {k: float(MU[i, j]) for j,k in enumerate(idx_names)}
        sig_i = {k: float(SIG[i, j]) for j,k in enumerate(idx_names)}
        alpha_i = {"T": float(ALPHA[i,0]), "A": float(ALPHA[i,1]), "V": float(ALPHA[i,2])}

        # weight: penalize average sigma (exclude microexpr_rate to avoid weirdness)
        sig_list = [sig_i[k] for k in idx_names if k != "microexpr_rate"]
        sig_bar = float(np.mean(sig_list)) if sig_list else float(np.mean(list(sig_i.values())))
        w = float(math.exp(-sigma_lambda * sig_bar))

        # trusted improvement: compare with previous in-session (MVP assumes one session)
        if i == 0:
            eff_raw = 0
            eff_trusted = 0
        else:
            # improvement if dep/sad/anx/stress decreased on average
            prev = out_rows[-1]["mu"]
            prev_sig = out_rows[-1]["sigma"]
            keys = [k for k in idx_names if k != "microexpr_rate"]
            dmu = float(np.mean([mu_i[k] - prev[k] for k in keys]))
            eff_raw = 1 if dmu < 0 else 0
            sig_ok = (sig_bar < sigma_max) and (float(np.mean([prev_sig[k] for k in keys])) < sigma_max)
            eff_trusted = 1 if (eff_raw == 1 and sig_ok) else 0

        p_ok = {k: float(compute_p_ok(mu_i[k], sig_i[k], float(tau.get(k,0.30)))) for k in idx_names}

        out = dict(r)
        out["mu"] = mu_i
        out["sigma"] = sig_i
        out["alpha"] = alpha_i
        out["weight"] = w
        out["effective_raw"] = eff_raw
        out["effective_trusted"] = eff_trusted
        out["p_ok"] = p_ok
        out_rows.append(out)

    os.makedirs(os.path.dirname(out_turns_path), exist_ok=True)
    with open(out_turns_path, "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return out_turns_path
