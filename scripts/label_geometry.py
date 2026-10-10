"""Why do some bin vocabularies learn and others not? The geometry of their output embeddings.

A slot's probability for bin k is softmax(h · E_k): LoRA trains h, but Gemma ties input and output
embeddings and the adapter does not touch them, so every label set's E is fixed. To put a smooth
bump of probability around the right bin, h has to be steered through the labels in order. That
is easy when the E_k already vary smoothly with k, and impossible when two labels share a row.

For every candidate label set this reports:
  norm      mean embedding norm (and its spread)
  cos       mean pairwise cosine; near-dup = share of pairs above 0.99 (cannot be told apart)
  order     mean cosine of neighbouring bins minus that of random pairs (> 0: ordered geometry)
  smooth    R² of E_k regressed on Fourier features of k/n (4 frequencies); chance ≈ 9/n
  rank      numerical rank of the label rows (rows a hidden state can address independently)

  python scripts/label_geometry.py   # reads the embedding from the served checkpoint, CPU only
"""

import json, math, os

import numpy as np
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoProcessor

from slotread import labels as L

REPO = "nvidia/Gemma-4-31B-IT-NVFP4"


def embedding():
    idx = json.load(open(hf_hub_download(REPO, "model.safetensors.index.json")))["weight_map"]
    key = next(k for k in idx if k.endswith("embed_tokens.weight") and "language_model" in k)
    with safe_open(hf_hub_download(REPO, idx[key]), "pt") as f:
        return f.get_tensor(key).float(), key


def stats(E, ids):
    X = E[ids].numpy().astype(np.float64)
    n = len(ids)
    norms = np.linalg.norm(X, axis=1)
    U = X / norms[:, None]
    C = U @ U.T
    off = C[~np.eye(n, dtype=bool)]
    rng = np.random.default_rng(0)
    rand = np.mean([C[i, j] for i, j in rng.integers(0, n, (4000, 2)) if i != j])
    order = float(np.mean([C[k, k + 1] for k in range(n - 1)]) - rand)
    t = (np.arange(n) + 0.5) / n
    F = np.column_stack([np.ones(n)] + [f(2 * math.pi * m * t) for m in range(1, 5) for f in (np.sin, np.cos)])
    Xc = X - X.mean(0)
    coef, *_ = np.linalg.lstsq(F, Xc, rcond=None)
    r2 = 1 - ((Xc - F @ coef) ** 2).sum() / (Xc ** 2).sum()
    s = np.linalg.svd(X, compute_uv=False)
    rank = int((s > s[0] * 1e-3).sum())
    return dict(n=n, norm=float(norms.mean()), norm_cv=float(norms.std() / norms.mean()), cos=float(off.mean()),
                near_dup=float((off > 0.99).mean()), order=order, smooth_r2=float(r2), chance_r2=9 / n, rank=rank)


def main():
    tok = AutoProcessor.from_pretrained(REPO).tokenizer
    E, key = embedding()
    print(f"{key}: {tuple(E.shape)}\n")
    v = tok.get_vocab()
    fb = [v[f"<0x{b:02X}>"] for b in range(256)]
    sets = {}
    sets["words 256 (A / C)"] = L.word_labels(tok, 256)
    sets["words 101 (B+w)"] = L.word_labels(tok, 101)
    sets["ascii 101 (B+: 84 chars + 17 two-letter)"] = L.byte_labels(tok, 101)[:101]
    sets["ascii 256 (E / F: 84 chars + 172 two-letter)"] = L.byte_labels(tok, 256)[:256]
    sets["byte-fallback 256 (<0x00>-<0xFF>)"] = fb
    sets["byte-fallback low 128 (<0x00>-<0x7F>)"] = fb[:128]
    sets["byte-fallback high 128 (<0x80>-<0xFF>)"] = fb[128:]
    sets["byte-fallback 101 (every 2.5th)"] = [fb[int(i * 256 / 101)] for i in range(101)]
    sets["digits 0-9 (coarse slot)"] = L.DIGITS
    rows = {}
    print("| label set | n | norm (cv) | mean cos | near-dup | order | smooth R² (chance) | rank |")
    print("|---|---|---|---|---|---|---|---|")
    for name, ids in sets.items():
        r = stats(E, ids)
        rows[name] = r
        print(f"| {name} | {r['n']} | {r['norm']:.2f} ({r['norm_cv']:.2f}) | {r['cos']:.3f} | {r['near_dup']:.1%} | "
              f"{r['order']:+.3f} | {r['smooth_r2']:.3f} ({r['chance_r2']:.3f}) | {r['rank']} |", flush=True)
    os.makedirs("results", exist_ok=True)
    json.dump(rows, open("results/label_geometry.json", "w"), indent=1)


if __name__ == "__main__":
    main()
