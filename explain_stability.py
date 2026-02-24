import argparse
import json
import os
import subprocess
import sys

import numpy as np


def _rank(x):
    x = np.asarray(x, dtype=np.float32)
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=np.float32)
    ranks[order] = np.arange(len(x), dtype=np.float32)
    return ranks


def _spearman(a, b):
    if a.size < 2:
        return 0.0
    ra = _rank(a)
    rb = _rank(b)
    return float(np.corrcoef(ra, rb)[0, 1])


def _pearson(a, b):
    if a.size < 2:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _jaccard_topk(a, b, k=20):
    if a.size == 0 or b.size == 0:
        return 0.0
    k = min(k, a.size, b.size)
    top_a = set(np.argsort(-a)[:k].tolist())
    top_b = set(np.argsort(-b)[:k].tolist())
    if not top_a and not top_b:
        return 1.0
    return float(len(top_a & top_b)) / float(len(top_a | top_b) + 1e-12)


def _extract_pair_ids(extra):
    drug_id = None
    prot_id = None
    for i, tok in enumerate(extra):
        if tok == "--drug_id" and i + 1 < len(extra):
            drug_id = extra[i + 1]
        if tok == "--prot_id" and i + 1 < len(extra):
            prot_id = extra[i + 1]
    return drug_id, prot_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=str, default="11,12,13")
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--method", type=str, default="attn")
    parser.add_argument("--out_dir", type=str, default="explain_stability")
    args, extra = parser.parse_known_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    os.makedirs(args.out_dir, exist_ok=True)
    drug_id, prot_id = _extract_pair_ids(extra)

    json_paths = []
    for seed in seeds:
        cmd = [sys.executable, "explain.py"] + extra
        cmd += ["--explain_seed", str(seed), "--export_json", "--export_dir", args.out_dir]
        subprocess.check_call(cmd)
        if drug_id is not None and prot_id is not None:
            json_paths.append(
                os.path.join(args.out_dir, f"drug_{drug_id}_prot_{prot_id}_explain.json")
            )
        else:
            # fallback: pick latest json
            files = [f for f in os.listdir(args.out_dir) if f.endswith("_explain.json")]
            latest = max(files, key=lambda f: os.path.getmtime(os.path.join(args.out_dir, f)))
            json_paths.append(os.path.join(args.out_dir, latest))

    results = []
    for path in json_paths:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        scores = data.get("scores", {}).get(args.method, {})
        atom = np.asarray(scores.get("atom_norm") or [], dtype=np.float32)
        resi = np.asarray(scores.get("res_norm") or [], dtype=np.float32)
        results.append((atom, resi))

    if len(results) < 2:
        print("Not enough runs for stability.")
        return
    base_atom, base_res = results[0]
    jacc_atoms, jacc_res, spe_atoms, spe_res, pear_atoms, pear_res = [], [], [], [], [], []
    for atom, resi in results[1:]:
        if base_atom.size and atom.size:
            jacc_atoms.append(_jaccard_topk(base_atom, atom, k=args.topk))
            spe_atoms.append(_spearman(base_atom, atom))
            pear_atoms.append(_pearson(base_atom, atom))
        if base_res.size and resi.size:
            jacc_res.append(_jaccard_topk(base_res, resi, k=args.topk))
            spe_res.append(_spearman(base_res, resi))
            pear_res.append(_pearson(base_res, resi))

    print(f"[STAB] atom jacc@{args.topk}={np.mean(jacc_atoms):.3f} spearman={np.mean(spe_atoms):.3f} pearson={np.mean(pear_atoms):.3f}")
    print(f"[STAB] res  jacc@{args.topk}={np.mean(jacc_res):.3f} spearman={np.mean(spe_res):.3f} pearson={np.mean(pear_res):.3f}")


if __name__ == "__main__":
    main()
