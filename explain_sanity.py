import argparse
import json
import os
import tempfile
import subprocess
import sys

import numpy as np
import torch


def _extract_pair_ids(extra):
    drug_id = None
    prot_id = None
    for i, tok in enumerate(extra):
        if tok == "--drug_id" and i + 1 < len(extra):
            drug_id = extra[i + 1]
        if tok == "--prot_id" and i + 1 < len(extra):
            prot_id = extra[i + 1]
    return drug_id, prot_id


def _load_scores(json_path, method="attn"):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    scores = data.get("scores", {}).get(method, {})
    atom = np.asarray(scores.get("atom_norm") or [], dtype=np.float32)
    resi = np.asarray(scores.get("res_norm") or [], dtype=np.float32)
    return atom, resi


def _corr(a, b):
    if a.size < 2 or b.size < 2:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--method", type=str, default="attn")
    parser.add_argument("--out_dir", type=str, default="explain_sanity")
    args, extra = parser.parse_known_args()

    os.makedirs(args.out_dir, exist_ok=True)
    drug_id, prot_id = _extract_pair_ids(extra)

    # Baseline explain
    cmd = [sys.executable, "explain.py"] + extra + ["--ckpt", args.ckpt, "--export_json", "--export_dir", args.out_dir]
    subprocess.check_call(cmd)
    if drug_id is not None and prot_id is not None:
        base_json = os.path.join(args.out_dir, f"drug_{drug_id}_prot_{prot_id}_explain.json")
    else:
        files = [f for f in os.listdir(args.out_dir) if f.endswith("_explain.json")]
        base_json = os.path.join(args.out_dir, files[0]) if files else None
    if not base_json or not os.path.exists(base_json):
        print("Baseline explain JSON not found.")
        return
    base_atom, base_res = _load_scores(base_json, method=args.method)

    # Randomize head weights and save temp ckpt
    ckpt = torch.load(args.ckpt, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt
    for k in list(state.keys()):
        if "edge_mlp" in k or "edge_hyper_mlp" in k:
            state[k] = torch.randn_like(state[k]) * 0.02
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".pth")
    os.close(tmp_fd)
    torch.save(state, tmp_path)

    # Sanity explain
    cmd = [sys.executable, "explain.py"] + extra + ["--ckpt", tmp_path, "--export_json", "--export_dir", args.out_dir]
    subprocess.check_call(cmd)
    if drug_id is not None and prot_id is not None:
        san_json = os.path.join(args.out_dir, f"drug_{drug_id}_prot_{prot_id}_explain.json")
    else:
        files = [f for f in os.listdir(args.out_dir) if f.endswith("_explain.json")]
        san_json = os.path.join(args.out_dir, files[-1]) if files else None
    if not san_json or not os.path.exists(san_json):
        print("Sanity explain JSON not found.")
        return
    san_atom, san_res = _load_scores(san_json, method=args.method)

    print(f"[SANITY] atom corr={_corr(base_atom, san_atom):.3f} res corr={_corr(base_res, san_res):.3f}")
    try:
        os.remove(tmp_path)
    except OSError:
        pass


if __name__ == "__main__":
    main()
