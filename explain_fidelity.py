import argparse
import json
import os
import subprocess
import sys


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
    parser.add_argument("--out_dir", type=str, default="explain_fidelity")
    args, extra = parser.parse_known_args()

    os.makedirs(args.out_dir, exist_ok=True)
    drug_id, prot_id = _extract_pair_ids(extra)
    cmd = [sys.executable, "explain.py"] + extra
    cmd += ["--explain_methods", "attn,occlusion", "--export_json", "--export_dir", args.out_dir]
    subprocess.check_call(cmd)

    if drug_id is not None and prot_id is not None:
        json_path = os.path.join(args.out_dir, f"drug_{drug_id}_prot_{prot_id}_explain.json")
    else:
        files = [f for f in os.listdir(args.out_dir) if f.endswith("_explain.json")]
        json_path = os.path.join(args.out_dir, files[0]) if files else None
    if not json_path or not os.path.exists(json_path):
        print("Explain JSON not found.")
        return
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    fidelity = data.get("fidelity", {})
    print(json.dumps(fidelity, indent=2))


if __name__ == "__main__":
    main()
