import argparse
import os
import pickle
import shutil
import time

import numpy as np
import pandas as pd


def _find_col(df, candidates):
    cols = list(df.columns)
    lower = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand in cols:
            return cand
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def _load_split_csv(dataset_dir):
    train_path = os.path.join(dataset_dir, "train.csv")
    test_path = os.path.join(dataset_dir, "test.csv")
    val_path = os.path.join(dataset_dir, "val.csv")
    if not os.path.exists(val_path):
        val_path = os.path.join(dataset_dir, "valid.csv")
    if not (os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path)):
        raise FileNotFoundError(
            f"Missing split csvs under {dataset_dir}. Required: train.csv, val.csv/valid.csv, test.csv"
        )
    train = pd.read_csv(train_path)
    val = pd.read_csv(val_path)
    test = pd.read_csv(test_path)
    return train, val, test, os.path.basename(val_path)


def _extract_ids(train, val, test):
    col_map = {
        "protein_sequence": [
            "protein_sequence", "Protein", "protein", "sequence", "Sequence",
            "Target Sequence", "target_sequence", "Target UniProt ID",
            "uniprot", "UniProt", "target"
        ],
        "drug_smiles": [
            "drug_smiles", "Ligand", "Ligand ID", "ligand", "SMILES", "smiles",
            "Ligand SMILES", "compound_smiles", "drug"
        ],
    }
    drug_col = _find_col(train, col_map["drug_smiles"])
    prot_col = _find_col(train, col_map["protein_sequence"])
    if drug_col is None or prot_col is None:
        raise ValueError(
            f"Could not identify columns. Found: {list(train.columns)}"
        )
    all_df = pd.concat([train, val, test], ignore_index=True)
    all_df = all_df.dropna(subset=[drug_col, prot_col])
    drug_ids = np.unique(all_df[drug_col].astype(str))
    prot_ids = np.unique(all_df[prot_col].astype(str))
    return drug_ids, prot_ids, drug_col, prot_col


def _check_mapping(path, ids):
    if not os.path.exists(path):
        return None, None, None, None
    with open(path, "rb") as f:
        mapping = pickle.load(f)
    missing = sum(1 for x in ids if x not in mapping)
    ratio = missing / max(len(ids), 1)
    max_idx = max(mapping.values()) if mapping else -1
    return mapping, missing, ratio, max_idx


def _check_feat_size(path):
    if not os.path.exists(path):
        return None
    try:
        arr = np.load(path)
    except Exception:
        return None
    if hasattr(arr, "shape") and len(arr.shape) >= 1:
        return int(arr.shape[0])
    return None


def _maybe_backup(paths, backup_root):
    if not backup_root:
        return
    os.makedirs(backup_root, exist_ok=True)
    for p in paths:
        if os.path.exists(p):
            dst = os.path.join(backup_root, os.path.basename(p))
            shutil.move(p, dst)


def check_and_fix(dataset_dir, fix=False, purge_processed=False, mismatch_threshold=0.01, backup=False):
    train, val, test, val_name = _load_split_csv(dataset_dir)
    drug_ids, prot_ids, drug_col, prot_col = _extract_ids(train, val, test)
    print(f"[INFO] Dataset: {dataset_dir}")
    print(f"[INFO] Split columns: drug={drug_col}, protein={prot_col}, val_file={val_name}")
    print(f"[INFO] ID counts: drugs={len(drug_ids)}, proteins={len(prot_ids)}")

    drug_map = os.path.join(dataset_dir, "drug_to_idx.pkl")
    prot_map = os.path.join(dataset_dir, "protein_to_idx.pkl")
    drug_feat = os.path.join(dataset_dir, "drug_features.npy")
    prot_feat = os.path.join(dataset_dir, "protein_features.npy")

    _, dmiss, dratio, dmax = _check_mapping(drug_map, drug_ids)
    _, pmiss, pratio, pmax = _check_mapping(prot_map, prot_ids)
    dfeat_n = _check_feat_size(drug_feat)
    pfeat_n = _check_feat_size(prot_feat)

    def _fmt_missing(name, miss, ratio):
        if miss is None:
            return f"{name}: MISSING"
        return f"{name}: missing {miss}/{len(drug_ids) if 'drug' in name else len(prot_ids)} ({ratio:.4f})"

    print("[INFO] Cache status:")
    print("  " + _fmt_missing("drug_to_idx.pkl", dmiss, dratio))
    print("  " + _fmt_missing("protein_to_idx.pkl", pmiss, pratio))
    print(f"  drug_features.npy: {'exists' if os.path.exists(drug_feat) else 'MISSING'}")
    print(f"  protein_features.npy: {'exists' if os.path.exists(prot_feat) else 'MISSING'}")
    if dmax is not None:
        print(f"  drug_to_idx max index: {dmax}")
    if pmax is not None:
        print(f"  protein_to_idx max index: {pmax}")
    if dfeat_n is not None:
        print(f"  drug_features rows: {dfeat_n}")
    if pfeat_n is not None:
        print(f"  protein_features rows: {pfeat_n}")

    mismatch = False
    if dmiss is None or pmiss is None:
        mismatch = True
    else:
        if dratio > mismatch_threshold or pratio > mismatch_threshold:
            mismatch = True
    if dmax is not None and dfeat_n is not None and dmax >= dfeat_n:
        mismatch = True
        print(f"[WARN] drug_features rows ({dfeat_n}) <= max drug index ({dmax})")
    if pmax is not None and pfeat_n is not None and pmax >= pfeat_n:
        mismatch = True
        print(f"[WARN] protein_features rows ({pfeat_n}) <= max protein index ({pmax})")

    if not fix:
        if mismatch:
            print("[WARN] Cache mismatch detected (or cache missing).")
        else:
            print("[INFO] Cache mapping looks consistent.")
        return mismatch

    if not mismatch:
        print("[INFO] No mismatch detected; nothing to fix.")
        return False

    stamp = time.strftime("%Y%m%d_%H%M%S")
    backup_root = os.path.join(dataset_dir, f"cache_backup_{stamp}") if backup else None

    to_remove = [drug_map, prot_map, drug_feat, prot_feat]
    if backup_root:
        _maybe_backup(to_remove, backup_root)
    for p in to_remove:
        if os.path.exists(p):
            os.remove(p)
            print(f"[FIX] Removed {p}")

    if purge_processed:
        for proc in ("processed", "processed_atomic"):
            proc_dir = os.path.join(dataset_dir, proc)
            if os.path.isdir(proc_dir):
                shutil.rmtree(proc_dir)
                print(f"[FIX] Removed {proc_dir}")

    print("[INFO] Fix complete. Re-run preprocessing to rebuild caches.")
    return True


def main():
    parser = argparse.ArgumentParser(description="Detect and fix feature cache mismatch.")
    parser.add_argument("--data-root", default="data", help="Data root (default: data)")
    parser.add_argument("--dataset", required=True, help="Dataset name (e.g., gpcr/split-target-inter-pure6)")
    parser.add_argument("--fix", action="store_true", help="Delete mismatched caches")
    parser.add_argument("--purge-processed", action="store_true", help="Also delete processed/ caches")
    parser.add_argument("--threshold", type=float, default=0.01, help="Mismatch ratio threshold")
    parser.add_argument("--backup", action="store_true", help="Move caches to backup folder instead of delete")
    args = parser.parse_args()

    dataset_dir = os.path.abspath(os.path.join(args.data_root, args.dataset))
    check_and_fix(
        dataset_dir,
        fix=args.fix,
        purge_processed=args.purge_processed,
        mismatch_threshold=args.threshold,
        backup=args.backup,
    )


if __name__ == "__main__":
    main()
