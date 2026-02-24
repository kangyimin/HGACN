import argparse
import os
import pickle

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


def _find_col(df, candidates):
    cols = list(df.columns)
    lower = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand in cols:
            return cand
        key = cand.lower()
        if key in lower:
            return lower[key]
    return None


def _resolve_csv_files(dataset_dir="", csv_dir="", csvs=None):
    files = []
    if csvs:
        for path in csvs:
            if path and os.path.exists(path):
                files.append(path)
    root = dataset_dir or csv_dir
    if root:
        for name in ("train.csv", "valid.csv", "val.csv", "test.csv"):
            path = os.path.join(root, name)
            if os.path.exists(path):
                files.append(path)
    out = []
    seen = set()
    for f in files:
        ab = os.path.abspath(f)
        if ab not in seen:
            seen.add(ab)
            out.append(f)
    return out


def collect_target_ids(csv_files):
    target_ids = set()
    for path in csv_files:
        df = pd.read_csv(path)
        col = _find_col(df, ["Target UniProt ID", "target", "target_id", "protein_id"])
        if col is None:
            continue
        target_ids.update(df[col].dropna().astype(str).tolist())
    return target_ids if target_ids else None


def collect_sequences_from_csvs(csv_files):
    unique_seqs = set()
    for path in csv_files:
        df = pd.read_csv(path)
        col = _find_col(df, ["Protein", "protein_sequence", "Target Sequence", "sequence"])
        if col is None:
            print(f"[WARN] No sequence column in {path}. Columns: {df.columns.tolist()}")
            continue
        seqs = df[col].dropna().astype(str).tolist()
        for s in seqs:
            if s and s.lower() != "nan":
                unique_seqs.add(s)
    return sorted(unique_seqs)


def load_sequences_from_idmapping(idmapping_path, target_ids=None):
    if not os.path.exists(idmapping_path):
        raise FileNotFoundError(f"idmapping_target.csv not found: {idmapping_path}")
    df = pd.read_csv(idmapping_path)
    col_id = _find_col(df, ["Target UniProt ID", "uniprot", "target_id", "protein_id"])
    col_seq = _find_col(df, ["Target Sequence", "sequence", "protein_sequence"])
    if col_id is None or col_seq is None:
        raise ValueError(
            f"Required columns not found in {idmapping_path}. Got columns: {list(df.columns)}"
        )
    df[col_id] = df[col_id].astype(str)
    df[col_seq] = df[col_seq].astype(str)
    if target_ids:
        df = df[df[col_id].isin(target_ids)]
    seqs = []
    for s in df[col_seq].tolist():
        if s and s.lower() != "nan":
            seqs.append(s)
    return sorted(set(seqs))


def extract_esm2_embeddings(seqs, model_name, batch_size, device, max_length, local_files_only=False):
    # Import after endpoint env is configured in main().
    from transformers import AutoTokenizer, EsmModel

    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
    model = EsmModel.from_pretrained(model_name, local_files_only=local_files_only).to(device)
    if device.type == "cuda":
        model = model.half()
    model.eval()

    results = {}
    with torch.no_grad():
        for i in tqdm(range(0, len(seqs), batch_size), desc="ESM2"):
            batch = seqs[i : i + batch_size]
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(device)
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state.to(torch.float32)
            for j, seq in enumerate(batch):
                valid_len = int(inputs["attention_mask"][j].sum().item())
                if valid_len <= 2:
                    results[seq] = np.zeros((0, embeddings.shape[-1]), dtype=np.float32)
                    continue
                results[seq] = embeddings[j, 1 : valid_len - 1, :].cpu().numpy()
    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract ESM2 residue embeddings to a pickle dict."
    )
    parser.add_argument("--idmapping_target", type=str, default="",
                        help="Path to idmapping_target.csv.")
    parser.add_argument("--dataset_dir", type=str, default="",
                        help="Dataset dir containing train/valid/val/test CSVs.")
    parser.add_argument("--csv_dir", type=str, default="",
                        help="Legacy alias of dataset_dir.")
    parser.add_argument("--csvs", nargs="*", default=None,
                        help="Optional explicit csv paths.")
    parser.add_argument("--output", type=str, default="",
                        help="Output pickle path.")
    parser.add_argument("--model", type=str, default="facebook/esm2_t33_650M_UR50D",
                        help="HuggingFace model id.")
    parser.add_argument("--model_name", type=str, default="",
                        help="Legacy alias of --model.")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size.")
    parser.add_argument("--max_length", type=int, default=1024,
                        help="Max sequence length.")
    parser.add_argument("--device", type=str, default="cuda",
                        help="cuda or cpu.")
    parser.add_argument("--hf_endpoint", type=str, default="https://hf-mirror.com",
                        help="HF endpoint mirror.")
    parser.add_argument("--local_files_only", action="store_true",
                        help="Load model/tokenizer from local cache/path only (offline).")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.hf_endpoint:
        endpoint = str(args.hf_endpoint).rstrip("/")
        # Set both vars for compatibility across transformers/huggingface_hub versions.
        os.environ["HF_ENDPOINT"] = endpoint
        os.environ["HUGGINGFACE_CO_RESOLVE_ENDPOINT"] = endpoint

    dataset_dir = args.dataset_dir or args.csv_dir
    model_name = args.model_name if args.model_name else args.model
    csv_files = _resolve_csv_files(dataset_dir=dataset_dir, csv_dir=args.csv_dir, csvs=args.csvs)

    if args.idmapping_target:
        target_ids = collect_target_ids(csv_files) if csv_files else None
        seqs = load_sequences_from_idmapping(args.idmapping_target, target_ids=target_ids)
    else:
        if not csv_files:
            raise ValueError(
                "No CSV files found. Provide --idmapping_target or set --dataset_dir/--csv_dir/--csvs."
            )
        seqs = collect_sequences_from_csvs(csv_files)

    if not seqs:
        raise RuntimeError("No sequences found to process.")

    if args.output:
        out_path = args.output
    else:
        if not dataset_dir:
            raise ValueError("--output is required when --dataset_dir/--csv_dir is not provided.")
        out_path = os.path.join(dataset_dir, "protein_esm2_650m.pkl")

    req_device = str(args.device).lower()
    if req_device.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA requested but unavailable, fallback to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(req_device)

    print(f"[INFO] sequences: {len(seqs)}")
    print(f"[INFO] model: {model_name}")
    print(f"[INFO] device: {device}")
    print(f"[INFO] HF_ENDPOINT: {os.environ.get('HF_ENDPOINT', '')}")
    print(f"[INFO] HUGGINGFACE_CO_RESOLVE_ENDPOINT: {os.environ.get('HUGGINGFACE_CO_RESOLVE_ENDPOINT', '')}")
    print(f"[INFO] output: {out_path}")
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    results = extract_esm2_embeddings(
        seqs=seqs,
        model_name=model_name,
        batch_size=int(args.batch_size),
        device=device,
        max_length=int(args.max_length),
        local_files_only=bool(args.local_files_only),
    )
    with open(out_path, "wb") as f:
        pickle.dump(results, f)
    print(f"[INFO] saved: {out_path}")


if __name__ == "__main__":
    main()
