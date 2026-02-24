import csv


def _parse_residue_key(line):
    chain = line[21].strip() or "A"
    resseq = line[22:26].strip()
    icode = line[26].strip()
    resname = line[17:20].strip()
    key = (chain, resseq, icode, resname)
    return key


def write_residue_scores_to_pdb(pdb_in, pdb_out, residue_scores, chain=None, mode="bfactor"):
    residue_scores = list(residue_scores) if residue_scores is not None else []
    residues = []
    res_index = {}
    with open(pdb_in, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            key = _parse_residue_key(line)
            if chain and key[0] != chain:
                continue
            if key not in res_index:
                res_index[key] = len(residues)
                residues.append(key)

    max_len = len(residues)
    if len(residue_scores) < max_len:
        residue_scores = residue_scores + [0.0] * (max_len - len(residue_scores))
    residue_scores = residue_scores[:max_len]

    with open(pdb_in, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    out_lines = []
    for line in lines:
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            out_lines.append(line)
            continue
        key = _parse_residue_key(line)
        if chain and key[0] != chain:
            out_lines.append(line)
            continue
        idx = res_index.get(key, None)
        score = residue_scores[idx] if idx is not None else 0.0
        if mode == "bfactor":
            b = f"{float(score):6.2f}"
            line = line[:60] + b + line[66:]
        out_lines.append(line)

    with open(pdb_out, "w", encoding="utf-8") as f:
        f.writelines(out_lines)

    return residues, residue_scores


def export_topk_residues_csv(out_csv, residues, residue_scores, topk=20):
    ranked = sorted(range(len(residue_scores)), key=lambda i: residue_scores[i], reverse=True)
    top_idx = ranked[: min(topk, len(residue_scores))]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "score", "chain", "resi", "icode", "resname"])
        for r, idx in enumerate(top_idx, start=1):
            chain, resi, icode, resname = residues[idx]
            writer.writerow([r, float(residue_scores[idx]), chain, resi, icode, resname])
