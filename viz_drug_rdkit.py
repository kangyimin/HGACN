import os


def render_drug_atom_importance(smiles_or_mol, atom_scores, out_png, topk=20):
    try:
        from rdkit import Chem
        from rdkit.Chem import rdMolDraw2D
    except Exception as e:
        raise RuntimeError(f"RDKit not available: {e}")

    mol = smiles_or_mol
    if not hasattr(mol, "GetAtoms"):
        mol = Chem.MolFromSmiles(str(smiles_or_mol))
    if mol is None:
        raise ValueError("Failed to parse SMILES/molecule for RDKit rendering.")

    scores = atom_scores
    if scores is None:
        scores = []
    scores = list(scores)
    n_atoms = mol.GetNumAtoms()
    if len(scores) < n_atoms:
        scores = scores + [0.0] * (n_atoms - len(scores))
    scores = scores[:n_atoms]
    max_score = max(scores) if scores else 1.0
    if max_score <= 0:
        max_score = 1.0

    ranked = sorted(range(n_atoms), key=lambda i: scores[i], reverse=True)
    top_atoms = ranked[: min(topk, n_atoms)]
    highlight_atoms = top_atoms
    highlight_colors = {}
    for idx in highlight_atoms:
        s = float(scores[idx]) / max_score
        # red->white
        highlight_colors[idx] = (1.0, 1.0 - s, 1.0 - s)

    drawer = rdMolDraw2D.MolDraw2DCairo(500, 400)
    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer,
        mol,
        highlightAtoms=highlight_atoms,
        highlightAtomColors=highlight_colors,
    )
    drawer.FinishDrawing()
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    with open(out_png, "wb") as f:
        f.write(drawer.GetDrawingText())

    top_info = []
    for idx in top_atoms:
        atom = mol.GetAtomWithIdx(int(idx))
        neighbors = [nbr.GetIdx() for nbr in atom.GetNeighbors()]
        top_info.append(
            {
                "atom_idx": int(idx),
                "score": float(scores[idx]),
                "symbol": atom.GetSymbol(),
                "neighbors": neighbors,
            }
        )
    return top_info
