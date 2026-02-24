"""Data preprocessing utilities for HGACN: feature extraction, attention alignment, and hypergraph construction."""

import os
import glob
import time
import hashlib
import numpy as np
import pandas as pd
from scipy import sparse
import pickle
import json
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, rdPartialCharges, rdchem
from rdkit.DataStructs import ConvertToNumpyArray
import warnings
import math
from sklearn.decomposition import PCA

# Ignore all warnings to make the output cleaner
warnings.filterwarnings("ignore")

_HAS_CALC_HEAVY = hasattr(rdMolDescriptors, "CalcHeavyAtomCount")

ATOM_TYPES = [
    "C", "N", "O", "S", "P", "F", "Cl", "Br", "I", "B", "Si", "Se", "Na", "K", "Ca", "Zn", "Fe", "Mg", "Al", "Hg",
    "Unknown"
]
RES_TYPES = list("ACDEFGHIKLMNPQRSTVWY") + ["X"]
DEGREE_LIST = [0, 1, 2, 3, 4, 5]
HCOUNT_LIST = [0, 1, 2, 3, 4]
VALENCE_LIST = [0, 1, 2, 3, 4, 5, 6]
HYBRID_LIST = [
    rdchem.HybridizationType.SP,
    rdchem.HybridizationType.SP2,
    rdchem.HybridizationType.SP3,
    rdchem.HybridizationType.SP3D,
    rdchem.HybridizationType.SP3D2,
]
CHIRAL_LIST = [
    rdchem.ChiralType.CHI_UNSPECIFIED,
    rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    rdchem.ChiralType.CHI_OTHER,
]
ATOM_FEAT_VERSION = 2
DRUG_DESC_VERSION = 3
DRUG_DESC_IMPUTE = "train_mean"
FEATURE_MISSING_STATS = {}

_ATOM_TYPE_TO_IDX = {v: i for i, v in enumerate(ATOM_TYPES)}
_DEGREE_TO_IDX = {v: i for i, v in enumerate(DEGREE_LIST)}
_HCOUNT_TO_IDX = {v: i for i, v in enumerate(HCOUNT_LIST)}
_VALENCE_TO_IDX = {v: i for i, v in enumerate(VALENCE_LIST)}
_HYBRID_TO_IDX = {v: i for i, v in enumerate(HYBRID_LIST)}
_CHIRAL_TO_IDX = {v: i for i, v in enumerate(CHIRAL_LIST)}
_RES_TYPE_TO_IDX = {v: i for i, v in enumerate(RES_TYPES)}
_RES_ONE_HOT = np.eye(len(RES_TYPES), dtype=np.float32)

# Residue physchem features (lightweight, no 3D).
# Order: [hydrophobicity, polarity, charge, aromatic, hbond_donor, hbond_acceptor,
#         mol_weight, is_small, is_proline, is_glycine]
_RES_PHYS_RAW = {
    "A": {"hydro": 1.8, "polarity": 8.1, "charge": 0.0, "aromatic": 0.0, "donor": 0.0, "acceptor": 0.0, "mw": 89.1, "small": 1.0, "pro": 0.0, "gly": 0.0},
    "C": {"hydro": 2.5, "polarity": 5.5, "charge": 0.0, "aromatic": 0.0, "donor": 0.0, "acceptor": 1.0, "mw": 121.2, "small": 1.0, "pro": 0.0, "gly": 0.0},
    "D": {"hydro": -3.5, "polarity": 13.0, "charge": -1.0, "aromatic": 0.0, "donor": 0.0, "acceptor": 1.0, "mw": 133.1, "small": 0.0, "pro": 0.0, "gly": 0.0},
    "E": {"hydro": -3.5, "polarity": 12.3, "charge": -1.0, "aromatic": 0.0, "donor": 0.0, "acceptor": 1.0, "mw": 147.1, "small": 0.0, "pro": 0.0, "gly": 0.0},
    "F": {"hydro": 2.8, "polarity": 5.2, "charge": 0.0, "aromatic": 1.0, "donor": 0.0, "acceptor": 0.0, "mw": 165.2, "small": 0.0, "pro": 0.0, "gly": 0.0},
    "G": {"hydro": -0.4, "polarity": 9.0, "charge": 0.0, "aromatic": 0.0, "donor": 0.0, "acceptor": 0.0, "mw": 75.1, "small": 1.0, "pro": 0.0, "gly": 1.0},
    "H": {"hydro": -3.2, "polarity": 10.4, "charge": 0.5, "aromatic": 0.0, "donor": 1.0, "acceptor": 1.0, "mw": 155.2, "small": 0.0, "pro": 0.0, "gly": 0.0},
    "I": {"hydro": 4.5, "polarity": 5.2, "charge": 0.0, "aromatic": 0.0, "donor": 0.0, "acceptor": 0.0, "mw": 131.2, "small": 0.0, "pro": 0.0, "gly": 0.0},
    "K": {"hydro": -3.9, "polarity": 11.3, "charge": 1.0, "aromatic": 0.0, "donor": 1.0, "acceptor": 0.0, "mw": 146.2, "small": 0.0, "pro": 0.0, "gly": 0.0},
    "L": {"hydro": 3.8, "polarity": 4.9, "charge": 0.0, "aromatic": 0.0, "donor": 0.0, "acceptor": 0.0, "mw": 131.2, "small": 0.0, "pro": 0.0, "gly": 0.0},
    "M": {"hydro": 1.9, "polarity": 5.7, "charge": 0.0, "aromatic": 0.0, "donor": 0.0, "acceptor": 1.0, "mw": 149.2, "small": 0.0, "pro": 0.0, "gly": 0.0},
    "N": {"hydro": -3.5, "polarity": 11.6, "charge": 0.0, "aromatic": 0.0, "donor": 1.0, "acceptor": 1.0, "mw": 132.1, "small": 0.0, "pro": 0.0, "gly": 0.0},
    "P": {"hydro": -1.6, "polarity": 8.0, "charge": 0.0, "aromatic": 0.0, "donor": 0.0, "acceptor": 0.0, "mw": 115.1, "small": 1.0, "pro": 1.0, "gly": 0.0},
    "Q": {"hydro": -3.5, "polarity": 10.5, "charge": 0.0, "aromatic": 0.0, "donor": 1.0, "acceptor": 1.0, "mw": 146.2, "small": 0.0, "pro": 0.0, "gly": 0.0},
    "R": {"hydro": -4.5, "polarity": 10.5, "charge": 1.0, "aromatic": 0.0, "donor": 1.0, "acceptor": 0.0, "mw": 174.2, "small": 0.0, "pro": 0.0, "gly": 0.0},
    "S": {"hydro": -0.8, "polarity": 9.2, "charge": 0.0, "aromatic": 0.0, "donor": 1.0, "acceptor": 1.0, "mw": 105.1, "small": 1.0, "pro": 0.0, "gly": 0.0},
    "T": {"hydro": -0.7, "polarity": 8.6, "charge": 0.0, "aromatic": 0.0, "donor": 1.0, "acceptor": 1.0, "mw": 119.1, "small": 1.0, "pro": 0.0, "gly": 0.0},
    "V": {"hydro": 4.2, "polarity": 5.9, "charge": 0.0, "aromatic": 0.0, "donor": 0.0, "acceptor": 0.0, "mw": 117.1, "small": 1.0, "pro": 0.0, "gly": 0.0},
    "W": {"hydro": -0.9, "polarity": 5.4, "charge": 0.0, "aromatic": 1.0, "donor": 1.0, "acceptor": 0.0, "mw": 204.2, "small": 0.0, "pro": 0.0, "gly": 0.0},
    "Y": {"hydro": -1.3, "polarity": 6.2, "charge": 0.0, "aromatic": 1.0, "donor": 1.0, "acceptor": 1.0, "mw": 181.2, "small": 0.0, "pro": 0.0, "gly": 0.0},
    "X": {"hydro": 0.0, "polarity": 0.0, "charge": 0.0, "aromatic": 0.0, "donor": 0.0, "acceptor": 0.0, "mw": 0.0, "small": 0.0, "pro": 0.0, "gly": 0.0},
}

_HYDRO_MAX = 4.5
_HYDRO_MIN = -4.5
_POLAR_MAX = 13.0
_POLAR_MIN = 4.9
_MW_MAX = 204.2
_MW_MIN = 75.1


def get_residue_physchem(aa):
    """Return normalized physicochemical features for a residue code."""
    aa = aa if aa in _RES_PHYS_RAW else "X"
    v = _RES_PHYS_RAW[aa]
    hydro = (v["hydro"] - _HYDRO_MIN) / max(_HYDRO_MAX - _HYDRO_MIN, 1e-6)
    polarity = (v["polarity"] - _POLAR_MIN) / max(_POLAR_MAX - _POLAR_MIN, 1e-6)
    mw = (v["mw"] - _MW_MIN) / max(_MW_MAX - _MW_MIN, 1e-6)
    return np.asarray(
        [
            hydro,
            polarity,
            v["charge"],
            v["aromatic"],
            v["donor"],
            v["acceptor"],
            mw,
            v["small"],
            v["pro"],
            v["gly"],
        ],
        dtype=np.float32,
    )

_ATOM_TYPE_OFFSET = 0
_DEGREE_OFFSET = _ATOM_TYPE_OFFSET + len(ATOM_TYPES)
_HCOUNT_OFFSET = _DEGREE_OFFSET + len(DEGREE_LIST)
_VALENCE_OFFSET = _HCOUNT_OFFSET + len(HCOUNT_LIST)
_HYBRID_OFFSET = _VALENCE_OFFSET + len(VALENCE_LIST)
_CHIRAL_OFFSET = _HYBRID_OFFSET + len(HYBRID_LIST)
_AROM_OFFSET = _CHIRAL_OFFSET + len(CHIRAL_LIST)
_RING_OFFSET = _AROM_OFFSET + 2
_FORMAL_OFFSET = _RING_OFFSET + 4
_ATOM_FEAT_DIM_BASE = _FORMAL_OFFSET + 1
_ATOM_FEAT_DIM_GAST = _ATOM_FEAT_DIM_BASE + 1
_GAST_OFFSET = _FORMAL_OFFSET + 1

_ATOM_NONE_FEAT_BASE = np.zeros(_ATOM_FEAT_DIM_BASE, dtype=np.float32)
_ATOM_NONE_FEAT_BASE[_ATOM_TYPE_OFFSET + _ATOM_TYPE_TO_IDX["Unknown"]] = 1.0
_ATOM_NONE_FEAT_BASE[_DEGREE_OFFSET + _DEGREE_TO_IDX[0]] = 1.0
_ATOM_NONE_FEAT_BASE[_HCOUNT_OFFSET + _HCOUNT_TO_IDX[0]] = 1.0
_ATOM_NONE_FEAT_BASE[_VALENCE_OFFSET + _VALENCE_TO_IDX[0]] = 1.0
_ATOM_NONE_FEAT_BASE[_HYBRID_OFFSET + _HYBRID_TO_IDX.get(HYBRID_LIST[0], 0)] = 1.0
_ATOM_NONE_FEAT_BASE[_CHIRAL_OFFSET + _CHIRAL_TO_IDX.get(CHIRAL_LIST[0], 0)] = 1.0
_ATOM_NONE_FEAT_GAST = np.zeros(_ATOM_FEAT_DIM_GAST, dtype=np.float32)
_ATOM_NONE_FEAT_GAST[:_ATOM_FEAT_DIM_BASE] = _ATOM_NONE_FEAT_BASE

def one_hot(items, vocab):
    """Create one-hot vector and map unknown token to the last bucket."""
    vec = np.zeros(len(vocab), dtype=np.float32)
    idx = vocab.index(items) if items in vocab else (len(vocab) - 1)
    vec[idx] = 1.0
    return vec

def _clip_float(val, lo, hi):
    """Clamp a scalar to [lo, hi] and return float."""
    return max(lo, min(hi, float(val)))

def _heavy_atom_count(mol):
    """Return heavy-atom count with RDKit-version fallback."""
    if mol is None:
        return 0
    if _HAS_CALC_HEAVY:
        return rdMolDescriptors.CalcHeavyAtomCount(mol)
    return mol.GetNumHeavyAtoms()

def _atom_feature_dim(use_gasteiger=False):
    """Return atom feature dimension based on optional Gasteiger charge."""
    return _ATOM_FEAT_DIM_GAST if use_gasteiger else _ATOM_FEAT_DIM_BASE

def atom_rdkit_features(atom, use_gasteiger=False):
    """Build atom-level RDKit feature vector for one atom."""
    if atom is None:
        return (_ATOM_NONE_FEAT_GAST if use_gasteiger else _ATOM_NONE_FEAT_BASE).copy()
    dim = _ATOM_FEAT_DIM_GAST if use_gasteiger else _ATOM_FEAT_DIM_BASE
    feats = np.zeros(dim, dtype=np.float32)
    sym_idx = _ATOM_TYPE_TO_IDX.get(atom.GetSymbol(), _ATOM_TYPE_TO_IDX["Unknown"])
    feats[_ATOM_TYPE_OFFSET + sym_idx] = 1.0
    degree = min(atom.GetDegree(), DEGREE_LIST[-1])
    feats[_DEGREE_OFFSET + _DEGREE_TO_IDX.get(degree, len(DEGREE_LIST) - 1)] = 1.0
    hcount = min(atom.GetTotalNumHs(), HCOUNT_LIST[-1])
    feats[_HCOUNT_OFFSET + _HCOUNT_TO_IDX.get(hcount, len(HCOUNT_LIST) - 1)] = 1.0
    valence = min(atom.GetImplicitValence(), VALENCE_LIST[-1])
    feats[_VALENCE_OFFSET + _VALENCE_TO_IDX.get(valence, len(VALENCE_LIST) - 1)] = 1.0
    hybrid_idx = _HYBRID_TO_IDX.get(atom.GetHybridization(), len(HYBRID_LIST) - 1)
    feats[_HYBRID_OFFSET + hybrid_idx] = 1.0
    chiral_idx = _CHIRAL_TO_IDX.get(atom.GetChiralTag(), len(CHIRAL_LIST) - 1)
    feats[_CHIRAL_OFFSET + chiral_idx] = 1.0
    feats[_AROM_OFFSET] = float(atom.GetIsAromatic())
    feats[_AROM_OFFSET + 1] = float(atom.IsInRing())
    feats[_RING_OFFSET] = float(atom.IsInRingSize(3))
    feats[_RING_OFFSET + 1] = float(atom.IsInRingSize(4))
    feats[_RING_OFFSET + 2] = float(atom.IsInRingSize(5))
    feats[_RING_OFFSET + 3] = float(atom.IsInRingSize(6))
    feats[_FORMAL_OFFSET] = _clip_float(atom.GetFormalCharge(), -2.0, 2.0) / 2.0
    if use_gasteiger:
        try:
            charge = float(atom.GetProp("_GasteigerCharge"))
        except Exception:
            charge = 0.0
        if not np.isfinite(charge):
            charge = 0.0
        feats[_GAST_OFFSET] = _clip_float(charge, -1.0, 1.0)
    return feats

def get_drug_descriptors(smiles_list, return_missing=False):
    """Compute molecular descriptors for a list of SMILES strings."""
    desc = []
    missing = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            desc.append([np.nan] * 10)
            missing.append(True)
            continue
        vals = [
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            rdMolDescriptors.CalcTPSA(mol),
            rdMolDescriptors.CalcNumHBD(mol),
            rdMolDescriptors.CalcNumHBA(mol),
            rdMolDescriptors.CalcNumRotatableBonds(mol),
            rdMolDescriptors.CalcNumRings(mol),
            rdMolDescriptors.CalcNumAromaticRings(mol),
            rdMolDescriptors.CalcFractionCSP3(mol),
            float(_heavy_atom_count(mol)),
        ]
        desc.append(vals)
        missing.append(False)
    desc = np.asarray(desc, dtype=np.float32)
    nan_mask = ~np.isfinite(desc).all(axis=1)
    if nan_mask.any():
        missing = np.logical_or(missing, nan_mask)
    desc = np.nan_to_num(desc, nan=np.nan, posinf=np.nan, neginf=np.nan)
    missing = np.asarray(missing, dtype=bool)
    if return_missing:
        return desc, missing
    return desc

def _load_embedding(path, expected_rows, name):
    """Load embedding array from disk and validate expected shape."""
    if not path or not os.path.exists(path):
        return None
    try:
        data = np.load(path)
        if isinstance(data, np.lib.npyio.NpzFile):
            if "emb" in data.files:
                emb = data["emb"]
            else:
                emb = data[data.files[0]]
        else:
            emb = data
    except Exception:
        print(f"[WARN] Failed to load {name} embedding from {path}")
        return None
    if emb.ndim != 2 or emb.shape[0] != expected_rows:
        print(f"[WARN] {name} embedding shape mismatch: {emb.shape} vs expected {expected_rows}")
        return None
    return emb.astype(np.float32)

def normalize_adj_safe(adj, add_self_loop=True):
    """
    Safe normalization with optional self-loop:
    A_hat = A + I, then D^{-1/2} * A_hat * D^{-1/2}
    """
    if sparse.isspmatrix(adj):
        A = adj.tocsr().astype(np.float64)
        if add_self_loop:
            n = A.shape[0]
            A = A + sparse.eye(n, dtype=A.dtype, format="csr")
        deg = np.array(A.sum(axis=1)).flatten()
        deg_inv_sqrt = np.zeros_like(deg)
        nz = deg > 0
        deg_inv_sqrt[nz] = 1.0 / np.sqrt(deg[nz])
        D_inv_sqrt = sparse.diags(deg_inv_sqrt)
        return D_inv_sqrt.dot(A).dot(D_inv_sqrt)
    A = np.asarray(adj, dtype=np.float64)
    if add_self_loop:
        n = A.shape[0]
        A = A + np.eye(n, dtype=A.dtype)
    deg = A.sum(axis=1)
    deg_inv_sqrt = np.zeros_like(deg)
    nz = deg > 0
    deg_inv_sqrt[nz] = 1.0 / np.sqrt(deg[nz])
    return (deg_inv_sqrt[:, None] * A) * deg_inv_sqrt[None, :]

def get_drug_features(smiles_list, n_bits=2048):
    """Build Morgan fingerprints for SMILES."""
    features = []
    radius = 2
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol:
            morgan_fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            bits = np.asarray(morgan_fp.ToList(), dtype=np.float32)
            features.append(bits)
        else:
            features.append(np.zeros(n_bits, dtype=np.float32))
            print(f"Invalid SMILES: {str(smiles)[:20]}...")
    return np.asarray(features, dtype=np.float32).reshape(-1, n_bits)

def get_protein_features(seq_list, max_len=1000):
    """One-hot encode protein sequences up to max_len."""
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    aa_to_idx = {aa: i for i, aa in enumerate(amino_acids)}
    features = []
    for seq in seq_list:
        seq = str(seq)[:max_len]
        feature = np.zeros((max_len, len(amino_acids)), dtype=np.float32)
        for i, aa in enumerate(seq):
            idx = aa_to_idx.get(aa)
            if idx is not None:
                feature[i, idx] = 1.0
        features.append(feature.flatten())
    return np.asarray(features, dtype=np.float32)


def build_protein_global_features(seq_list, use_physchem=True):
    """
    Build compact protein-level features (composition + mean physchem + log length).
    This is a lightweight fallback when precomputed protein features are missing.
    """
    seq_list = list(seq_list)
    comp_dim = len(RES_TYPES)
    physchem_dim = len(get_residue_physchem("X"))
    extra_dim = 1
    out_dim = comp_dim + (physchem_dim if use_physchem else 0) + extra_dim
    feats = np.zeros((len(seq_list), out_dim), dtype=np.float32)
    for i, seq in enumerate(seq_list):
        seq = str(seq) if seq is not None else ""
        counts = np.zeros(comp_dim, dtype=np.float32)
        if seq:
            for aa in seq:
                idx = _RES_TYPE_TO_IDX.get(aa, _RES_TYPE_TO_IDX["X"])
                counts[idx] += 1.0
            counts = counts / max(len(seq), 1)
        phys = None
        if use_physchem:
            if seq:
                acc = np.zeros(physchem_dim, dtype=np.float32)
                for aa in seq:
                    acc += get_residue_physchem(aa if aa in _RES_PHYS_RAW else "X")
                phys = acc / max(len(seq), 1)
            else:
                phys = get_residue_physchem("X")
        log_len = np.log1p(len(seq))
        if use_physchem:
            feats[i] = np.concatenate([counts, phys, np.asarray([log_len], dtype=np.float32)], axis=0)
        else:
            feats[i] = np.concatenate([counts, np.asarray([log_len], dtype=np.float32)], axis=0)
    return feats

def build_atom_nodes(drug_ids, use_gasteiger=False):
    """
    Build atom-level nodes from SMILES.
    Returns atom_features, atom_to_drug_idx, drug_to_atom_ids.
    """
    atom_features = []
    atom_to_drug = []
    drug_to_atom_ids = {}
    expected_dim = _atom_feature_dim(use_gasteiger=use_gasteiger)
    warn_dim = False
    for drug_idx, smiles in enumerate(drug_ids):
        mol = Chem.MolFromSmiles(str(smiles))
        atom_ids = []
        if mol is None or mol.GetNumAtoms() == 0:
            feat = atom_rdkit_features(None, use_gasteiger=use_gasteiger)
            if feat.size != expected_dim:
                if not warn_dim:
                    print(f"[WARN] Atom feature dim mismatch: {feat.size} vs {expected_dim}.")
                    warn_dim = True
                fixed = np.zeros(expected_dim, dtype=np.float32)
                if feat.size:
                    fixed[:min(feat.size, expected_dim)] = feat[:min(feat.size, expected_dim)]
                feat = fixed
            atom_features.append(feat)
            atom_to_drug.append(drug_idx)
            atom_ids.append(len(atom_features) - 1)
            drug_to_atom_ids[smiles] = atom_ids
            continue
        if use_gasteiger:
            try:
                rdPartialCharges.ComputeGasteigerCharges(mol)
            except Exception:
                pass
        for atom in mol.GetAtoms():
            feat = atom_rdkit_features(atom, use_gasteiger=use_gasteiger)
            if feat.size != expected_dim:
                if not warn_dim:
                    print(f"[WARN] Atom feature dim mismatch: {feat.size} vs {expected_dim}.")
                    warn_dim = True
                fixed = np.zeros(expected_dim, dtype=np.float32)
                if feat.size:
                    fixed[:min(feat.size, expected_dim)] = feat[:min(feat.size, expected_dim)]
                feat = fixed
            atom_features.append(feat)
            atom_to_drug.append(drug_idx)
            atom_ids.append(len(atom_features) - 1)
        drug_to_atom_ids[smiles] = atom_ids
    return np.asarray(atom_features, dtype=np.float32), np.asarray(atom_to_drug, dtype=np.int64), drug_to_atom_ids

def build_residue_nodes(protein_ids):
    """
    Build residue-level nodes from sequences.
    Returns residue_features, residue_to_protein_idx, protein_to_residue_ids.
    """
    residue_features = []
    residue_to_prot = []
    protein_to_res_ids = {}
    expected_dim = len(RES_TYPES)
    warn_dim = False
    for prot_idx, seq in enumerate(protein_ids):
        res_ids = []
        seq = str(seq)
        if len(seq) == 0:
            feat = _RES_ONE_HOT[_RES_TYPE_TO_IDX["X"]]
            if feat.size != expected_dim:
                if not warn_dim:
                    print(f"[WARN] Residue feature dim mismatch: {feat.size} vs {expected_dim}.")
                    warn_dim = True
                fixed = np.zeros(expected_dim, dtype=np.float32)
                if feat.size:
                    fixed[:min(feat.size, expected_dim)] = feat[:min(feat.size, expected_dim)]
                feat = fixed
            residue_features.append(feat)
            residue_to_prot.append(prot_idx)
            res_ids.append(len(residue_features) - 1)
            protein_to_res_ids[seq] = res_ids
            continue
        for aa in seq:
            aa = aa if aa in RES_TYPES else "X"
            feat = _RES_ONE_HOT[_RES_TYPE_TO_IDX.get(aa, _RES_TYPE_TO_IDX["X"])]
            if feat.size != expected_dim:
                if not warn_dim:
                    print(f"[WARN] Residue feature dim mismatch: {feat.size} vs {expected_dim}.")
                    warn_dim = True
                fixed = np.zeros(expected_dim, dtype=np.float32)
                if feat.size:
                    fixed[:min(feat.size, expected_dim)] = feat[:min(feat.size, expected_dim)]
                feat = fixed
            residue_features.append(feat)
            residue_to_prot.append(prot_idx)
            res_ids.append(len(residue_features) - 1)
        protein_to_res_ids[seq] = res_ids
    return np.asarray(residue_features, dtype=np.float32), np.asarray(residue_to_prot, dtype=np.int64), protein_to_res_ids


def build_residue_physchem_features(protein_ids):
    """
    Build physchem features aligned with residue nodes from sequences.
    Returns (num_residues, physchem_dim).
    """
    physchem_features = []
    for seq in protein_ids:
        seq = str(seq)
        if len(seq) == 0:
            physchem_features.append(get_residue_physchem("X"))
            continue
        for aa in seq:
            aa = aa if aa in RES_TYPES else "X"
            physchem_features.append(get_residue_physchem(aa))
    return np.asarray(physchem_features, dtype=np.float32)


def load_esm2_dict(path):
    """Load per-protein ESM2 embeddings from a serialized file."""
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
    except Exception:
        print(f"[WARN] Failed to load ESM2 features from {path}")
        return None
    if not isinstance(data, dict):
        print(f"[WARN] ESM2 feature file is not a dict: {path}")
        return None
    return data


def _align_esm_array(seq_len, emb, esm_special_tokens="auto", esm_strict=False):
    """Align token-level ESM array to residue-level sequence length."""
    if esm_special_tokens == "bos":
        esm_special_tokens = "bos_only"
    if esm_special_tokens == "eos":
        esm_special_tokens = "eos_only"
    if emb is None:
        return None, None, "missing"
    if emb.ndim != 2:
        return None, None, "dim_mismatch"
    t_len = int(emb.shape[0])
    if t_len == seq_len + 2:
        start = 1
        mode = "bos_eos"
    elif t_len == seq_len:
        start = 0
        mode = "none"
    elif t_len == seq_len + 1:
        if esm_special_tokens in ("auto", "bos_only"):
            start = 1
            mode = "bos_only"
        elif esm_special_tokens in ("eos_only", "none"):
            start = 0
            mode = "eos_only"
        else:
            if esm_strict:
                raise ValueError("ESM alignment failed: L+1 length with bos_eos setting.")
            return None, None, "mismatch"
    else:
        if esm_strict:
            raise ValueError("ESM alignment failed: length mismatch.")
        return None, None, "mismatch"
    aligned = emb[start:start + seq_len]
    if aligned.shape[0] != seq_len:
        if esm_strict:
            raise ValueError("ESM alignment failed: unexpected length after slicing.")
        return None, None, "mismatch"
    return aligned, start, mode


def debug_esm_alignment(seq=None, emb=None):
    """Run sanity checks for sequence-to-ESM alignment."""
    if seq is None:
        seq = "ACDEG"
    if emb is None:
        feat_dim = 4
        emb = np.arange((len(seq) + 2) * feat_dim, dtype=np.float32).reshape(len(seq) + 2, feat_dim)
    else:
        emb = np.asarray(emb, dtype=np.float32)
    aligned, start, mode = _align_esm_array(len(seq), emb, esm_special_tokens="bos_eos", esm_strict=True)
    ok = aligned is not None and start == 1 and np.allclose(aligned[0], emb[1])
    if ok:
        print("[INFO] debug_esm_alignment: PASS")
    else:
        print("[WARN] debug_esm_alignment: FAIL")


def build_residue_features_from_esm(
    esm_dict,
    protein_ids,
    residue_to_protein,
    residue_orig_pos,
    feat_dim=1280,
    fallback="onehot_only",
    esm_special_tokens="auto",
    esm_norm="per_protein_zscore",
    esm_strict=False,
    debug_samples=5,
    return_meta=False,
):
    """Build residue features from ESM with robust fallback rules."""
    num_nodes = int(residue_to_protein.size)
    features = np.zeros((num_nodes, feat_dim), dtype=np.float32)
    proteins_total = len(protein_ids)
    proteins_missing = 0
    proteins_mismatch = 0
    proteins_with_esm = 0
    residues_missing = 0
    cnt_lp2 = 0
    cnt_l = 0
    cnt_lp1 = 0
    cnt_mismatch = 0
    cnt_lp1_bos = 0
    cnt_lp1_eos = 0
    warned_auto = False
    warned = set()
    aligned_cache = {}
    align_mode = {}
    align_offset = {}
    emb_len_cache = {}
    mismatch_records = []
    mismatch_global_fallback = 0
    sum_vec = None
    sumsq_vec = None
    count_total = 0

    prot_missing_mask = np.zeros(proteins_total, dtype=bool)
    prot_mismatch_mask = np.zeros(proteins_total, dtype=bool)
    for prot_idx in range(proteins_total):
        seq = protein_ids[prot_idx] if 0 <= prot_idx < proteins_total else None
        if not seq:
            proteins_missing += 1
            aligned_cache[prot_idx] = None
            align_mode[prot_idx] = "missing"
            prot_missing_mask[prot_idx] = True
            continue
        emb = esm_dict.get(seq)
        if emb is None:
            proteins_missing += 1
            aligned_cache[prot_idx] = None
            align_mode[prot_idx] = "missing"
            prot_missing_mask[prot_idx] = True
            if seq not in warned:
                print("[WARN] Missing ESM2 for protein sequence; fallback used.")
                warned.add(seq)
            continue
        emb = np.asarray(emb, dtype=np.float32)
        emb_len_cache[prot_idx] = int(emb.shape[0])
        seq_len = len(seq)
        if emb.shape[1] != feat_dim:
            cnt_mismatch += 1
            aligned_cache[prot_idx] = None
            align_mode[prot_idx] = "dim_mismatch"
            proteins_mismatch += 1
            prot_mismatch_mask[prot_idx] = True
            mismatch_records.append(
                {
                    "prot_idx": int(prot_idx),
                    "protein_id": seq,
                    "seq_len": int(seq_len),
                    "emb_len": int(emb.shape[0]),
                    "mode": "dim_mismatch",
                }
            )
            if seq not in warned:
                print("[WARN] ESM2 dim mismatch; fallback used.")
                warned.add(seq)
            continue
        if emb.shape[0] == seq_len + 2:
            cnt_lp2 += 1
        elif emb.shape[0] == seq_len:
            cnt_l += 1
        elif emb.shape[0] == seq_len + 1:
            cnt_lp1 += 1
            if esm_special_tokens in ("auto", "bos_only"):
                cnt_lp1_bos += 1
                if esm_special_tokens == "auto" and not warned_auto:
                    print("[WARN] ESM2 length L+1 detected; assuming BOS-only. Set --esm_special_tokens to override.")
                    warned_auto = True
            elif esm_special_tokens in ("eos_only", "none"):
                cnt_lp1_eos += 1
        else:
            cnt_mismatch += 1
        aligned, start, mode = _align_esm_array(seq_len, emb, esm_special_tokens, esm_strict)
        if aligned is None:
            proteins_mismatch += 1
            prot_mismatch_mask[prot_idx] = True
            mismatch_records.append(
                {
                    "prot_idx": int(prot_idx),
                    "protein_id": seq,
                    "seq_len": int(seq_len),
                    "emb_len": int(emb.shape[0]),
                    "mode": "mismatch",
                }
            )
            aligned_cache[prot_idx] = None
            align_mode[prot_idx] = "mismatch"
            if seq not in warned:
                print("[WARN] ESM2 length mismatch; fallback used.")
                warned.add(seq)
            continue
        if esm_norm == "per_dim_global":
            if sum_vec is None:
                sum_vec = aligned.sum(axis=0, dtype=np.float64)
                sumsq_vec = (aligned.astype(np.float64) ** 2).sum(axis=0)
            else:
                sum_vec += aligned.sum(axis=0, dtype=np.float64)
                sumsq_vec += (aligned.astype(np.float64) ** 2).sum(axis=0)
            count_total += int(aligned.shape[0])
        elif esm_norm == "per_protein_zscore":
            mean = aligned.mean(axis=0, keepdims=True)
            std = aligned.std(axis=0, keepdims=True)
            aligned = (aligned - mean) / np.maximum(std, 1e-6)
            aligned = np.clip(aligned, -5.0, 5.0)
        aligned_cache[prot_idx] = aligned
        align_mode[prot_idx] = mode
        align_offset[prot_idx] = start
        proteins_with_esm += 1

    if esm_norm == "per_dim_global" and count_total > 0 and sum_vec is not None and sumsq_vec is not None:
        mean = sum_vec / max(count_total, 1)
        var = sumsq_vec / max(count_total, 1) - mean ** 2
        std = np.sqrt(np.maximum(var, 1e-6))
        for prot_idx, aligned in aligned_cache.items():
            if aligned is None:
                continue
            aligned = (aligned - mean) / std
            aligned = np.clip(aligned, -5.0, 5.0)
            aligned_cache[prot_idx] = aligned.astype(np.float32)
        print(
            f"[INFO] ESM2 global norm: mean_range=({mean.min():.4f},{mean.max():.4f}), "
            f"std_range=({std.min():.4f},{std.max():.4f})"
        )

    valid_mask = np.zeros(num_nodes, dtype=bool)
    for i in range(num_nodes):
        prot_idx = int(residue_to_protein[i])
        pos = int(residue_orig_pos[i])
        emb = aligned_cache.get(prot_idx)
        if emb is None:
            residues_missing += 1
            continue
        if emb.shape[0] == 1:
            features[i] = emb[0]
            valid_mask[i] = True
            continue
        if pos < 0 or pos >= emb.shape[0]:
            residues_missing += 1
            continue
        features[i] = emb[pos]
        valid_mask[i] = True

    if residues_missing > 0:
        print(f"[WARN] ESM2 fallback applied for {residues_missing} residues.")
    if fallback == "mean":
        if valid_mask.any():
            mean_vec = features[valid_mask].mean(axis=0, keepdims=True)
        else:
            mean_vec = np.zeros((1, feat_dim), dtype=np.float32)
        if (~valid_mask).any():
            features[~valid_mask] = mean_vec
    elif fallback == "uniform_noise":
        if (~valid_mask).any():
            rng = np.random.default_rng(0)
            noise = rng.uniform(-0.1, 0.1, size=(int((~valid_mask).sum()), feat_dim)).astype(np.float32)
            features[~valid_mask] = noise
    else:
        # onehot_only / zeros: keep zeros for ESM features
        pass

    proteins_unreliable = int(proteins_missing + proteins_mismatch)
    print(
        f"[INFO] ESM2 proteins: total={proteins_total}, "
        f"with_esm={proteins_with_esm}, missing={proteins_missing}, mismatch={proteins_mismatch}, "
        f"unreliable={proteins_unreliable}"
    )
    print(
        f"[INFO] ESM2 residues: total={num_nodes}, missing={residues_missing}"
    )
    print(
        "[INFO] ESM2 length stats: "
        f"L+2={cnt_lp2}, L={cnt_l}, L+1={cnt_lp1} "
        f"(bos_only={cnt_lp1_bos}, eos_only={cnt_lp1_eos}), mismatch={cnt_mismatch}"
    )
    if mismatch_global_fallback:
        print(f"[INFO] ESM2 mismatch global-mean fallback proteins={mismatch_global_fallback}")

    if proteins_total > 0 and debug_samples > 0:
        rng = np.random.default_rng(0)
        sample_idx = rng.choice(np.arange(proteins_total), size=min(debug_samples, proteins_total), replace=False)
        for prot_idx in sample_idx:
            seq = protein_ids[prot_idx]
            seq_len = len(seq) if seq else -1
            t_len = emb_len_cache.get(prot_idx, -1)
            mode = align_mode.get(prot_idx, "missing")
            offset = align_offset.get(prot_idx, None)
            res_pos = residue_orig_pos[residue_to_protein == prot_idx]
            res_pos = res_pos[res_pos >= 0]
            res_pos = res_pos[:3] if res_pos.size else np.asarray([], dtype=np.int64)
            mapping = []
            for p in res_pos.tolist():
                if offset is None:
                    mapping.append((p, None))
                else:
                    mapping.append((p, p + int(offset)))
            print(
                f"[INFO] ESM2 sample prot={prot_idx} L={seq_len} T={t_len} mode={mode} map={mapping}"
            )

    prot_unreliable_mask = (prot_missing_mask | prot_mismatch_mask)
    proteins_unreliable = int(prot_unreliable_mask.sum())
    if return_meta:
        meta = {
            "mismatch_records": mismatch_records,
            "mismatch_count": int(len(mismatch_records)),
            "missing_count": int(proteins_missing),
            "unreliable_count": int(proteins_unreliable),
            "mismatch_global_fallback_count": int(mismatch_global_fallback),
        }
        return (
            features.astype(np.float32),
            prot_missing_mask.astype(np.uint8),
            prot_unreliable_mask.astype(np.uint8),
            meta,
        )
    return features.astype(np.float32)

def load_psichic_attention(attn_path):
    """
    Load PSICHIC attention outputs.
    Expected formats:
      - dict[(protein_seq, drug_smiles)] -> {"atom_scores": np.ndarray, "residue_scores": np.ndarray}
      - dict["protein||smiles"] -> same payload
    """
    if not attn_path or not os.path.exists(attn_path):
        return {}
    with open(attn_path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict):
        return data
    raise ValueError(f"Unsupported PSICHIC attention format in {attn_path}")

count = 0
def get_pair_attention(attn_db, protein_seq, drug_smiles):
    """Fetch atom/residue attention scores for one protein-drug pair."""
    global count
    key = (protein_seq, drug_smiles)
    if key in attn_db:
        payload = attn_db[key]
        atom = payload.get("atom_scores", payload.get("atom_attn"))
        residue = payload.get("residue_scores", payload.get("residue_attn"))
        return atom, residue
    key2 = f"{protein_seq}||{drug_smiles}"
    if key2 in attn_db:
        payload = attn_db[key2]
        atom = payload.get("atom_scores", payload.get("atom_attn"))
        residue = payload.get("residue_scores", payload.get("residue_attn"))
        return atom, residue
    if count < 3:
        prot_show = str(protein_seq)[:10] if protein_seq is not None else "None"
        smi_show = str(drug_smiles)[:10] if drug_smiles is not None else "None"
        print(f"DEBUG: 鍖归厤澶辫触! Seq[:10]: {prot_show}, Smi[:10]: {smi_show}")
        count += 1
    return None, None


def _get_attn_signature(attn_path):
    """Build cache signature for attention source path."""
    if not attn_path or not os.path.exists(attn_path):
        return {"attn_path": "", "attn_mtime": 0, "attn_size": 0}
    return {
        "attn_path": os.path.abspath(attn_path),
        "attn_mtime": int(os.path.getmtime(attn_path)),
        "attn_size": int(os.path.getsize(attn_path)),
    }

def _hash_file(path, algo="md5", chunk_size=1 << 20):
    """Compute deterministic file hash; returns empty string on failure."""
    if not path or not os.path.exists(path):
        return ""
    try:
        h = hashlib.new(algo)
        with open(path, "rb") as f:
            while True:
                buf = f.read(int(chunk_size))
                if not buf:
                    break
                h.update(buf)
        return h.hexdigest()
    except Exception:
        return ""

def _get_file_signature(path, prefix, include_hash=False):
    """Build file signature from path/mtime/size, with optional content hash."""
    if not path or not os.path.exists(path):
        sig = {f"{prefix}_path": "", f"{prefix}_mtime": 0, f"{prefix}_size": 0}
        if include_hash:
            sig[f"{prefix}_hash"] = ""
        return sig
    sig = {
        f"{prefix}_path": os.path.abspath(path),
        f"{prefix}_mtime": int(os.path.getmtime(path)),
        f"{prefix}_size": int(os.path.getsize(path)),
    }
    if include_hash:
        sig[f"{prefix}_hash"] = _hash_file(path, algo="md5")
    return sig


def _find_col_from_candidates(df, candidates):
    """Find first matching column by exact or case-insensitive name."""
    cols = list(df.columns)
    lower = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand in cols:
            return cand
        key = str(cand).lower()
        if key in lower:
            return lower[key]
    return None


def _standardize_dti_df_columns(df, name="train"):
    """Map heterogeneous DTI CSV headers to standard columns."""
    col_map = {
        "protein_sequence": [
            "protein_sequence", "Protein", "protein", "sequence", "Sequence",
            "Target Sequence", "target_sequence", "Target UniProt ID", "uniprot", "UniProt", "target",
        ],
        "drug_smiles": [
            "drug_smiles", "Ligand", "ligand", "SMILES", "smiles", "Ligand SMILES",
            "compound_smiles", "drug",
        ],
        "label": [
            "label", "Label", "classification_label", "class", "Class", "y", "Y", "interaction", "Interaction",
        ],
    }
    renamed = {}
    for std_col, candidates in col_map.items():
        if std_col == "protein_sequence" and "Protein" in df.columns:
            found = "Protein"
        elif std_col == "drug_smiles" and "Ligand" in df.columns:
            found = "Ligand"
        else:
            found = _find_col_from_candidates(df, candidates)
        if found is None:
            raise ValueError(
                f"{name}.csv missing required column for '{std_col}'. "
                f"Available columns: {list(df.columns)}"
            )
        if found != std_col:
            renamed[found] = std_col
    if renamed:
        df = df.rename(columns=renamed)
        print(f"[INFO] {name}.csv column remap: {renamed}")
    return df


def _prepare_split_and_mappings(dataset_dir):
    """Build split edge arrays and id mappings under dataset_dir/processed."""
    processed_dir = os.path.join(dataset_dir, "processed")
    os.makedirs(processed_dir, exist_ok=True)
    train_path = os.path.join(dataset_dir, "train.csv")
    val_path = os.path.join(dataset_dir, "val.csv")
    test_path = os.path.join(dataset_dir, "test.csv")

    train_data = _standardize_dti_df_columns(pd.read_csv(train_path, header=0), name="train")
    val_data = _standardize_dti_df_columns(pd.read_csv(val_path, header=0), name="val")
    test_data = _standardize_dti_df_columns(pd.read_csv(test_path, header=0), name="test")

    for df in (train_data, val_data, test_data):
        df["protein_sequence"] = df["protein_sequence"].astype(str).str.strip()
        df["drug_smiles"] = df["drug_smiles"].astype(str).str.strip()
        df["label"] = pd.to_numeric(df["label"], errors="coerce")
        df.dropna(subset=["label", "drug_smiles", "protein_sequence"], inplace=True)
        df["label"] = df["label"].apply(lambda x: 1 if x > 0 else 0).astype(int)

    if train_data["label"].nunique() < 2:
        raise ValueError("train.csv must contain both labels")

    all_data = pd.concat([train_data, val_data, test_data], ignore_index=True)
    all_data = all_data.dropna(subset=["drug_smiles", "protein_sequence"])
    drug_ids = np.unique(all_data["drug_smiles"])
    protein_ids = np.unique(all_data["protein_sequence"])
    drug2idx = {d: i for i, d in enumerate(drug_ids)}
    prot2idx = {p: i for i, p in enumerate(protein_ids)}

    with open(os.path.join(dataset_dir, "drug_to_idx.pkl"), "wb") as f:
        pickle.dump(drug2idx, f)
    with open(os.path.join(dataset_dir, "protein_to_idx.pkl"), "wb") as f:
        pickle.dump(prot2idx, f)

    def _build_edges(df):
        edges = np.array(
            [
                [drug2idx[row["drug_smiles"]], prot2idx[row["protein_sequence"]], row["label"]]
                for _, row in df.iterrows()
                if row["drug_smiles"] in drug2idx and row["protein_sequence"] in prot2idx
            ],
            dtype=np.int64,
        )
        if edges.size == 0:
            return np.empty((0, 3), dtype=np.int64)
        return edges

    train_edges = _build_edges(train_data)
    val_edges = _build_edges(val_data)
    test_edges = _build_edges(test_data)

    if train_edges.size == 0:
        print(f"[WARN] train_edges is empty after preprocessing. train_data shape={train_data.shape}")
    for name, edges in (("train", train_edges), ("val", val_edges), ("test", test_edges)):
        if edges.size and not np.all(np.isin(np.unique(edges[:, 2]), [0, 1])):
            raise ValueError(f"{name}_edges labels invalid: {np.unique(edges[:, 2])}")

    np.save(os.path.join(processed_dir, "train_edges.npy"), train_edges)
    np.save(os.path.join(processed_dir, "val_edges.npy"), val_edges)
    np.save(os.path.join(processed_dir, "test_edges.npy"), test_edges)
    return train_edges, val_edges, test_edges


def normalize_incidence(scores, method="sum"):
    """Normalize incidence weights using the selected strategy."""
    scores = np.asarray(scores, dtype=np.float32)
    if scores.size == 0:
        return scores
    if method == "softmax":
        scores = scores - np.max(scores)
        exp_scores = np.exp(scores)
        denom = float(exp_scores.sum())
        if denom <= 0:
            return np.full_like(scores, 1.0 / max(scores.size, 1))
        return exp_scores / denom
    denom = float(scores.sum())
    if denom <= 0:
        return np.full_like(scores, 1.0 / max(scores.size, 1))
    return scores / denom


def _l2_normalize(x, eps=1e-6):
    """L2-normalize rows with epsilon protection."""
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, eps)


def mean_pool_by_index(x, index, num_groups):
    """Mean-pool features by integer group index."""
    x = np.asarray(x, dtype=np.float32)
    index = np.asarray(index, dtype=np.int64)
    if x.size == 0 or num_groups <= 0:
        return np.zeros((num_groups, x.shape[1] if x.ndim == 2 else 0), dtype=np.float32)
    out = np.zeros((num_groups, x.shape[1]), dtype=np.float32)
    counts = np.zeros((num_groups,), dtype=np.float32)
    np.add.at(out, index, x)
    np.add.at(counts, index, 1.0)
    counts = np.maximum(counts, 1.0).reshape(-1, 1)
    return out / counts


def build_knn_graph(
    vecs,
    k=20,
    metric="cosine",
    symmetric=True,
    weight_temp=0.1,
    chunk_size=512,
    filter_idx=None,
    key_idx=None,
):
    """Construct KNN edges as (query -> neighbor)."""
    vecs = np.asarray(vecs, dtype=np.float32)
    n = vecs.shape[0]
    if n == 0 or k <= 0:
        return np.empty((2, 0), dtype=np.int64), np.empty((0,), dtype=np.float32)
    if metric != "cosine":
        raise ValueError(f"Unsupported knn metric: {metric}")
    vecs_norm = _l2_normalize(vecs)
    if key_idx is None:
        key_ids = np.arange(n, dtype=np.int64)
        key_vecs = vecs_norm
        key_pos = None
    else:
        key_ids = np.asarray(key_idx, dtype=np.int64)
        key_vecs = vecs_norm[key_ids]
        key_pos = {int(v): i for i, v in enumerate(key_ids.tolist())}
    if filter_idx is None:
        rows = np.arange(n, dtype=np.int64)
    else:
        rows = np.asarray(filter_idx, dtype=np.int64)
    edge_src = []
    edge_dst = []
    edge_w = []
    for start in range(0, rows.size, chunk_size):
        row_idx = rows[start:start + chunk_size]
        sims = vecs_norm[row_idx] @ key_vecs.T
        for i, r in enumerate(row_idx):
            sim_row = sims[i]
            if key_pos is None:
                sim_row[r] = -np.inf
            else:
                pos = key_pos.get(int(r))
                if pos is not None:
                    sim_row[pos] = -np.inf
            if k >= n:
                idx = np.argsort(sim_row)[::-1]
            else:
                idx = np.argpartition(sim_row, -k)[-k:]
                idx = idx[np.argsort(sim_row[idx])[::-1]]
            edge_src.extend([int(r)] * len(idx))
            edge_dst.extend(key_ids[idx].tolist())
            if weight_temp and weight_temp > 0:
                w = np.exp(sim_row[idx] / float(weight_temp))
                w = w / max(w.sum(), 1e-12)
            else:
                w = sim_row[idx]
            edge_w.extend(w.tolist())
    if symmetric and edge_src:
        edge_src_sym = list(edge_dst)
        edge_dst_sym = list(edge_src)
        edge_w_sym = list(edge_w)
        edge_src.extend(edge_src_sym)
        edge_dst.extend(edge_dst_sym)
        edge_w.extend(edge_w_sym)
    edge_index = np.vstack([np.asarray(edge_src, dtype=np.int64), np.asarray(edge_dst, dtype=np.int64)])
    edge_weight = np.asarray(edge_w, dtype=np.float32)
    return edge_index, edge_weight


def align_scores_to_length(scores, target_len, fill=1.0):
    """Pad or trim scores to match the requested target length."""
    target_len = int(target_len)
    if target_len <= 0:
        return np.zeros((0,), dtype=np.float32)
    if scores is None:
        return np.full((target_len,), float(fill), dtype=np.float32)
    s = np.asarray(scores, dtype=np.float32).reshape(-1)
    if s.size == target_len:
        return s
    if s.size > target_len:
        return s[:target_len]
    out = np.full((target_len,), float(fill), dtype=np.float32)
    out[:s.size] = s
    return out


def pack_ragged_int(list_of_arrays):
    """Pack ragged integer arrays into ptr/nodes representation."""
    ptr = np.zeros(len(list_of_arrays) + 1, dtype=np.int64)
    total = 0
    for i, arr in enumerate(list_of_arrays):
        arr = np.asarray(arr, dtype=np.int64)
        total += int(arr.size)
        ptr[i + 1] = total
    data = np.empty(total, dtype=np.int64)
    offset = 0
    for arr in list_of_arrays:
        arr = np.asarray(arr, dtype=np.int64)
        if arr.size:
            data[offset:offset + arr.size] = arr
            offset += int(arr.size)
    return ptr, data


def save_ragged_int(path, list_of_arrays):
    """Save ragged integer arrays to compressed NPZ."""
    ptr, data = pack_ragged_int(list_of_arrays)
    np.savez(path, ptr=ptr, nodes=data)


def load_ragged_int(path):
    """Load ragged integer arrays from compressed NPZ."""
    data = np.load(path)
    return data["ptr"].astype(np.int64), data["nodes"].astype(np.int64)


def topk_indices(values, k):
    """Return indices of top-k values in descending order."""
    v = np.asarray(values, dtype=np.float32).reshape(-1)
    if v.size == 0:
        return np.empty((0,), dtype=np.int64)
    k = int(k)
    if k <= 0:
        return np.empty((0,), dtype=np.int64)
    k = min(k, v.size)
    # argpartition: indices are not sorted, so sort to keep position stability.
    idx = np.argpartition(-v, k - 1)[:k]
    idx = np.sort(idx).astype(np.int64)
    return idx


def apply_top_k_indices(ids, scores, top_k, fallback_mode="uniform"):
    """Select ids by top-k scores with safe fallback policy."""
    if top_k is None or top_k <= 0 or len(ids) <= top_k:
        return ids, scores
    if scores is None or len(scores) != len(ids):
        ids = _fallback_select(np.asarray(ids), top_k, fallback_mode)
        scores = np.ones(len(ids), dtype=np.float32)
        return ids, scores
    scores = np.asarray(scores, dtype=np.float32)
    finite_scores = scores[np.isfinite(scores)]
    if finite_scores.size == 0 or np.allclose(finite_scores, finite_scores[0]):
        ids = _fallback_select(np.asarray(ids), top_k, fallback_mode)
        scores = np.ones(len(ids), dtype=np.float32)
        return ids, scores
    top_idx = np.argpartition(scores, -top_k)[-top_k:]
    top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
    ids = [ids[i] for i in top_idx]
    scores = scores[top_idx]
    return ids, scores


def apply_topk_randk_indices(
    ids,
    scores,
    top_k,
    rand_k,
    rand_seed,
    rand_weight_mode="floor_prior",
    prior_floor=1e-4,
    fallback_mode="uniform",
):
    """Select top-k plus random-k ids for exploration."""
    ids = np.asarray(ids, dtype=np.int64)
    if ids.size == 0:
        return ids, np.asarray([], dtype=np.float32)
    if scores is None or len(scores) != len(ids):
        scores = np.ones(ids.size, dtype=np.float32)
        scores_valid = False
    else:
        scores = np.asarray(scores, dtype=np.float32)
        finite_scores = scores[np.isfinite(scores)]
        scores_valid = finite_scores.size > 0 and not np.allclose(finite_scores, finite_scores[0])

    if top_k is None or top_k <= 0 or ids.size <= top_k:
        top_mask = np.ones(ids.size, dtype=bool)
    else:
        if not scores_valid:
            top_ids = _fallback_select(ids, top_k, fallback_mode)
            top_mask = np.isin(ids, np.asarray(top_ids, dtype=np.int64))
        else:
            top_idx = np.argpartition(scores, -top_k)[-top_k:]
            top_mask = np.zeros(ids.size, dtype=bool)
            top_mask[top_idx] = True

    top_ids = ids[top_mask]
    top_scores = scores[top_mask] if top_ids.size else np.asarray([], dtype=np.float32)

    rand_k = int(rand_k) if rand_k is not None else 0
    if rand_k <= 0:
        return top_ids, top_scores

    remain_mask = ~top_mask
    remain_ids = ids[remain_mask]
    if remain_ids.size == 0:
        return top_ids, top_scores

    rng = np.random.RandomState(int(rand_seed))
    choose_k = min(rand_k, remain_ids.size)
    rand_sel = rng.choice(remain_ids.size, size=choose_k, replace=False)
    rand_ids = remain_ids[rand_sel]
    if rand_weight_mode == "uniform":
        rand_scores = np.ones(rand_ids.size, dtype=np.float32)
    else:
        remain_scores = scores[remain_mask][rand_sel]
        if rand_weight_mode == "prior":
            rand_scores = remain_scores
        else:
            rand_scores = np.maximum(remain_scores, float(prior_floor))

    ids_out = np.concatenate([top_ids, rand_ids], axis=0)
    scores_out = np.concatenate([top_scores, rand_scores], axis=0)
    return ids_out, scores_out


def _read_keep_env(var_name, default_val=None):
    """Read keep-ratio override from environment variable."""
    raw = os.environ.get(var_name)
    if raw is None:
        return default_val
    try:
        val = int(raw)
    except ValueError:
        return default_val
    return val if val > 0 else None


def _fallback_select(ids, k, mode):
    """Fallback selector when attention scores are unavailable."""
    if k is None or len(ids) <= k:
        return list(ids)
    if mode == "random":
        return list(np.random.choice(ids, size=k, replace=False))
    # uniform: deterministic
    return list(ids)[:k]


def _prune_nodes_by_attention(ids_by_group, attn_scores, keep_k, fallback_mode):
    """Prune grouped nodes according to attention keep policy."""
    if keep_k is None:
        return np.ones(len(attn_scores), dtype=bool)
    keep_mask = np.zeros(len(attn_scores), dtype=bool)
    for ids in ids_by_group:
        if ids.size == 0:
            continue
        if ids.size <= keep_k:
            keep_mask[ids] = True
            continue
        scores = np.asarray(attn_scores[ids], dtype=np.float32)
        finite_scores = scores[np.isfinite(scores)]
        if finite_scores.size == 0:
            keep_ids = _fallback_select(ids, keep_k, fallback_mode)
        elif np.allclose(finite_scores, finite_scores[0]):
            keep_ids = _fallback_select(ids, keep_k, fallback_mode)
        else:
            top_idx = np.argpartition(scores, -keep_k)[-keep_k:]
            top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
            keep_ids = [ids[i] for i in top_idx]
        keep_mask[keep_ids] = True
    return keep_mask



def load_or_build_features(items, feat_func, feat_name, dataset_dir, train_ids=None, impute_missing=False,
                           log_missing=True, add_missing_flag=False):
    """Load cached features or build and cache them."""
    items = list(items)
    feat_path = os.path.join(dataset_dir, f"{feat_name}_features.npy")
    map_path = os.path.join(dataset_dir, f"{feat_name}_to_idx.pkl")

    if os.path.exists(feat_path) and os.path.exists(map_path):
        with open(map_path, "rb") as f:
            mapping = pickle.load(f)
        feats_all = np.load(feat_path)
        train_mean = None
        if impute_missing:
            if train_ids:
                train_idx = [mapping.get(it) for it in train_ids if mapping.get(it) is not None]
                train_idx = [i for i in train_idx if i is not None and i < feats_all.shape[0]]
            else:
                train_idx = []
            if train_idx:
                train_mean = feats_all[train_idx].mean(axis=0)
            elif feats_all.size:
                train_mean = feats_all.mean(axis=0)
        feat_list, missing = [], 0
        missing_mask = []
        for it in items:
            idx = mapping.get(it)
            if idx is None or idx >= len(feats_all):
                missing += 1
                missing_mask.append(True)
                if impute_missing and train_mean is not None:
                    feat_list.append(train_mean.astype(np.float32))
                else:
                    feat_list.append(np.zeros(feats_all.shape[1], dtype=np.float32))
            else:
                feat_list.append(feats_all[idx])
                missing_mask.append(False)
        ratio = missing / max(len(items), 1)
        FEATURE_MISSING_STATS[feat_name] = {
            "missing_count": int(missing),
            "missing_ratio": float(ratio),
            "impute": bool(impute_missing and train_mean is not None),
            "train_only": bool(train_ids),
            "add_missing_flag": bool(add_missing_flag),
        }
        if missing > 0 and log_missing:
            if impute_missing and train_mean is not None:
                print(
                    f"[WARN] {feat_name}_features missing {missing} rows ({ratio:.4f}); "
                    "imputed with train mean (train_only=1)"
                )
            else:
                print(
                    f"[WARN] {feat_name}_features missing {missing} rows ({ratio:.4f}); "
                    "filled with zero vectors"
                )
        features = np.asarray(feat_list, dtype=np.float32)
        if add_missing_flag:
            missing_flag = np.asarray(missing_mask, dtype=np.float32).reshape(-1, 1)
            features = np.hstack([features, missing_flag])
        return features

    feats = feat_func(items).astype(np.float32)
    mapping = {it: i for i, it in enumerate(items)}
    np.save(feat_path, feats)
    with open(map_path, "wb") as f:
        pickle.dump(mapping, f)
    print(f"[INFO] Cached {feat_name} features to {feat_path}")
    FEATURE_MISSING_STATS[feat_name] = {
        "missing_count": 0,
        "missing_ratio": 0.0,
        "impute": False,
        "train_only": bool(train_ids),
        "add_missing_flag": bool(add_missing_flag),
    }
    return feats


def load_precomputed_features(items, feat_name, dataset_dir):
    """Load precomputed features; raise if missing."""
    items = list(items)
    feat_path = os.path.join(dataset_dir, f"{feat_name}_features.npy")
    map_path = os.path.join(dataset_dir, f"{feat_name}_to_idx.pkl")
    if not (os.path.exists(feat_path) and os.path.exists(map_path)):
        raise FileNotFoundError(
            f"Missing precomputed {feat_name} features: {feat_path} and {map_path}"
        )
    with open(map_path, "rb") as f:
        mapping = pickle.load(f)
    feats_all = np.load(feat_path)
    feat_list, missing = [], 0
    for it in items:
        idx = mapping.get(it)
        if idx is None or idx >= len(feats_all):
            missing += 1
            feat_list.append(np.zeros(feats_all.shape[1], dtype=np.float32))
        else:
            feat_list.append(feats_all[idx])
    if missing > 0:
        print(f"[WARN] {feat_name} missing {missing} rows; filled with zero vectors")
    return np.asarray(feat_list, dtype=np.float32)


def load_and_construct_hypergraphs_atomic(
    dataset_name,
    data_root,
    psichic_attention_path=None,
    add_self_loop=True,
    protein_feat_mode="concat",
    esm_special_tokens="auto",
    esm_norm="per_protein_zscore",
    esm_fallback="onehot_only",
    esm_strict=False,
    use_physchem_feat=True,
    reuse_cache=False,
    prune_strategy_tag=None,
    atom_topk=None,
    res_topk=None,
    atom_randk=0,
    res_randk=0,
    randk_seed=None,
    randk_weight_mode="floor_prior",
    prior_floor=1e-4,
    use_knn_graph=False,
    knn_setting="inductive",
    cold_deg_th_drug=2,
    cold_deg_th_prot=2,
    cold_mode="th",
    cold_q=0.1,
    cold_q_drug=None,
    cold_q_prot=None,
    drug_knn_k=20,
    prot_knn_k=20,
    knn_metric="cosine",
    knn_symmetric=True,
    knn_weight_temp=0.1,
):
    """
    Atomic-level hypergraph:
    - nodes are ligand atoms and protein residues
    - hyperedges are training interactions
    - PSICHIC attention scores can be used as incidence weights and node features
    """
    dataset_dir = os.path.join(data_root, dataset_name)
    processed_dir = os.path.join(dataset_dir, "processed_atomic")
    os.makedirs(processed_dir, exist_ok=True)
    atom_top_k = int(os.environ.get("HGACN_ATOM_TOPK", "6"))
    residue_top_k = int(os.environ.get("HGACN_RES_TOPK", "12"))
    if atom_topk is not None and int(atom_topk) > 0:
        atom_top_k = int(atom_topk)
    if res_topk is not None and int(res_topk) > 0:
        residue_top_k = int(res_topk)
    atom_keep = _read_keep_env("HGACN_ATOM_KEEP", default_val=16)
    residue_keep = _read_keep_env("HGACN_RES_KEEP", default_val=64)
    prune_fallback = os.environ.get("HGACN_PRUNE_FALLBACK", "uniform").strip().lower()
    atom_randk = max(int(atom_randk or 0), 0)
    residue_randk = max(int(res_randk or 0), 0)
    if randk_seed is None:
        randk_seed = 2025
    randk_weight_mode = (randk_weight_mode or "floor_prior").strip().lower()
    if randk_weight_mode not in ("prior", "uniform", "floor_prior"):
        randk_weight_mode = "floor_prior"
    prior_floor = max(float(prior_floor), 0.0)
    prune_strategy = "train_avg_topk"
    if not prune_strategy_tag:
        prune_strategy_tag = prune_strategy
    protein_feat_mode = str(protein_feat_mode or "concat").strip().lower()
    if protein_feat_mode not in ("onehot", "esm2", "concat"):
        protein_feat_mode = "concat"
    attn_sig = _get_attn_signature(psichic_attention_path)
    esm_path = os.environ.get("HGACN_PROT_ESM2", "").strip()
    if not esm_path:
        cand = os.path.join(dataset_dir, "protein_esm2_650m.pkl")
        if os.path.exists(cand):
            esm_path = cand
    esm_sig = _get_file_signature(esm_path, "prot_esm2")
    esm_dict = load_esm2_dict(esm_path) if esm_sig["prot_esm2_path"] else None
    if protein_feat_mode != "onehot" and esm_dict is None:
        print(f"[WARN] protein_feat_mode={protein_feat_mode} but ESM2 dict not found; falling back to onehot.")
        protein_feat_mode = "onehot"
    use_gasteiger = os.environ.get("HGACN_USE_GASTEIGER", "0").strip() == "1"
    use_drug_desc = os.environ.get("HGACN_USE_DRUG_DESC", "1").strip() != "0"
    drug_gnn_path = os.environ.get("HGACN_DRUG_GNN_EMB", "").strip()
    if not drug_gnn_path:
        cand = os.path.join(dataset_dir, "drug_gnn_emb.npy")
        if os.path.exists(cand):
            drug_gnn_path = cand
    atom_gnn_path = os.environ.get("HGACN_ATOM_GNN_EMB", "").strip()
    if not atom_gnn_path:
        cand = os.path.join(dataset_dir, "atom_gnn_emb.npy")
        if os.path.exists(cand):
            atom_gnn_path = cand
    drug_gnn_sig = _get_file_signature(drug_gnn_path, "drug_gnn")
    atom_gnn_sig = _get_file_signature(atom_gnn_path, "atom_gnn")
    if atom_top_k <= 0:
        atom_top_k = None
    if residue_top_k <= 0:
        residue_top_k = None
    if prune_fallback not in ("uniform", "random"):
        prune_fallback = "uniform"
    print(f"[INFO] Atomic top-k: atom={atom_top_k}, residue={residue_top_k}")
    if atom_randk or residue_randk:
        print(
            f"[INFO] Atomic rand-k: atom={atom_randk}, residue={residue_randk}, "
            f"mode={randk_weight_mode}, prior_floor={prior_floor}"
        )
    if atom_keep or residue_keep:
        print(f"[INFO] Atomic keep: atom={atom_keep}, residue={residue_keep}, fallback={prune_fallback}")
    print(f"[INFO] Prune strategy: {prune_strategy_tag}")
    if use_gasteiger:
        print("[INFO] Atom features: RDKit + Gasteiger")
    if use_drug_desc:
        print("[INFO] Drug descriptors: enabled")
    if drug_gnn_sig["drug_gnn_path"]:
        print(f"[INFO] Drug GNN emb: {drug_gnn_sig['drug_gnn_path']}")
    if atom_gnn_sig["atom_gnn_path"]:
        print(f"[INFO] Atom GNN emb: {atom_gnn_sig['atom_gnn_path']}")
    if esm_sig["prot_esm2_path"]:
        print(f"[INFO] Protein ESM2: {esm_sig['prot_esm2_path']}")
    print(
        f"[INFO] Protein feature mode: {protein_feat_mode}, esm_norm={esm_norm}, "
        f"esm_special_tokens={esm_special_tokens}, esm_fallback={esm_fallback}, "
        f"physchem={'on' if use_physchem_feat else 'off'}"
    )

    cache_files = {
        "G_atom": os.path.join(processed_dir, "G_atom.npz"),
        "G_residue": os.path.join(processed_dir, "G_residue.npz"),
        "H_atom": os.path.join(processed_dir, "H_atom.npz"),
        "H_residue": os.path.join(processed_dir, "H_residue.npz"),
        "features_atom": os.path.join(processed_dir, "features_atom.npy"),
        "features_residue": os.path.join(processed_dir, "features_residue.npy"),
        "atom_to_drug": os.path.join(processed_dir, "atom_to_drug.npy"),
        "residue_to_protein": os.path.join(processed_dir, "residue_to_protein.npy"),
        "atom_attn": os.path.join(processed_dir, "atom_attn.npy"),
        "residue_attn": os.path.join(processed_dir, "residue_attn.npy"),
        "atom_orig_pos": os.path.join(processed_dir, "atom_orig_pos.npy"),
        "residue_orig_pos": os.path.join(processed_dir, "residue_orig_pos.npy"),
        "drug_atom_nodes": os.path.join(processed_dir, "drug_atom_nodes.npz"),
        "prot_res_nodes": os.path.join(processed_dir, "prot_res_nodes.npz"),
        "prot_esm_missing": os.path.join(processed_dir, "prot_esm_missing.npy"),
        "prot_esm_unreliable": os.path.join(processed_dir, "prot_esm_unreliable.npy"),
        "drug_knn_edge_index": os.path.join(processed_dir, "drug_knn_edge_index.npy"),
        "drug_knn_edge_weight": os.path.join(processed_dir, "drug_knn_edge_weight.npy"),
        "prot_knn_edge_index": os.path.join(processed_dir, "prot_knn_edge_index.npy"),
        "prot_knn_edge_weight": os.path.join(processed_dir, "prot_knn_edge_weight.npy"),
        "drug_deg": os.path.join(processed_dir, "drug_deg.npy"),
        "prot_deg": os.path.join(processed_dir, "prot_deg.npy"),
    }
    deg_train_drug_path = os.path.join(processed_dir, "deg_train_drug.npy")
    deg_train_prot_path = os.path.join(processed_dir, "deg_train_prot.npy")
    deg_train_meta_path = os.path.join(processed_dir, "deg_train_meta.json")

    split_train_path = os.path.join(dataset_dir, "processed", "train_edges.npy")
    split_val_path = os.path.join(dataset_dir, "processed", "val_edges.npy")
    split_test_path = os.path.join(dataset_dir, "processed", "test_edges.npy")
    required_mapping = [
        os.path.join(dataset_dir, "drug_to_idx.pkl"),
        os.path.join(dataset_dir, "protein_to_idx.pkl"),
        split_train_path,
        split_val_path,
        split_test_path,
    ]
    if not all(os.path.exists(p) for p in required_mapping):
        _prepare_split_and_mappings(dataset_dir)
    split_train_sig = _get_file_signature(split_train_path, "split_train", include_hash=True)
    split_val_sig = _get_file_signature(split_val_path, "split_val", include_hash=True)
    split_test_sig = _get_file_signature(split_test_path, "split_test", include_hash=True)

    config_path = os.path.join(processed_dir, "config.json")
    base_keys = [
        "G_atom", "G_residue", "H_atom", "H_residue",
        "features_atom", "features_residue",
        "atom_to_drug", "residue_to_protein",
        "atom_attn", "residue_attn",
        "atom_orig_pos", "residue_orig_pos",
        "drug_atom_nodes", "prot_res_nodes",
        "drug_deg", "prot_deg", "prot_esm_missing", "prot_esm_unreliable",
    ]
    knn_keys = [
        "drug_knn_edge_index", "drug_knn_edge_weight",
        "prot_knn_edge_index", "prot_knn_edge_weight",
    ]
    cache_ready = all(os.path.exists(cache_files[k]) for k in base_keys)
    if cache_ready:
        if not os.path.exists(config_path):
            cache_ready = False
        else:
            try:
                with open(config_path, "r") as f:
                    cfg = json.load(f)
            except Exception:
                cfg = {}
            cache_mismatch = (
                cfg.get("atom_keep", 0) != (atom_keep or 0)
                or cfg.get("residue_keep", 0) != (residue_keep or 0)
                or cfg.get("atom_top_k", 0) != (atom_top_k or 0)
                or cfg.get("residue_top_k", 0) != (residue_top_k or 0)
                or cfg.get("atom_randk", 0) != (atom_randk or 0)
                or cfg.get("res_randk", 0) != (residue_randk or 0)
                or int(cfg.get("randk_seed", 0)) != int(randk_seed or 0)
                or cfg.get("randk_weight_mode", "floor_prior") != randk_weight_mode
                or float(cfg.get("prior_floor", 0.0)) != float(prior_floor)
                or cfg.get("prune_fallback", "uniform") != prune_fallback
                or cfg.get("prune_strategy", "") != prune_strategy_tag
                or cfg.get("protein_feat_mode", "") != protein_feat_mode
                or cfg.get("esm_special_tokens", "auto") != esm_special_tokens
                or cfg.get("esm_norm", "per_protein_zscore") != esm_norm
                or cfg.get("esm_fallback", "onehot_only") != esm_fallback
                or bool(cfg.get("use_physchem_feat", True)) != bool(use_physchem_feat)
                or bool(cfg.get("use_knn_graph", False)) != bool(use_knn_graph)
                or cfg.get("knn_setting", "inductive") != knn_setting
                or cfg.get("drug_knn_k", 0) != (drug_knn_k or 0)
                or cfg.get("prot_knn_k", 0) != (prot_knn_k or 0)
                or cfg.get("knn_metric", "cosine") != knn_metric
                or bool(cfg.get("knn_symmetric", True)) != bool(knn_symmetric)
                or float(cfg.get("knn_weight_temp", 0.1)) != float(knn_weight_temp)
                or cfg.get("cold_deg_th_drug", 0) != int(cold_deg_th_drug or 0)
                or cfg.get("cold_deg_th_prot", 0) != int(cold_deg_th_prot or 0)
                or cfg.get("cold_mode", "th") != str(cold_mode)
                or float(cfg.get("cold_q", 0.0)) != float(cold_q or 0.0)
                or float(cfg.get("cold_q_drug", 0.0)) != float((cold_q_drug if cold_q_drug is not None else cold_q) or 0.0)
                or float(cfg.get("cold_q_prot", 0.0)) != float((cold_q_prot if cold_q_prot is not None else cold_q) or 0.0)
                or cfg.get("attn_path", "") != attn_sig["attn_path"]
                or int(cfg.get("attn_mtime", 0)) != attn_sig["attn_mtime"]
                or int(cfg.get("attn_size", 0)) != attn_sig["attn_size"]
                or int(cfg.get("atom_feat_version", 0)) != ATOM_FEAT_VERSION
                or int(cfg.get("drug_desc_version", 0)) != DRUG_DESC_VERSION
                or bool(cfg.get("use_drug_desc", False)) != bool(use_drug_desc)
                or bool(cfg.get("use_gasteiger", False)) != bool(use_gasteiger)
                or cfg.get("drug_gnn_path", "") != drug_gnn_sig["drug_gnn_path"]
                or int(cfg.get("drug_gnn_mtime", 0)) != drug_gnn_sig["drug_gnn_mtime"]
                or int(cfg.get("drug_gnn_size", 0)) != drug_gnn_sig["drug_gnn_size"]
                or cfg.get("atom_gnn_path", "") != atom_gnn_sig["atom_gnn_path"]
                or int(cfg.get("atom_gnn_mtime", 0)) != atom_gnn_sig["atom_gnn_mtime"]
                or int(cfg.get("atom_gnn_size", 0)) != atom_gnn_sig["atom_gnn_size"]
                or cfg.get("prot_esm2_path", "") != esm_sig["prot_esm2_path"]
                or int(cfg.get("prot_esm2_mtime", 0)) != esm_sig["prot_esm2_mtime"]
                or int(cfg.get("prot_esm2_size", 0)) != esm_sig["prot_esm2_size"]
                or cfg.get("split_train_path", "") != split_train_sig["split_train_path"]
                or int(cfg.get("split_train_mtime", 0)) != split_train_sig["split_train_mtime"]
                or int(cfg.get("split_train_size", 0)) != split_train_sig["split_train_size"]
                or cfg.get("split_train_hash", "") != split_train_sig["split_train_hash"]
                or cfg.get("split_val_path", "") != split_val_sig["split_val_path"]
                or int(cfg.get("split_val_mtime", 0)) != split_val_sig["split_val_mtime"]
                or int(cfg.get("split_val_size", 0)) != split_val_sig["split_val_size"]
                or cfg.get("split_val_hash", "") != split_val_sig["split_val_hash"]
                or cfg.get("split_test_path", "") != split_test_sig["split_test_path"]
                or int(cfg.get("split_test_mtime", 0)) != split_test_sig["split_test_mtime"]
                or int(cfg.get("split_test_size", 0)) != split_test_sig["split_test_size"]
                or cfg.get("split_test_hash", "") != split_test_sig["split_test_hash"]
                or bool(cfg.get("add_self_loop", True)) != bool(add_self_loop)
            )
            if cache_mismatch:
                if reuse_cache:
                    print("[WARN] Cache meta mismatch detected; using --reuse_cache to proceed.")
                else:
                    print("[INFO] Cache meta mismatch detected; rebuilding preprocessing cache.")
                    for path in list(cache_files.values()) + [config_path]:
                        try:
                            if os.path.exists(path):
                                os.remove(path)
                        except Exception:
                            pass
                    cache_ready = False
            if cache_ready and use_knn_graph:
                if not all(os.path.exists(cache_files[k]) for k in knn_keys):
                    cache_ready = False
    if cache_ready:
        train_edges = np.load(os.path.join(dataset_dir, "processed", "train_edges.npy"))
        val_edges = np.load(os.path.join(dataset_dir, "processed", "val_edges.npy"))
        test_edges = np.load(os.path.join(dataset_dir, "processed", "test_edges.npy"))
        G_atom = sparse.load_npz(cache_files["G_atom"]).astype(np.float32)
        G_residue = sparse.load_npz(cache_files["G_residue"]).astype(np.float32)
        H_atom = sparse.load_npz(cache_files["H_atom"]).astype(np.float32)
        H_residue = sparse.load_npz(cache_files["H_residue"]).astype(np.float32)
        features_atom = np.load(cache_files["features_atom"]).astype(np.float32)
        features_residue = np.load(cache_files["features_residue"]).astype(np.float32)
        atom_to_drug = np.load(cache_files["atom_to_drug"]).astype(np.int64)
        residue_to_protein = np.load(cache_files["residue_to_protein"]).astype(np.int64)
        atom_attn = np.load(cache_files["atom_attn"]).astype(np.float32)
        residue_attn = np.load(cache_files["residue_attn"]).astype(np.float32)
        atom_orig_pos = np.load(cache_files["atom_orig_pos"]).astype(np.int64)
        residue_orig_pos = np.load(cache_files["residue_orig_pos"]).astype(np.int64)
        drug_atom_ptr, drug_atom_nodes = load_ragged_int(cache_files["drug_atom_nodes"])
        prot_res_ptr, prot_res_nodes = load_ragged_int(cache_files["prot_res_nodes"])
        expected_hyper_edges = int(train_edges.shape[0])
        if H_atom.shape[1] != expected_hyper_edges or H_residue.shape[1] != expected_hyper_edges:
            print(
                "[WARN] Cache mismatch: hyperedge column count mismatch "
                f"(expected={expected_hyper_edges}, H_atom={H_atom.shape[1]}, H_residue={H_residue.shape[1]}); rebuilding."
            )
            for path in list(cache_files.values()) + [config_path]:
                try:
                    if os.path.exists(path):
                        os.remove(path)
                except Exception:
                    pass
            cache_ready = False
        if cache_ready:
            print(
                f"[INFO] Hypergraph edges={expected_hyper_edges}, "
                f"supervised_train_edges={expected_hyper_edges}"
            )
        if cache_ready and (
            atom_orig_pos.shape[0] != features_atom.shape[0]
            or atom_orig_pos.shape[0] != atom_to_drug.shape[0]
            or residue_orig_pos.shape[0] != features_residue.shape[0]
            or residue_orig_pos.shape[0] != residue_to_protein.shape[0]
        ):
            print("[WARN] Cache mismatch: orig_pos size does not match features mapping; rebuilding.")
            for path in list(cache_files.values()) + [config_path]:
                try:
                    if os.path.exists(path):
                        os.remove(path)
                except Exception:
                    pass
            cache_ready = False
        if cache_ready:
            num_drugs = int(atom_to_drug.max()) + 1 if atom_to_drug.size else 0
            num_prots = int(residue_to_protein.max()) + 1 if residue_to_protein.size else 0
            if (
                drug_atom_ptr.shape[0] != num_drugs + 1
                or prot_res_ptr.shape[0] != num_prots + 1
                or (drug_atom_nodes.size and int(drug_atom_nodes.max()) >= features_atom.shape[0])
                or (prot_res_nodes.size and int(prot_res_nodes.max()) >= features_residue.shape[0])
            ):
                print("[WARN] Cache mismatch: drug/protein kept-node mapping invalid; rebuilding.")
                for path in list(cache_files.values()) + [config_path]:
                    try:
                        if os.path.exists(path):
                            os.remove(path)
                    except Exception:
                        pass
                cache_ready = False
            else:
                prot_esm_missing = np.load(cache_files["prot_esm_missing"]).astype(np.uint8)
                prot_esm_unreliable = np.load(cache_files["prot_esm_unreliable"]).astype(np.uint8)
                drug_knn_edge_index = None
                drug_knn_edge_weight = None
                prot_knn_edge_index = None
                prot_knn_edge_weight = None
                if os.path.exists(cache_files["drug_knn_edge_index"]):
                    drug_knn_edge_index = np.load(cache_files["drug_knn_edge_index"]).astype(np.int64)
                    drug_knn_edge_weight = np.load(cache_files["drug_knn_edge_weight"]).astype(np.float32)
                if os.path.exists(cache_files["prot_knn_edge_index"]):
                    prot_knn_edge_index = np.load(cache_files["prot_knn_edge_index"]).astype(np.int64)
                    prot_knn_edge_weight = np.load(cache_files["prot_knn_edge_weight"]).astype(np.float32)
                if not (os.path.exists(deg_train_drug_path) and os.path.exists(deg_train_prot_path)):
                    deg_train_drug = np.bincount(train_edges[:, 0], minlength=num_drugs).astype(np.float32) if train_edges.size else np.zeros(num_drugs, dtype=np.float32)
                    deg_train_prot = np.bincount(train_edges[:, 1], minlength=num_prots).astype(np.float32) if train_edges.size else np.zeros(num_prots, dtype=np.float32)
                    np.save(deg_train_drug_path, deg_train_drug)
                    np.save(deg_train_prot_path, deg_train_prot)
                    meta = {
                        "degree_definition": "train_edges_incidence",
                        "split_tag": "train_only",
                        "train_edges_count": int(train_edges.shape[0]),
                        "num_drugs": int(num_drugs),
                        "num_proteins": int(num_prots),
                        "deg_source": "GLOBAL_TRAIN_ONLY",
                    }
                    try:
                        with open(deg_train_meta_path, "w", encoding="utf-8") as f:
                            json.dump(meta, f, indent=2, sort_keys=True)
                    except Exception:
                        pass
                    drug_pcts = np.percentile(deg_train_drug, [0, 10, 50, 90, 100]).tolist() if deg_train_drug.size else [0, 0, 0, 0, 0]
                    prot_pcts = np.percentile(deg_train_prot, [0, 10, 50, 90, 100]).tolist() if deg_train_prot.size else [0, 0, 0, 0, 0]
                    print(f"[INFO] deg_train percentiles drug={np.round(drug_pcts, 3).tolist()} prot={np.round(prot_pcts, 3).tolist()}")
                    print(f"[INFO] deg_train zero ratio drug={float((deg_train_drug==0).mean()) if deg_train_drug.size else 0.0:.4f} "
                          f"prot={float((deg_train_prot==0).mean()) if deg_train_prot.size else 0.0:.4f}")
                return (train_edges, val_edges, test_edges, num_drugs, num_prots,
                        H_atom, H_residue, G_atom, G_residue,
                        features_atom, features_residue,
                        atom_to_drug, residue_to_protein,
                        atom_attn, residue_attn,
                        atom_orig_pos, residue_orig_pos,
                        drug_atom_ptr, drug_atom_nodes,
                        prot_res_ptr, prot_res_nodes,
                        prot_esm_missing, prot_esm_unreliable,
                        drug_knn_edge_index, drug_knn_edge_weight,
                        prot_knn_edge_index, prot_knn_edge_weight)

    with open(os.path.join(dataset_dir, "drug_to_idx.pkl"), "rb") as f:
        drug_to_idx = pickle.load(f)
    with open(os.path.join(dataset_dir, "protein_to_idx.pkl"), "rb") as f:
        prot_to_idx = pickle.load(f)

    max_drug_idx = max(drug_to_idx.values()) if drug_to_idx else -1
    max_prot_idx = max(prot_to_idx.values()) if prot_to_idx else -1
    drug_ids = [None] * (max_drug_idx + 1)
    for k, v in drug_to_idx.items():
        if v >= 0 and v < len(drug_ids):
            drug_ids[v] = k
    protein_ids = [None] * (max_prot_idx + 1)
    for k, v in prot_to_idx.items():
        if v >= 0 and v < len(protein_ids):
            protein_ids[v] = k
    num_drugs = len(drug_ids)
    num_prots = len(protein_ids)

    train_edges = np.load(os.path.join(dataset_dir, "processed", "train_edges.npy"))
    val_edges = np.load(os.path.join(dataset_dir, "processed", "val_edges.npy"))
    test_edges = np.load(os.path.join(dataset_dir, "processed", "test_edges.npy"))
    if train_edges.size == 0:
        raise ValueError("No training edges available for hypergraph construction.")
    print(
        f"[INFO] Hypergraph edges={int(train_edges.shape[0])}, "
        f"supervised_train_edges={int(train_edges.shape[0])}"
    )

    atom_features, atom_to_drug, drug_to_atom_ids = build_atom_nodes(
        drug_ids, use_gasteiger=use_gasteiger
    )
    drug_desc_dim = 0
    drug_desc_missing_ratio = 0.0
    drug_desc_missing_count = 0
    feat_stat = FEATURE_MISSING_STATS.get("drug")
    drug_desc_mean_hash = ""
    if use_drug_desc:
        raw_desc, desc_missing = get_drug_descriptors(drug_ids, return_missing=True)
        if raw_desc.ndim == 2 and raw_desc.shape[0] == len(drug_ids):
            train_drug_idx = np.unique(train_edges[:, 0]).astype(np.int64) if train_edges.size else np.empty((0,), dtype=np.int64)
            train_mask = np.zeros(len(drug_ids), dtype=bool)
            if train_drug_idx.size:
                train_mask[train_drug_idx] = True
            valid_mask = train_mask & (~desc_missing)
            if valid_mask.any():
                desc_mean = raw_desc[valid_mask].mean(axis=0, keepdims=True)
                desc_std = raw_desc[valid_mask].std(axis=0, keepdims=True)
            else:
                desc_mean = np.zeros((1, raw_desc.shape[1]), dtype=np.float32)
                desc_std = np.ones((1, raw_desc.shape[1]), dtype=np.float32)
            desc_filled = raw_desc.copy()
            desc_filled[desc_missing] = desc_mean
            desc_norm = (desc_filled - desc_mean) / np.maximum(desc_std, 1e-6)
            missing_flag = desc_missing.astype(np.float32).reshape(-1, 1)
            drug_desc = np.hstack([desc_norm, missing_flag])
            atom_features = np.hstack([atom_features, drug_desc[atom_to_drug]])
            drug_desc_dim = drug_desc.shape[1]
            drug_desc_missing_count = int(desc_missing.sum())
            drug_desc_missing_ratio = float(desc_missing.mean()) if desc_missing.size else 0.0
            try:
                drug_desc_mean_hash = hashlib.md5(desc_mean.tobytes()).hexdigest()[:8]
            except Exception:
                drug_desc_mean_hash = ""
            print(
                f"[INFO] Drug desc missing={drug_desc_missing_count} "
                f"({drug_desc_missing_ratio:.4f}), impute={DRUG_DESC_IMPUTE}, train_only=1"
            )
            if feat_stat is not None:
                print(
                    f"[INFO] drug_features_missing count={feat_stat.get('missing_count', 0)} "
                    f"ratio={feat_stat.get('missing_ratio', 0.0):.4f} "
                    f"impute={'train_mean' if feat_stat.get('impute', False) else 'none'} "
                    f"train_only={1 if feat_stat.get('train_only', False) else 0} "
                    f"missing_flag={'on' if feat_stat.get('add_missing_flag', False) else 'off'}"
                )
        else:
            print("[WARN] Drug descriptor shape mismatch; skip.")
    drug_gnn_dim = 0
    drug_gnn_emb = _load_embedding(drug_gnn_path, len(drug_ids), "drug_gnn")
    if drug_gnn_emb is not None:
        atom_features = np.hstack([atom_features, drug_gnn_emb[atom_to_drug]])
        drug_gnn_dim = drug_gnn_emb.shape[1]
    atom_gnn_dim = 0
    atom_gnn_emb = _load_embedding(atom_gnn_path, atom_features.shape[0], "atom_gnn")
    if atom_gnn_emb is not None:
        atom_features = np.hstack([atom_features, atom_gnn_emb])
        atom_gnn_dim = atom_gnn_emb.shape[1]
    if use_drug_desc or drug_gnn_emb is not None or atom_gnn_emb is not None:
        print(
            f"[INFO] Atom feature dim: {atom_features.shape[1]} "
            f"(desc={drug_desc_dim}, drug_gnn={drug_gnn_dim}, atom_gnn={atom_gnn_dim})"
        )
    residue_features, residue_to_protein, prot_to_res_ids = build_residue_nodes(protein_ids)
    residue_physchem = build_residue_physchem_features(protein_ids)

    attn_db = load_psichic_attention(psichic_attention_path)

    drug_atom_ids_full = [
        np.asarray(drug_to_atom_ids.get(drug_ids[i], []), dtype=np.int64)
        for i in range(len(drug_ids))
    ]
    prot_res_ids_full = [
        np.asarray(prot_to_res_ids.get(protein_ids[i], []), dtype=np.int64)
        for i in range(len(protein_ids))
    ]
    drug_atom_pos_full = [
        np.arange(drug_atom_ids_full[i].size, dtype=np.int64)
        for i in range(len(drug_ids))
    ]
    prot_res_pos_full = [
        np.arange(prot_res_ids_full[i].size, dtype=np.int64)
        for i in range(len(protein_ids))
    ]

    # Two-phase build:
    # Phase-1 aggregates PSICHIC attention (pre-prune length) on train_edges to get
    # per-drug/protein average scores and select keep_pos (top-k).
    # Phase-2 rescans train_edges after keep_pos is fixed to build H with aligned
    # incidence weights sliced by orig_pos (H depends on the pruned node set).
    drug_atom_sum = {}
    drug_atom_cnt = {}
    prot_res_sum = {}
    prot_res_sq_sum = {}
    prot_res_cnt = {}
    atom_attn_hits = 0
    atom_attn_total = 0
    res_attn_hits = 0
    res_attn_total = 0
    atom_score_mismatch = 0
    res_score_mismatch = 0

    for drug_idx, prot_idx, _ in train_edges:
        drug_idx = int(drug_idx)
        prot_idx = int(prot_idx)
        if drug_idx >= len(drug_ids) or prot_idx >= len(protein_ids):
            continue
        drug = drug_ids[drug_idx]
        prot = protein_ids[prot_idx]
        if drug is None or prot is None:
            continue
        drug_key = str(drug).strip()
        prot_key = str(prot).strip()
        atom_ids_full = drug_atom_ids_full[drug_idx]
        res_ids_full = prot_res_ids_full[prot_idx]
        n_atoms_full = int(atom_ids_full.size)
        n_res_full = int(res_ids_full.size)
        if n_atoms_full == 0 and n_res_full == 0:
            continue
        atom_scores_raw, residue_scores_raw = get_pair_attention(attn_db, prot_key, drug_key)
        if n_atoms_full > 0:
            atom_attn_total += 1
            if atom_scores_raw is not None:
                atom_attn_hits += 1
            atom_scores_full = align_scores_to_length(atom_scores_raw, n_atoms_full, fill=1.0)
            prev = drug_atom_sum.get(drug_idx)
            if prev is None or prev.shape != atom_scores_full.shape:
                if prev is not None and prev.shape != atom_scores_full.shape:
                    atom_score_mismatch += 1
                drug_atom_sum[drug_idx] = atom_scores_full.copy()
                drug_atom_cnt[drug_idx] = 1
            else:
                drug_atom_sum[drug_idx] += atom_scores_full
                drug_atom_cnt[drug_idx] += 1
        if n_res_full > 0:
            res_attn_total += 1
            if residue_scores_raw is not None:
                res_attn_hits += 1
            res_scores_full = align_scores_to_length(residue_scores_raw, n_res_full, fill=1.0)
            prev = prot_res_sum.get(prot_idx)
            if prev is None or prev.shape != res_scores_full.shape:
                if prev is not None and prev.shape != res_scores_full.shape:
                    res_score_mismatch += 1
                prot_res_sum[prot_idx] = res_scores_full.copy()
                prot_res_sq_sum[prot_idx] = res_scores_full * res_scores_full
                prot_res_cnt[prot_idx] = 1
            else:
                prot_res_sum[prot_idx] += res_scores_full
                prot_res_sq_sum[prot_idx] += res_scores_full * res_scores_full
                prot_res_cnt[prot_idx] += 1

    if atom_attn_total > 0 or res_attn_total > 0:
        atom_ratio = atom_attn_hits / max(atom_attn_total, 1)
        res_ratio = res_attn_hits / max(res_attn_total, 1)
        print(
            f"[INFO] PSICHIC coverage (train pairs): "
            f"atom={atom_attn_hits}/{atom_attn_total} ({atom_ratio:.3f}), "
            f"residue={res_attn_hits}/{res_attn_total} ({res_ratio:.3f})"
        )
    if atom_score_mismatch or res_score_mismatch:
        print(
            f"[WARN] PSICHIC length mismatch: atom={atom_score_mismatch}, residue={res_score_mismatch}. "
            "Falling back to per-pair lengths."
        )

    drug_keep_pos = []
    for drug_idx in range(len(drug_ids)):
        n_atoms_full = int(drug_atom_ids_full[drug_idx].size)
        if n_atoms_full <= 0:
            keep_pos = np.empty((0,), dtype=np.int64)
        elif atom_keep is None:
            keep_pos = np.arange(n_atoms_full, dtype=np.int64)
        else:
            cnt = drug_atom_cnt.get(drug_idx, 0)
            if cnt > 0:
                avg = drug_atom_sum[drug_idx] / float(cnt)
                keep_pos = topk_indices(avg, atom_keep)
            else:
                keep_pos = np.asarray(
                    _fallback_select(np.arange(n_atoms_full), atom_keep, prune_fallback),
                    dtype=np.int64
                )
                keep_pos = np.sort(keep_pos)
        drug_keep_pos.append(keep_pos)

    prot_keep_pos = []
    for prot_idx in range(len(protein_ids)):
        n_res_full = int(prot_res_ids_full[prot_idx].size)
        if n_res_full <= 0:
            keep_pos = np.empty((0,), dtype=np.int64)
        elif residue_keep is None:
            keep_pos = np.arange(n_res_full, dtype=np.int64)
        else:
            cnt = prot_res_cnt.get(prot_idx, 0)
            if cnt > 0:
                avg = prot_res_sum[prot_idx] / float(cnt)
                var = None
                sq_sum = prot_res_sq_sum.get(prot_idx)
                if sq_sum is not None and sq_sum.shape == avg.shape:
                    var = sq_sum / float(cnt) - avg * avg
                # Prefer residues with high variability across ligands; fallback to mean if unstable.
                if var is None or (not np.all(np.isfinite(var))) or np.allclose(var, var[0]):
                    keep_pos = topk_indices(avg, residue_keep)
                else:
                    keep_pos = topk_indices(var, residue_keep)
            else:
                keep_pos = np.asarray(
                    _fallback_select(np.arange(n_res_full), residue_keep, prune_fallback),
                    dtype=np.int64
                )
                keep_pos = np.sort(keep_pos)
        prot_keep_pos.append(keep_pos)

    atom_orig_pos_full = np.full(atom_features.shape[0], -1, dtype=np.int64)
    for drug_idx, atom_ids in enumerate(drug_atom_ids_full):
        if atom_ids.size == 0:
            continue
        atom_orig_pos_full[atom_ids] = drug_atom_pos_full[drug_idx]
    res_orig_pos_full = np.full(residue_features.shape[0], -1, dtype=np.int64)
    for prot_idx, res_ids in enumerate(prot_res_ids_full):
        if res_ids.size == 0:
            continue
        res_orig_pos_full[res_ids] = prot_res_pos_full[prot_idx]

    atom_keep_mask = np.zeros(atom_features.shape[0], dtype=bool)
    for drug_idx, keep_pos in enumerate(drug_keep_pos):
        if keep_pos.size == 0:
            continue
        atom_ids = drug_atom_ids_full[drug_idx]
        if atom_ids.size == 0:
            continue
        atom_keep_mask[atom_ids[keep_pos]] = True

    res_keep_mask = np.zeros(residue_features.shape[0], dtype=bool)
    for prot_idx, keep_pos in enumerate(prot_keep_pos):
        if keep_pos.size == 0:
            continue
        res_ids = prot_res_ids_full[prot_idx]
        if res_ids.size == 0:
            continue
        res_keep_mask[res_ids[keep_pos]] = True

    if atom_keep is not None or residue_keep is not None:
        print(
            f"[INFO] Prune atoms {atom_features.shape[0]} -> {int(atom_keep_mask.sum())}, "
            f"residues {residue_features.shape[0]} -> {int(res_keep_mask.sum())}"
        )

    old_atom_to_new = -np.ones(atom_features.shape[0], dtype=np.int64)
    old_atom_to_new[atom_keep_mask] = np.arange(int(atom_keep_mask.sum()), dtype=np.int64)
    old_res_to_new = -np.ones(residue_features.shape[0], dtype=np.int64)
    old_res_to_new[res_keep_mask] = np.arange(int(res_keep_mask.sum()), dtype=np.int64)

    atom_features = atom_features[atom_keep_mask]
    atom_to_drug = atom_to_drug[atom_keep_mask]
    residue_features = residue_features[res_keep_mask]
    residue_physchem = residue_physchem[res_keep_mask]
    residue_to_protein = residue_to_protein[res_keep_mask]
    atom_orig_pos = atom_orig_pos_full[atom_keep_mask]
    residue_orig_pos = res_orig_pos_full[res_keep_mask]

    esm_features = None
    prot_esm_missing = None
    prot_esm_unreliable = None
    esm_meta = None
    residue_onehot_dim = int(residue_features.shape[1]) if residue_features is not None else 0
    if protein_feat_mode != "onehot":
        if esm_dict is None:
            print(f"[WARN] protein_feat_mode={protein_feat_mode} but no ESM2 dict found; using onehot only.")
        else:
            if os.environ.get("HGACN_DEBUG_ESM", "0").strip() == "1":
                debug_esm_alignment()
            esm_features, prot_esm_missing, prot_esm_unreliable, esm_meta = build_residue_features_from_esm(
                esm_dict,
                protein_ids,
                residue_to_protein,
                residue_orig_pos,
                feat_dim=1280,
                fallback=esm_fallback,
                esm_special_tokens=esm_special_tokens,
                esm_norm=esm_norm,
                esm_strict=esm_strict,
                debug_samples=5,
                return_meta=True,
            )
    if protein_feat_mode == "esm2" and esm_features is not None:
        residue_features = esm_features
    elif protein_feat_mode == "concat" and esm_features is not None:
        residue_features = np.hstack([residue_features, esm_features])
        if use_physchem_feat:
            residue_features = np.hstack([residue_features, residue_physchem])
    elif protein_feat_mode == "concat" and esm_features is None and use_physchem_feat:
        residue_features = np.hstack([residue_features, residue_physchem])
    if prot_esm_missing is None:
        prot_esm_missing = np.ones(len(protein_ids), dtype=np.uint8)
    if prot_esm_unreliable is None:
        prot_esm_unreliable = prot_esm_missing.copy()
    esm_mismatch_path = None
    esm_mismatch_count = 0
    esm_mismatch_global_fallback = 0
    esm_missing_count = 0
    esm_unreliable_count = 0
    if esm_meta is not None:
        esm_mismatch_count = int(esm_meta.get("mismatch_count", 0))
        esm_mismatch_global_fallback = int(esm_meta.get("mismatch_global_fallback_count", 0))
        esm_missing_count = int(esm_meta.get("missing_count", 0))
        esm_unreliable_count = int(esm_meta.get("unreliable_count", 0))
        mismatch_records = esm_meta.get("mismatch_records", [])
        if mismatch_records:
            esm_mismatch_path = os.path.join(processed_dir, "esm_mismatch_proteins.json")
            try:
                with open(esm_mismatch_path, "w", encoding="utf-8") as f:
                    json.dump(mismatch_records, f, indent=2, ensure_ascii=False)
                print(f"[INFO] ESM2 mismatch list saved: {esm_mismatch_path}")
            except Exception as e:
                print(f"[WARN] Failed to write ESM mismatch list: {e}")
        if esm_missing_count or esm_mismatch_count:
            print(
                f"[INFO] ESM2 unreliable proteins: missing={esm_missing_count}, "
                f"mismatch={esm_mismatch_count}, unreliable={esm_unreliable_count}"
            )
    if protein_feat_mode == "concat":
        esm_dim = int(esm_features.shape[1]) if esm_features is not None else 0
        physchem_dim = int(residue_physchem.shape[1]) if (residue_physchem is not None and use_physchem_feat) else 0
        total_dim = int(residue_features.shape[1]) if residue_features is not None else 0
        print(
            f"[INFO] Protein feature dims: onehot={residue_onehot_dim}, esm={esm_dim}, "
            f"physchem={physchem_dim}, total={total_dim}"
        )

    drug_keep_atom_ids_new = []
    drug_keep_atom_orig_pos = []
    for drug_idx, keep_pos in enumerate(drug_keep_pos):
        atom_ids = drug_atom_ids_full[drug_idx]
        if atom_ids.size == 0 or keep_pos.size == 0:
            drug_keep_atom_ids_new.append(np.empty((0,), dtype=np.int64))
            drug_keep_atom_orig_pos.append(np.empty((0,), dtype=np.int64))
            continue
        kept_global = atom_ids[keep_pos]
        kept_new = old_atom_to_new[kept_global]
        valid = kept_new >= 0
        drug_keep_atom_ids_new.append(kept_new[valid])
        drug_keep_atom_orig_pos.append(keep_pos[valid])

    prot_keep_res_ids_new = []
    prot_keep_res_orig_pos = []
    for prot_idx, keep_pos in enumerate(prot_keep_pos):
        res_ids = prot_res_ids_full[prot_idx]
        if res_ids.size == 0 or keep_pos.size == 0:
            prot_keep_res_ids_new.append(np.empty((0,), dtype=np.int64))
            prot_keep_res_orig_pos.append(np.empty((0,), dtype=np.int64))
            continue
        kept_global = res_ids[keep_pos]
        kept_new = old_res_to_new[kept_global]
        valid = kept_new >= 0
        prot_keep_res_ids_new.append(kept_new[valid])
        prot_keep_res_orig_pos.append(keep_pos[valid])

    num_train_edges = len(train_edges)
    H_atom = sparse.lil_matrix((atom_features.shape[0], num_train_edges), dtype=np.float32)
    H_residue = sparse.lil_matrix((residue_features.shape[0], num_train_edges), dtype=np.float32)
    atom_attn_sum = np.zeros(atom_features.shape[0], dtype=np.float32)
    atom_attn_cnt = np.zeros(atom_features.shape[0], dtype=np.int64)
    residue_attn_sum = np.zeros(residue_features.shape[0], dtype=np.float32)
    residue_attn_cnt = np.zeros(residue_features.shape[0], dtype=np.int64)
    atom_inc_counts = np.zeros(num_train_edges, dtype=np.int32)
    res_inc_counts = np.zeros(num_train_edges, dtype=np.int32)
    prior_ent_atom = np.zeros(num_train_edges, dtype=np.float32)
    prior_ent_res = np.zeros(num_train_edges, dtype=np.float32)

    for i, (drug_idx, prot_idx, _) in enumerate(train_edges):
        drug_idx = int(drug_idx)
        prot_idx = int(prot_idx)
        if drug_idx >= len(drug_ids) or prot_idx >= len(protein_ids):
            continue
        drug = drug_ids[drug_idx]
        prot = protein_ids[prot_idx]
        if drug is None or prot is None:
            continue
        drug_key = str(drug).strip()
        prot_key = str(prot).strip()
        atom_ids_full = drug_atom_ids_full[drug_idx]
        res_ids_full = prot_res_ids_full[prot_idx]
        n_atoms_full = int(atom_ids_full.size)
        n_res_full = int(res_ids_full.size)

        atom_ids = drug_keep_atom_ids_new[drug_idx]
        atom_orig_pos_local = drug_keep_atom_orig_pos[drug_idx]
        res_ids = prot_keep_res_ids_new[prot_idx]
        res_orig_pos_local = prot_keep_res_orig_pos[prot_idx]

        atom_scores_raw, residue_scores_raw = get_pair_attention(attn_db, prot_key, drug_key)
        if atom_ids.size > 0 and n_atoms_full > 0:
            atom_scores_full = align_scores_to_length(atom_scores_raw, n_atoms_full, fill=1.0)
            valid = (atom_orig_pos_local >= 0) & (atom_orig_pos_local < atom_scores_full.size)
            if not valid.all():
                atom_ids = atom_ids[valid]
                atom_orig_pos_local = atom_orig_pos_local[valid]
            atom_scores = (
                atom_scores_full[atom_orig_pos_local]
                if atom_ids.size > 0
                else np.asarray([], dtype=np.float32)
            )
        else:
            atom_scores = np.asarray([], dtype=np.float32)

        if res_ids.size > 0 and n_res_full > 0:
            res_scores_full = align_scores_to_length(residue_scores_raw, n_res_full, fill=1.0)
            valid = (res_orig_pos_local >= 0) & (res_orig_pos_local < res_scores_full.size)
            if not valid.all():
                res_ids = res_ids[valid]
                res_orig_pos_local = res_orig_pos_local[valid]
            residue_scores = (
                res_scores_full[res_orig_pos_local]
                if res_ids.size > 0
                else np.asarray([], dtype=np.float32)
            )
        else:
            residue_scores = np.asarray([], dtype=np.float32)

        if atom_randk > 0:
            atom_ids, atom_scores = apply_topk_randk_indices(
                atom_ids,
                atom_scores,
                atom_top_k,
                atom_randk,
                randk_seed + i * 2 + 0,
                rand_weight_mode=randk_weight_mode,
                prior_floor=prior_floor,
                fallback_mode=prune_fallback,
            )
        else:
            atom_ids, atom_scores = apply_top_k_indices(
                atom_ids, atom_scores, atom_top_k, fallback_mode=prune_fallback
            )
        if residue_randk > 0:
            res_ids, residue_scores = apply_topk_randk_indices(
                res_ids,
                residue_scores,
                residue_top_k,
                residue_randk,
                randk_seed + i * 2 + 1,
                rand_weight_mode=randk_weight_mode,
                prior_floor=prior_floor,
                fallback_mode=prune_fallback,
            )
        else:
            res_ids, residue_scores = apply_top_k_indices(
                res_ids, residue_scores, residue_top_k, fallback_mode=prune_fallback
            )
        atom_ids = np.asarray(atom_ids, dtype=np.int64)
        res_ids = np.asarray(res_ids, dtype=np.int64)
        atom_scores = np.asarray(atom_scores, dtype=np.float32)
        residue_scores = np.asarray(residue_scores, dtype=np.float32)
        if atom_ids.size != atom_scores.size and atom_ids.size > 0:
            atom_scores = np.ones(atom_ids.size, dtype=np.float32)
        if res_ids.size != residue_scores.size and res_ids.size > 0:
            residue_scores = np.ones(res_ids.size, dtype=np.float32)
        atom_scores = normalize_incidence(atom_scores, method="sum")
        residue_scores = normalize_incidence(residue_scores, method="sum")
        atom_inc_counts[i] = int(atom_ids.size)
        res_inc_counts[i] = int(res_ids.size)
        if atom_scores.size > 1:
            prior_ent_atom[i] = float(
                -np.sum(atom_scores * np.log(np.clip(atom_scores, 1e-12, None))) / np.log(atom_scores.size)
            )
        if residue_scores.size > 1:
            prior_ent_res[i] = float(
                -np.sum(residue_scores * np.log(np.clip(residue_scores, 1e-12, None))) / np.log(residue_scores.size)
            )

        if atom_ids.size > 0:
            H_atom[atom_ids, i] = atom_scores
            atom_attn_sum[atom_ids] += atom_scores
            atom_attn_cnt[atom_ids] += 1
        if res_ids.size > 0:
            H_residue[res_ids, i] = residue_scores
            residue_attn_sum[res_ids] += residue_scores
            residue_attn_cnt[res_ids] += 1

    atom_attn = atom_attn_sum / np.maximum(atom_attn_cnt, 1)
    residue_attn = residue_attn_sum / np.maximum(residue_attn_cnt, 1)

    if atom_inc_counts.size:
        print(
            "[INFO] inc_atom_per_edge stats: "
            f"mean={atom_inc_counts.mean():.2f}, "
            f"p50={np.percentile(atom_inc_counts, 50):.0f}, "
            f"p90={np.percentile(atom_inc_counts, 90):.0f}, "
            f"max={atom_inc_counts.max():.0f}"
        )
    if res_inc_counts.size:
        print(
            "[INFO] inc_res_per_edge stats: "
            f"mean={res_inc_counts.mean():.2f}, "
            f"p50={np.percentile(res_inc_counts, 50):.0f}, "
            f"p90={np.percentile(res_inc_counts, 90):.0f}, "
            f"max={res_inc_counts.max():.0f}"
        )
    if prior_ent_atom.size:
        print(
            "[INFO] prior_entropy(atom) stats: "
            f"mean={prior_ent_atom.mean():.4f}, "
            f"p50={np.percentile(prior_ent_atom, 50):.4f}, "
            f"p90={np.percentile(prior_ent_atom, 90):.4f}"
        )
    if prior_ent_res.size:
        print(
            "[INFO] prior_entropy(res) stats: "
            f"mean={prior_ent_res.mean():.4f}, "
            f"p50={np.percentile(prior_ent_res, 50):.4f}, "
            f"p90={np.percentile(prior_ent_res, 90):.4f}"
        )
    if prot_esm_missing is not None and train_edges.size:
        unreliable_mask = prot_esm_missing[train_edges[:, 1].astype(np.int64)] > 0
        unreliable_ratio = float(unreliable_mask.mean()) if unreliable_mask.size else 0.0
        if unreliable_mask.any() and prior_ent_res.size:
            unreliable_ent = float(prior_ent_res[unreliable_mask].mean())
        else:
            unreliable_ent = 0.0
        print(
            f"[INFO] ESM unreliable proteins ratio={unreliable_ratio:.4f}, "
            f"unreliable_prior_entropy_res_norm_mean={unreliable_ent:.4f}"
        )

    if atom_attn.size > 0:
        features_atom = np.hstack([atom_features, atom_attn.reshape(-1, 1)])
    else:
        features_atom = atom_features
    if residue_attn.size > 0:
        features_residue = np.hstack([residue_features, residue_attn.reshape(-1, 1)])
    else:
        features_residue = residue_features

    adj_atom = H_atom.dot(H_atom.T).astype(np.float32).tocsr()
    adj_residue = H_residue.dot(H_residue.T).astype(np.float32).tocsr()
    G_atom = normalize_adj_safe(adj_atom, add_self_loop=add_self_loop).astype(np.float32).tocsr()
    G_residue = normalize_adj_safe(adj_residue, add_self_loop=add_self_loop).astype(np.float32).tocsr()
    H_atom = H_atom.tocsr().astype(np.float32)
    H_residue = H_residue.tocsr().astype(np.float32)
    if H_atom.shape[1] != int(train_edges.shape[0]) or H_residue.shape[1] != int(train_edges.shape[0]):
        raise RuntimeError(
            "Atomic hypergraph column mismatch: "
            f"H_atom={H_atom.shape[1]}, H_residue={H_residue.shape[1]}, expected={int(train_edges.shape[0])}"
        )

    sparse.save_npz(cache_files["G_atom"], G_atom)
    sparse.save_npz(cache_files["G_residue"], G_residue)
    sparse.save_npz(cache_files["H_atom"], H_atom)
    sparse.save_npz(cache_files["H_residue"], H_residue)
    np.save(cache_files["features_atom"], features_atom)
    np.save(cache_files["features_residue"], features_residue)
    np.save(cache_files["atom_to_drug"], atom_to_drug)
    np.save(cache_files["residue_to_protein"], residue_to_protein)
    np.save(cache_files["atom_attn"], atom_attn)
    np.save(cache_files["residue_attn"], residue_attn)
    np.save(cache_files["atom_orig_pos"], atom_orig_pos)
    np.save(cache_files["residue_orig_pos"], residue_orig_pos)
    save_ragged_int(cache_files["drug_atom_nodes"], drug_keep_atom_ids_new)
    save_ragged_int(cache_files["prot_res_nodes"], prot_keep_res_ids_new)
    drug_deg = np.bincount(train_edges[:, 0], minlength=num_drugs).astype(np.float32) if len(train_edges) else np.zeros(num_drugs, dtype=np.float32)
    prot_deg = np.bincount(train_edges[:, 1], minlength=num_prots).astype(np.float32) if len(train_edges) else np.zeros(num_prots, dtype=np.float32)
    np.save(cache_files["drug_deg"], drug_deg)
    np.save(cache_files["prot_deg"], prot_deg)
    np.save(deg_train_drug_path, drug_deg)
    np.save(deg_train_prot_path, prot_deg)
    try:
        with open(deg_train_meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "degree_definition": "train_edges_incidence",
                    "split_tag": "train_only",
                    "train_edges_count": int(train_edges.shape[0]),
                    "num_drugs": int(num_drugs),
                    "num_proteins": int(num_prots),
                    "deg_source": "GLOBAL_TRAIN_ONLY",
                },
                f,
                indent=2,
                sort_keys=True,
            )
    except Exception:
        pass
    drug_pcts = np.percentile(drug_deg, [0, 10, 50, 90, 100]).tolist() if drug_deg.size else [0, 0, 0, 0, 0]
    prot_pcts = np.percentile(prot_deg, [0, 10, 50, 90, 100]).tolist() if prot_deg.size else [0, 0, 0, 0, 0]
    print(f"[INFO] deg_train percentiles drug={np.round(drug_pcts, 3).tolist()} prot={np.round(prot_pcts, 3).tolist()}")
    print(f"[INFO] deg_train zero ratio drug={float((drug_deg==0).mean()) if drug_deg.size else 0.0:.4f} "
          f"prot={float((prot_deg==0).mean()) if prot_deg.size else 0.0:.4f}")
    np.save(cache_files["prot_esm_missing"], prot_esm_missing.astype(np.uint8))
    np.save(cache_files["prot_esm_unreliable"], prot_esm_unreliable.astype(np.uint8))

    drug_knn_edge_index = np.empty((2, 0), dtype=np.int64)
    drug_knn_edge_weight = np.empty((0,), dtype=np.float32)
    prot_knn_edge_index = np.empty((2, 0), dtype=np.int64)
    prot_knn_edge_weight = np.empty((0,), dtype=np.float32)
    prot_knn_disabled_reason = None
    if use_knn_graph:
        print(
            f"[INFO] Build kNN graphs: drug_k={drug_knn_k}, prot_k={prot_knn_k}, "
            f"metric={knn_metric}, symmetric={knn_symmetric}, temp={knn_weight_temp}, "
            f"setting={knn_setting}"
        )
        drug_vec = mean_pool_by_index(atom_features, atom_to_drug, num_drugs)
        prot_vec = mean_pool_by_index(residue_features, residue_to_protein, num_prots)
        cold_mode = (cold_mode or "th").strip().lower()
        if cold_mode == "quantile":
            q_drug = cold_q_drug if cold_q_drug is not None else cold_q
            q_prot = cold_q_prot if cold_q_prot is not None else cold_q
            q_drug = max(0.0, min(1.0, float(q_drug)))
            q_prot = max(0.0, min(1.0, float(q_prot)))
            cold_drug_th = float(np.quantile(drug_deg, q_drug)) if drug_deg.size else 0.0
            cold_prot_th = float(np.quantile(prot_deg, q_prot)) if prot_deg.size else 0.0
        else:
            cold_drug_th = float(cold_deg_th_drug)
            cold_prot_th = float(cold_deg_th_prot)
        print(
            f"[INFO] cold_mode={cold_mode}, cold_th_drug={cold_drug_th:.3f}, "
            f"cold_th_prot={cold_prot_th:.3f}"
        )
        cold_drug_idx = np.where(drug_deg <= cold_drug_th)[0]
        cold_prot_idx = np.where(prot_deg <= cold_prot_th)[0]
        print(
            f"[INFO] cold_node_count drug={int(cold_drug_idx.size)}/{int(num_drugs)} "
            f"prot={int(cold_prot_idx.size)}/{int(num_prots)}"
        )
        if cold_prot_idx.size == 0:
            prot_knn_disabled_reason = "all_warm_or_deg>=th"
        if knn_setting == "inductive":
            train_drugs = np.unique(train_edges[:, 0]).astype(np.int64) if train_edges.size else np.empty((0,), dtype=np.int64)
            val_drugs = np.unique(val_edges[:, 0]).astype(np.int64) if val_edges.size else np.empty((0,), dtype=np.int64)
            test_drugs = np.unique(test_edges[:, 0]).astype(np.int64) if test_edges.size else np.empty((0,), dtype=np.int64)
            train_prots = np.unique(train_edges[:, 1]).astype(np.int64) if train_edges.size else np.empty((0,), dtype=np.int64)
            val_prots = np.unique(val_edges[:, 1]).astype(np.int64) if val_edges.size else np.empty((0,), dtype=np.int64)
            test_prots = np.unique(test_edges[:, 1]).astype(np.int64) if test_edges.size else np.empty((0,), dtype=np.int64)

            cold_train_drugs = train_drugs[np.isin(train_drugs, cold_drug_idx)]
            cold_val_drugs = val_drugs[np.isin(val_drugs, cold_drug_idx)]
            cold_test_drugs = test_drugs[np.isin(test_drugs, cold_drug_idx)]
            cold_train_prots = train_prots[np.isin(train_prots, cold_prot_idx)]
            cold_val_prots = val_prots[np.isin(val_prots, cold_prot_idx)]
            cold_test_prots = test_prots[np.isin(test_prots, cold_prot_idx)]

            drug_edges = []
            prot_edges = []
            if cold_train_drugs.size:
                e_idx, e_w = build_knn_graph(
                    drug_vec,
                    k=int(drug_knn_k),
                    metric=knn_metric,
                    symmetric=True,
                    weight_temp=float(knn_weight_temp),
                    chunk_size=512,
                    filter_idx=cold_train_drugs,
                    key_idx=train_drugs,
                )
                drug_edges.append((e_idx, e_w))
            if cold_val_drugs.size:
                e_idx, e_w = build_knn_graph(
                    drug_vec,
                    k=int(drug_knn_k),
                    metric=knn_metric,
                    symmetric=False,
                    weight_temp=float(knn_weight_temp),
                    chunk_size=512,
                    filter_idx=cold_val_drugs,
                    key_idx=train_drugs,
                )
                drug_edges.append((e_idx, e_w))
            if cold_test_drugs.size:
                e_idx, e_w = build_knn_graph(
                    drug_vec,
                    k=int(drug_knn_k),
                    metric=knn_metric,
                    symmetric=False,
                    weight_temp=float(knn_weight_temp),
                    chunk_size=512,
                    filter_idx=cold_test_drugs,
                    key_idx=train_drugs,
                )
                drug_edges.append((e_idx, e_w))
            if drug_edges:
                drug_knn_edge_index = np.concatenate([e[0] for e in drug_edges], axis=1)
                drug_knn_edge_weight = np.concatenate([e[1] for e in drug_edges], axis=0)

            if cold_train_prots.size:
                e_idx, e_w = build_knn_graph(
                    prot_vec,
                    k=int(prot_knn_k),
                    metric=knn_metric,
                    symmetric=True,
                    weight_temp=float(knn_weight_temp),
                    chunk_size=512,
                    filter_idx=cold_train_prots,
                    key_idx=train_prots,
                )
                prot_edges.append((e_idx, e_w))
            if cold_val_prots.size:
                e_idx, e_w = build_knn_graph(
                    prot_vec,
                    k=int(prot_knn_k),
                    metric=knn_metric,
                    symmetric=False,
                    weight_temp=float(knn_weight_temp),
                    chunk_size=512,
                    filter_idx=cold_val_prots,
                    key_idx=train_prots,
                )
                prot_edges.append((e_idx, e_w))
            if cold_test_prots.size:
                e_idx, e_w = build_knn_graph(
                    prot_vec,
                    k=int(prot_knn_k),
                    metric=knn_metric,
                    symmetric=False,
                    weight_temp=float(knn_weight_temp),
                    chunk_size=512,
                    filter_idx=cold_test_prots,
                    key_idx=train_prots,
                )
                prot_edges.append((e_idx, e_w))
            if prot_edges:
                prot_knn_edge_index = np.concatenate([e[0] for e in prot_edges], axis=1)
                prot_knn_edge_weight = np.concatenate([e[1] for e in prot_edges], axis=0)
            if drug_knn_edge_index.size:
                # edge[1] are neighbor ids (should remain train-only in inductive mode).
                bad = ~np.isin(drug_knn_edge_index[1], train_drugs)
                if bad.any():
                    raise ValueError("[LEAK] inductive drug kNN has non-train neighbor ids.")
            if prot_knn_edge_index.size:
                # edge[1] are neighbor ids (should remain train-only in inductive mode).
                bad = ~np.isin(prot_knn_edge_index[1], train_prots)
                if bad.any():
                    raise ValueError("[LEAK] inductive protein kNN has non-train neighbor ids.")
        else:
            drug_knn_edge_index, drug_knn_edge_weight = build_knn_graph(
                drug_vec,
                k=int(drug_knn_k),
                metric=knn_metric,
                symmetric=bool(knn_symmetric),
                weight_temp=float(knn_weight_temp),
                chunk_size=512,
                filter_idx=cold_drug_idx,
            )
            prot_knn_edge_index, prot_knn_edge_weight = build_knn_graph(
                prot_vec,
                k=int(prot_knn_k),
                metric=knn_metric,
                symmetric=bool(knn_symmetric),
                weight_temp=float(knn_weight_temp),
                chunk_size=512,
                filter_idx=cold_prot_idx,
            )
        if prot_knn_edge_index.shape[1] == 0 and prot_knn_disabled_reason is None:
            prot_knn_disabled_reason = "no_edges_built"
        print(
            f"[INFO] kNN edges: drug={drug_knn_edge_index.shape[1]}, prot={prot_knn_edge_index.shape[1]}"
        )
        if prot_knn_edge_index.shape[1] == 0:
            print(f"[INFO] prot_knn_disabled_reason={prot_knn_disabled_reason}")
    np.save(cache_files["drug_knn_edge_index"], drug_knn_edge_index)
    np.save(cache_files["drug_knn_edge_weight"], drug_knn_edge_weight)
    np.save(cache_files["prot_knn_edge_index"], prot_knn_edge_index)
    np.save(cache_files["prot_knn_edge_weight"], prot_knn_edge_weight)
    try:
        with open(config_path, "w") as f:
                json.dump(
                    {
                        "add_self_loop": bool(add_self_loop),
                        "atom_keep": atom_keep or 0,
                        "residue_keep": residue_keep or 0,
                        "atom_top_k": atom_top_k or 0,
                        "residue_top_k": residue_top_k or 0,
                        "atom_randk": atom_randk or 0,
                        "res_randk": residue_randk or 0,
                        "randk_seed": int(randk_seed or 0),
                        "randk_weight_mode": randk_weight_mode,
                        "prior_floor": float(prior_floor),
                        "prune_fallback": prune_fallback,
                        "prune_strategy": prune_strategy_tag,
                        "protein_feat_mode": protein_feat_mode,
                        "esm_special_tokens": esm_special_tokens,
                        "esm_norm": esm_norm,
                        "esm_fallback": esm_fallback,
                        "use_knn_graph": bool(use_knn_graph),
                        "knn_setting": knn_setting,
                        "drug_knn_k": int(drug_knn_k or 0),
                        "prot_knn_k": int(prot_knn_k or 0),
                        "knn_metric": knn_metric,
                        "knn_symmetric": bool(knn_symmetric),
                        "knn_weight_temp": float(knn_weight_temp),
                        "cold_deg_th_drug": int(cold_deg_th_drug or 0),
                        "cold_deg_th_prot": int(cold_deg_th_prot or 0),
                        "cold_mode": str(cold_mode),
                        "cold_q": float(cold_q),
                        "cold_q_drug": float(cold_q_drug) if cold_q_drug is not None else float(cold_q),
                        "cold_q_prot": float(cold_q_prot) if cold_q_prot is not None else float(cold_q),
                        "cold_th_drug": float(cold_drug_th) if use_knn_graph else float(cold_deg_th_drug or 0),
                        "cold_th_prot": float(cold_prot_th) if use_knn_graph else float(cold_deg_th_prot or 0),
                        "deg_source": "GLOBAL_TRAIN_ONLY",
                        "use_physchem_feat": bool(use_physchem_feat),
                        "attn_path": attn_sig["attn_path"],
                        "attn_mtime": attn_sig["attn_mtime"],
                        "attn_size": attn_sig["attn_size"],
                        "atom_feat_version": ATOM_FEAT_VERSION,
                        "drug_desc_version": DRUG_DESC_VERSION,
                        "use_drug_desc": bool(use_drug_desc),
                        "use_gasteiger": bool(use_gasteiger),
                        "drug_desc_dim": int(drug_desc_dim),
                        "drug_desc_impute": DRUG_DESC_IMPUTE,
                        "drug_desc_missing_flag": bool(use_drug_desc),
                        "drug_desc_missing_count": int(drug_desc_missing_count),
                        "drug_desc_missing_ratio": float(drug_desc_missing_ratio),
                        "drug_feat_missing_count": int(feat_stat.get("missing_count", 0)) if feat_stat else 0,
                        "drug_feat_missing_ratio": float(feat_stat.get("missing_ratio", 0.0)) if feat_stat else 0.0,
                        "drug_feat_missing_impute": "train_mean" if (feat_stat and feat_stat.get("impute", False)) else "none",
                        "drug_feat_missing_flag": bool(feat_stat.get("add_missing_flag", False)) if feat_stat else False,
                        "drug_desc_mean_hash": drug_desc_mean_hash,
                        "drug_gnn_dim": int(drug_gnn_dim),
                        "atom_gnn_dim": int(atom_gnn_dim),
                        "drug_gnn_path": drug_gnn_sig["drug_gnn_path"],
                        "drug_gnn_mtime": drug_gnn_sig["drug_gnn_mtime"],
                        "drug_gnn_size": drug_gnn_sig["drug_gnn_size"],
                        "atom_gnn_path": atom_gnn_sig["atom_gnn_path"],
                        "atom_gnn_mtime": atom_gnn_sig["atom_gnn_mtime"],
                        "atom_gnn_size": atom_gnn_sig["atom_gnn_size"],
                        "prot_esm2_path": esm_sig["prot_esm2_path"],
                        "prot_esm2_mtime": esm_sig["prot_esm2_mtime"],
                        "prot_esm2_size": esm_sig["prot_esm2_size"],
                        "split_train_path": split_train_sig["split_train_path"],
                        "split_train_mtime": split_train_sig["split_train_mtime"],
                        "split_train_size": split_train_sig["split_train_size"],
                        "split_train_hash": split_train_sig["split_train_hash"],
                        "split_val_path": split_val_sig["split_val_path"],
                        "split_val_mtime": split_val_sig["split_val_mtime"],
                        "split_val_size": split_val_sig["split_val_size"],
                        "split_val_hash": split_val_sig["split_val_hash"],
                        "split_test_path": split_test_sig["split_test_path"],
                        "split_test_mtime": split_test_sig["split_test_mtime"],
                        "split_test_size": split_test_sig["split_test_size"],
                        "split_test_hash": split_test_sig["split_test_hash"],
                        "esm_mismatch_count": int(esm_mismatch_count),
                        "esm_missing_count": int(esm_missing_count),
                        "esm_unreliable_count": int(esm_unreliable_count),
                        "esm_mismatch_global_fallback": int(esm_mismatch_global_fallback),
                        "esm_mismatch_path": esm_mismatch_path or "",
                        "prot_knn_disabled_reason": prot_knn_disabled_reason or "",
                    },
                    f,
                )
    except Exception:
        pass

    num_drugs = len(drug_ids)
    num_prots = len(protein_ids)
    drug_atom_ptr, drug_atom_nodes = pack_ragged_int(drug_keep_atom_ids_new)
    prot_res_ptr, prot_res_nodes = pack_ragged_int(prot_keep_res_ids_new)
    return (train_edges, val_edges, test_edges, num_drugs, num_prots,
            H_atom, H_residue, G_atom, G_residue,
            features_atom, features_residue,
            atom_to_drug, residue_to_protein,
            atom_attn, residue_attn,
            atom_orig_pos, residue_orig_pos,
            drug_atom_ptr, drug_atom_nodes,
            prot_res_ptr, prot_res_nodes,
            prot_esm_missing, prot_esm_unreliable,
            drug_knn_edge_index, drug_knn_edge_weight,
            prot_knn_edge_index, prot_knn_edge_weight)

def load_and_construct_hypergraphs(
    dataset_name,
    data_root,
    node_level="atomic",
    psichic_attention_path=None,
    add_self_loop=True,
    protein_feat_mode="concat",
    esm_special_tokens="auto",
    esm_norm="per_protein_zscore",
    esm_fallback="onehot_only",
    esm_strict=False,
    use_physchem_feat=True,
    reuse_cache=False,
    prune_strategy_tag=None,
    atom_topk=None,
    res_topk=None,
    atom_randk=0,
    res_randk=0,
    randk_seed=None,
    randk_weight_mode="floor_prior",
    prior_floor=1e-4,
    use_knn_graph=False,
    knn_setting="inductive",
    cold_deg_th_drug=2,
    cold_deg_th_prot=2,
    cold_mode="th",
    cold_q=0.1,
    cold_q_drug=None,
    cold_q_prot=None,
    drug_knn_k=20,
    prot_knn_k=20,
    knn_metric="cosine",
    knn_symmetric=True,
    knn_weight_temp=0.1,
):
    """Atomic-only entry: atom/residue nodes with PSICHIC attention."""
    if str(node_level).lower() != "atomic":
        raise ValueError(
            f"Only atomic mode is supported now, got node_level={node_level!r}."
        )
    return load_and_construct_hypergraphs_atomic(
        dataset_name,
        data_root,
        psichic_attention_path,
        add_self_loop=add_self_loop,
        protein_feat_mode=protein_feat_mode,
        esm_special_tokens=esm_special_tokens,
        esm_norm=esm_norm,
        esm_fallback=esm_fallback,
        esm_strict=esm_strict,
        use_physchem_feat=use_physchem_feat,
        reuse_cache=reuse_cache,
        prune_strategy_tag=prune_strategy_tag,
        atom_topk=atom_topk,
        res_topk=res_topk,
        atom_randk=atom_randk,
        res_randk=res_randk,
        randk_seed=randk_seed,
        randk_weight_mode=randk_weight_mode,
        prior_floor=prior_floor,
        use_knn_graph=use_knn_graph,
        knn_setting=knn_setting,
        cold_deg_th_drug=cold_deg_th_drug,
        cold_deg_th_prot=cold_deg_th_prot,
        cold_mode=cold_mode,
        cold_q=cold_q,
        cold_q_drug=cold_q_drug,
        cold_q_prot=cold_q_prot,
        drug_knn_k=drug_knn_k,
        prot_knn_k=prot_knn_k,
        knn_metric=knn_metric,
        knn_symmetric=knn_symmetric,
        knn_weight_temp=knn_weight_temp,
    )


if __name__ == "__main__":
    dataset_name = "DTI-main-data/split-random-hyper"
    data_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "data"))
    dataset_dir = os.path.join(data_root, dataset_name)

    def _auto_detect_attention_path(base_dir):
        env_path = os.environ.get("HGACN_PSICHIC_ATTENTION", "").strip()
        if env_path and os.path.exists(env_path):
            return env_path
        candidates = [
            "psichic_attention.pkl",
            "psichic_attn.pkl",
            "psichic_attention.pickle",
            "psichic_attn.pickle",
            "attention.pkl",
            "attn.pkl",
            "psichic_attention.npy",
            "psichic_attn.npy",
            "attention.npy",
            "attn.npy",
        ]
        for root in (base_dir, os.path.join(base_dir, "processed"), os.path.join(base_dir, "processed_atomic")):
            for name in candidates:
                path = os.path.join(root, name)
                if os.path.exists(path):
                    return path
            for pattern in ("*psichic*attn*.pkl", "*psichic*attention*.pkl", "*attn*.pkl"):
                hits = sorted(glob.glob(os.path.join(root, pattern)))
                if hits:
                    return hits[0]
        return None

    psichic_attention_path = _auto_detect_attention_path(dataset_dir)
    if psichic_attention_path:
        print(f"[INFO] Auto-detected PSICHIC attention: {psichic_attention_path}")
    else:
        print("[INFO] PSICHIC attention not found; proceeding without attention.")

    # Atomic-level nodes with DTI interaction hyperedges.
    load_and_construct_hypergraphs(
        dataset_name,
        data_root,
        node_level="atomic",
        psichic_attention_path=psichic_attention_path,
    )

