"""Thermodynamic stability, atomic BUNS, MolProbity clashscore, and variant scoring (FoldX / ESM-1v / heuristic)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Dict, List, Literal, Optional, Set, Tuple

import numpy as np
from scipy.spatial import cKDTree

from .constants import (
    ATOMIC_BUNS_SASA_CUTOFF,
    ATOMIC_HBOND_DIST_MAX,
    BURIAL_TRANSFER_PRIOR,
    HELIX_PENALTY,
    MOLPROBITY_OVERLAP_CUTOFF,
    ONE_TO_THREE,
    RESIDUE_VOLUME,
    SHEET_PENALTY,
    SIDECHAIN_HBOND_ACCEPTORS,
    SIDECHAIN_HBOND_DONORS,
)
from .io_parser import ResidueData, StructureData, compute_intrinsic_molecular_frame, write_sanitized_pdb
from .sap import compute_shrake_rupley_sasa
from .schema import StabilityMetrics, VariantScore


@dataclass
class StabilityResult:
    """Internal structural quality, BUNS, clash, and stability outputs."""

    secondary_structures: List[Literal["helix", "sheet", "coil"]]
    res_stability_score: np.ndarray
    res_ala_score: np.ndarray
    res_clashes: np.ndarray
    res_buns: np.ndarray
    res_packing: np.ndarray
    stable_core_indices: List[int]
    destabilized_indices: List[int]
    metrics: StabilityMetrics


_MUTATION_RE = re.compile(
    r"^(?:([A-Za-z0-9]):)?([ACDEFGHIKLMNPQRSTVWY])(-?\d+)([A-Za-z]?)([ACDEFGHIKLMNPQRSTVWY])$"
)


def parse_mutation_spec(spec: str, default_chain: str = "A") -> Tuple[str, str, int, str, str]:
    """Parse mutation strings like 'A:I19K', 'I19K', or 'A:W100AF' (with insertion code)."""
    cleaned = spec.strip()
    m = _MUTATION_RE.match(cleaned)
    if not m:
        raise ValueError(
            f"Invalid mutation spec '{spec}'. Expected format like 'A:I19K', 'I19K', or 'A:W100AF'."
        )
    chain = m.group(1) or default_chain
    wt = m.group(2)
    pos = int(m.group(3))
    ins_code = m.group(4) or ""
    mut = m.group(5)
    return chain, wt, pos, ins_code, mut


def _dihedral_deg(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    b0 = -1.0 * (p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2
    b1_norm = b1 / (np.linalg.norm(b1) + 1e-12)
    v = b0 - np.dot(b0, b1_norm) * b1_norm
    w = b2 - np.dot(b2, b1_norm) * b1_norm
    x = np.dot(v, w)
    y = np.dot(np.cross(b1_norm, v), w)
    return float(np.degrees(np.arctan2(y, x)))


def _compute_backbone_ss(
    structure: StructureData,
) -> Tuple[List[Literal["helix", "sheet", "coil"]], List[Optional[Tuple[float, float]]]]:
    """Assign secondary structure (helix/sheet/coil) from (phi, psi) dihedrals and i->i+4 H-bonds."""
    n_res = len(structure.residues)
    ss_list: List[Literal["helix", "sheet", "coil"]] = ["coil"] * n_res
    dihedrals: List[Optional[Tuple[float, float]]] = [None] * n_res

    for i, res in enumerate(structure.residues):
        prev_res = structure.residues[i - 1] if i > 0 and structure.residues[i - 1].chain_id == res.chain_id else None
        next_res = structure.residues[i + 1] if i + 1 < n_res and structure.residues[i + 1].chain_id == res.chain_id else None

        if prev_res is None or next_res is None:
            continue
        if (
            prev_res.c_coord is None
            or res.n_coord is None
            or res.ca_coord is None
            or res.c_coord is None
            or next_res.n_coord is None
        ):
            continue

        phi = _dihedral_deg(prev_res.c_coord, res.n_coord, res.ca_coord, res.c_coord)
        psi = _dihedral_deg(res.n_coord, res.ca_coord, res.c_coord, next_res.n_coord)
        dihedrals[i] = (phi, psi)

        if -100.0 <= phi <= -30.0 and -75.0 <= psi <= -10.0:
            ss_list[i] = "helix"
        elif -165.0 <= phi <= -75.0 and (75.0 <= psi <= 175.0 or -180.0 <= psi <= -150.0):
            ss_list[i] = "sheet"

    smoothed = list(ss_list)
    for i in range(1, n_res - 1):
        if ss_list[i] == "coil" and ss_list[i - 1] == ss_list[i + 1] and ss_list[i - 1] != "coil":
            smoothed[i] = ss_list[i - 1]
    for i in range(n_res):
        if smoothed[i] == "coil":
            if (
                i + 4 < n_res
                and smoothed[i + 1] == "helix"
                and smoothed[i + 2] == "helix"
                and structure.residues[i].chain_id == structure.residues[i + 4].chain_id
                and structure.residues[i].o_coord is not None
                and structure.residues[i + 4].n_coord is not None
                and float(np.linalg.norm(structure.residues[i].o_coord - structure.residues[i + 4].n_coord)) <= 3.55
            ):
                smoothed[i] = "helix"
            elif (
                i - 4 >= 0
                and smoothed[i - 1] == "helix"
                and smoothed[i - 2] == "helix"
                and structure.residues[i].chain_id == structure.residues[i - 4].chain_id
                and structure.residues[i - 4].o_coord is not None
                and structure.residues[i].n_coord is not None
                and float(np.linalg.norm(structure.residues[i - 4].o_coord - structure.residues[i].n_coord)) <= 3.55
            ):
                smoothed[i] = "helix"
    return smoothed, dihedrals


def _ramachandran_outlier_flags(
    dihedrals: List[Optional[Tuple[float, float]]],
    residues: List[ResidueData],
) -> Tuple[np.ndarray, np.ndarray]:
    """Identify backbone Ramachandran outliers and compute dimensionless strain."""
    n_res = len(residues)
    outliers = np.zeros(n_res, dtype=bool)
    strain = np.zeros(n_res, dtype=np.float64)

    for i, res in enumerate(residues):
        phi_psi = dihedrals[i]
        if phi_psi is None or res.resname == "GLY":
            continue
        phi, psi = phi_psi
        if phi > 0.0:
            # Left-handed alpha region (~35..95, 10..80) allowed for ASN/ASP/HIS/etc., outside is outlier
            if 35.0 <= phi <= 95.0 and -10.0 <= psi <= 85.0:
                strain[i] = 0.35 if res.resname in {"ASN", "ASP"} else 0.85
            elif 45.0 <= phi <= 115.0 and -175.0 <= psi <= -110.0:
                strain[i] = 0.70
            else:
                outliers[i] = True
                strain[i] = 1.80
        elif res.resname == "PRO" and not (-100.0 <= phi <= -25.0):
            outliers[i] = True
            strain[i] = 1.60
    return outliers, strain


def _classify_polar_atoms(
    structure: StructureData,
    disulfide_cys: Set[int],
) -> Tuple[Set[int], Set[int], Set[int], Dict[int, np.ndarray]]:
    """Build typed H-bond donor/acceptor sets, BUNS-eligible polar indices, and amide pseudo-H."""
    donors: Set[int] = set()
    acceptors: Set[int] = set()
    buns_eligible: Set[int] = set()
    pseudo_h: Dict[int, np.ndarray] = {}

    n_res = len(structure.residues)
    for i, res in enumerate(structure.residues):
        sc_don = SIDECHAIN_HBOND_DONORS.get(res.resname, frozenset())
        sc_acc = SIDECHAIN_HBOND_ACCEPTORS.get(res.resname, frozenset())
        for idx in res.atom_indices:
            aname = structure.atoms[idx].name
            # Disulfide CYS SG is covalently bonded and non-ionizable/non-BUNS
            if res.index in disulfide_cys and aname == "SG":
                continue

            if aname == "N" and res.resname != "PRO":
                donors.add(idx)
                buns_eligible.add(idx)
                prev_res = structure.residues[i - 1] if i > 0 and structure.residues[i - 1].chain_id == res.chain_id else None
                if prev_res is not None and prev_res.c_coord is not None and res.ca_coord is not None:
                    v1 = res.n_coord - prev_res.c_coord
                    v2 = res.n_coord - res.ca_coord
                    bisec = v1 / (np.linalg.norm(v1) + 1e-12) + v2 / (np.linalg.norm(v2) + 1e-12)
                    norm = float(np.linalg.norm(bisec))
                    if norm > 1e-6:
                        pseudo_h[idx] = res.n_coord + 1.01 * (bisec / norm)
            elif aname in sc_don:
                donors.add(idx)
                buns_eligible.add(idx)

            if aname in {"O", "OXT"} or aname in sc_acc:
                acceptors.add(idx)
                # Methionine thioether SD is weak/non-BUNS in Rosetta/Coventry conventions
                if not (res.resname == "MET" and aname == "SD"):
                    buns_eligible.add(idx)

            # Crystallographic Asn/Gln/His sidechain amide/imidazole flip ambiguity (Word et al. REDUCE)
            if (res.resname == "ASN" and aname in {"OD1", "ND2"}) or (
                res.resname == "GLN" and aname in {"OE1", "NE2"}
            ) or (res.resname == "HIS" and aname in {"ND1", "NE2"}):
                donors.add(idx)
                acceptors.add(idx)

    return donors, acceptors, buns_eligible, pseudo_h


def _is_valid_hbond_pair(
    structure: StructureData,
    idx_a: int,
    idx_b: int,
    donors: Set[int],
    acceptors: Set[int],
    pseudo_h: Dict[int, np.ndarray],
) -> bool:
    """Check chemical donor-acceptor compatibility and local geometry between two heavy atoms."""
    ra = int(structure.atom_residue_idx[idx_a])
    rb = int(structure.atom_residue_idx[idx_b])
    if ra == rb:
        return False

    res_a = structure.residues[ra]
    res_b = structure.residues[rb]
    aname = structure.atoms[idx_a].name
    bname = structure.atoms[idx_b].name

    # Covalently adjacent peptide O(i)--N(i+1) across C(i) is 1-3 bonded, NOT an H-bond
    if res_a.chain_id == res_b.chain_id and abs(ra - rb) == 1:
        if {aname, bname} <= {"N", "O"}:
            return False

    a_don_b_acc = idx_a in donors and idx_b in acceptors
    b_don_a_acc = idx_b in donors and idx_a in acceptors
    if not (a_don_b_acc or b_don_a_acc):
        return False

    dist = float(np.linalg.norm(structure.coords[idx_a] - structure.coords[idx_b]))
    if dist < 2.30 or dist > ATOMIC_HBOND_DIST_MAX:
        return False

    # Check directional amide N-H...acceptor angle when pseudo-H exists (>= 90 deg for strained turns)
    for d_idx, acc_idx in ((idx_a, idx_b), (idx_b, idx_a)):
        if d_idx in donors and acc_idx in acceptors and d_idx in pseudo_h:
            h_pt = pseudo_h[d_idx]
            v_hn = structure.coords[d_idx] - h_pt
            v_ha = structure.coords[acc_idx] - h_pt
            cos_ang = float(np.dot(-v_hn, v_ha) / ((np.linalg.norm(v_hn) * np.linalg.norm(v_ha)) + 1e-12))
            if cos_ang < 0.0:
                return False

    # Check C=O...donor angle for backbone carbonyl acceptors (>= 85 deg)
    for acc_idx, d_idx in ((idx_a, idx_b), (idx_b, idx_a)):
        if structure.atoms[acc_idx].name == "O" and acc_idx in acceptors and d_idx in donors:
            acc_res = structure.residues[int(structure.atom_residue_idx[acc_idx])]
            if acc_res.c_coord is not None:
                v_co = structure.coords[acc_idx] - acc_res.c_coord
                v_od = structure.coords[d_idx] - structure.coords[acc_idx]
                cos_co = float(np.dot(v_co, v_od) / ((np.linalg.norm(v_co) * np.linalg.norm(v_od)) + 1e-12))
                if cos_co < -0.10:
                    return False

    return True


def _compute_atomic_buns_and_bb_hbond(
    structure: StructureData,
    atom_sasa: np.ndarray,
    pocket_sasa: np.ndarray,
    donors: Set[int],
    acceptors: Set[int],
    buns_eligible: Set[int],
    pseudo_h: Dict[int, np.ndarray],
) -> Tuple[float, np.ndarray, List[str], np.ndarray]:
    """Compute Coventry/Baker atomic BUNS (excluding water-pocket exposed, disulfide SG, and Met SD)."""
    n_res = len(structure.residues)
    coords = structure.coords
    tree = cKDTree(coords)

    polar_indices = sorted(buns_eligible)
    buns_atoms: List[str] = []
    res_buns = np.zeros(n_res, dtype=np.int32)
    bb_satisfied = np.zeros(n_res, dtype=bool)

    atom_is_satisfied: Dict[int, bool] = {}
    for a_idx in polar_indices:
        # Satisfied by bulk solvent or tight water-pocket accessibility (probe 1.05 A)
        if atom_sasa[a_idx] >= ATOMIC_BUNS_SASA_CUTOFF or pocket_sasa[a_idx] >= 1.25:
            atom_is_satisfied[a_idx] = True
            continue

        nbrs = tree.query_ball_point(coords[a_idx], r=ATOMIC_HBOND_DIST_MAX)
        satisfied = False
        for b_idx in nbrs:
            if b_idx == a_idx or (b_idx not in donors and b_idx not in acceptors):
                continue
            if _is_valid_hbond_pair(structure, a_idx, b_idx, donors, acceptors, pseudo_h):
                satisfied = True
                break
        atom_is_satisfied[a_idx] = satisfied
        if not satisfied:
            res_idx = int(structure.atom_residue_idx[a_idx])
            res_buns[res_idx] += 1
            buns_atoms.append(f"{structure.residues[res_idx].short_label}:{structure.atoms[a_idx].name}")

    for i, res in enumerate(structure.residues):
        n_idx = next((idx for idx in res.backbone_atom_indices if structure.atoms[idx].name == "N"), None)
        o_idx = next((idx for idx in res.backbone_atom_indices if structure.atoms[idx].name == "O"), None)
        n_ok = True if (n_idx is None or res.resname == "PRO") else atom_is_satisfied.get(n_idx, True)
        o_ok = True if o_idx is None else atom_is_satisfied.get(o_idx, True)
        bb_satisfied[i] = n_ok and o_ok

    sat_rate = float(np.mean(bb_satisfied)) if n_res else 1.0
    return sat_rate, bb_satisfied, buns_atoms, res_buns


def _compute_molprobity_clashes_and_packing(
    structure: StructureData,
    res_rsasa: np.ndarray,
    donors: Set[int],
    acceptors: Set[int],
    pseudo_h: Dict[int, np.ndarray],
) -> Tuple[np.ndarray, int, float, np.ndarray, np.ndarray]:
    """Detect MolProbity heavy-atom clashes (overlap >= 0.40 A), clashscore, and packing."""
    n_res = len(structure.residues)
    res_clashes = np.zeros(n_res, dtype=np.int32)
    res_lj = np.zeros(n_res, dtype=np.float64)
    res_packing = np.zeros(n_res, dtype=np.float64)

    n_atoms = len(structure.atoms)
    if n_atoms == 0:
        return res_clashes, 0, 0.0, res_packing, res_lj

    coords = structure.coords
    radii = structure.vdw_radii
    atom_res = structure.atom_residue_idx
    tree = cKDTree(coords)
    pairs = tree.query_pairs(r=4.6)

    contact_counts = np.zeros(n_res, dtype=np.int32)
    total_clash_pairs = 0

    for idx_a, idx_b in pairs:
        ra = int(atom_res[idx_a])
        rb = int(atom_res[idx_b])
        if ra == rb:
            continue

        res_a = structure.residues[ra]
        res_b = structure.residues[rb]
        aname = structure.atoms[idx_a].name
        bname = structure.atoms[idx_b].name

        # Skip covalent peptide link 1-2/1-3/1-4 atoms across verified peptide bonds (never skip CB-CB)
        if res_a.chain_id == res_b.chain_id and abs(ra - rb) == 1:
            r_prev, r_next = (res_a, res_b) if ra < rb else (res_b, res_a)
            is_peptide_bonded = (
                r_prev.c_coord is not None
                and r_next.n_coord is not None
                and float(np.linalg.norm(r_prev.c_coord - r_next.n_coord)) <= 1.85
            )
            if is_peptide_bonded and not (aname == "CB" and bname == "CB"):
                adj_a = {"N", "CA", "C", "O", "CB"} | ({"CD", "CG"} if res_a.resname == "PRO" else set())
                adj_b = {"N", "CA", "C", "O", "CB"} | ({"CD", "CG"} if res_b.resname == "PRO" else set())
                if aname in adj_a and bname in adj_b:
                    continue

        dist = float(np.linalg.norm(coords[idx_a] - coords[idx_b]))
        # Skip valid covalent disulfide SG-SG bonds (1.80 - 2.35 A)
        if aname == "SG" and bname == "SG" and 1.80 <= dist <= 2.35:
            continue

        r_sum = float(radii[idx_a] + radii[idx_b])
        overlap = r_sum - dist

        # MolProbity grants extra overlap tolerance (~0.60 A) for true donor-acceptor H-bonds
        is_hbond = _is_valid_hbond_pair(structure, idx_a, idx_b, donors, acceptors, pseudo_h)
        clash_threshold = 0.65 if is_hbond else MOLPROBITY_OVERLAP_CUTOFF

        if overlap >= clash_threshold:
            total_clash_pairs += 1
            res_clashes[ra] += 1
            res_clashes[rb] += 1
            excess = overlap - clash_threshold
            penalty = min(4.5, 4.0 * ((excess + 0.25) ** 2))
            res_lj[ra] += 0.5 * penalty
            res_lj[rb] += 0.5 * penalty
        else:
            contact_counts[ra] += 1
            contact_counts[rb] += 1
            ratio = r_sum / max(dist, 2.2)
            attr = -0.04 * (ratio ** 3)
            res_lj[ra] += 0.5 * attr
            res_lj[rb] += 0.5 * attr

    for i, res in enumerate(structure.residues):
        expected_contacts = max(8.0, 0.14 * RESIDUE_VOLUME.get(res.resname, 130.0))
        burial_factor = 1.0 - min(0.85, float(res_rsasa[i]))
        raw_pack = min(1.0, contact_counts[i] / max(4.0, expected_contacts * (0.4 + 0.6 * burial_factor)))
        if res_clashes[i] > 0:
            raw_pack *= max(0.2, 1.0 - 0.25 * res_clashes[i])
        res_packing[i] = raw_pack

    clashscore = round((total_clash_pairs * 1000.0) / max(n_atoms, 1), 2)
    return res_clashes, total_clash_pairs, clashscore, res_packing, res_lj


def detect_available_stability_backends(foldx_binary: str = "foldx") -> List[str]:
    """Return list of stability/variant backends available in the environment."""
    backends: List[str] = ["biophysical"]
    if shutil.which(foldx_binary):
        backends.append("foldx")
    try:
        import torch  # noqa: F401
        import esm  # noqa: F401

        backends.append("esm1v")
    except ImportError:
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401

            backends.append("esm1v")
        except ImportError:
            pass
    return backends


def _classify_variant(score_val: float) -> Literal["favorable_or_stabilizing", "neutral", "mildly_destabilizing", "strongly_destabilizing"]:
    if score_val <= -0.5:
        return "favorable_or_stabilizing"
    if score_val < 0.5:
        return "neutral"
    if score_val < 1.8:
        return "mildly_destabilizing"
    return "strongly_destabilizing"


def _score_point_mutation_heuristic(
    res: ResidueData,
    mutant_aa1: str,
    rsasa: float,
    ss: Literal["helix", "sheet", "coil"],
    in_salt_bridge: bool,
    in_disulfide: bool,
    packing_score: float,
    sap_val: float,
) -> Tuple[float, str]:
    """Compute dimensionless structural stability heuristic (heuristic_au; >0 unfavorable, <0 favorable)."""
    wt_3 = res.resname
    mut_3 = ONE_TO_THREE[mutant_aa1]
    core_burial = max(0.0, (0.35 - min(0.35, rsasa)) / 0.35)
    solv_burial = max(0.0, (0.60 - min(0.60, rsasa)) / 0.60)
    exposure = min(1.0, rsasa / 0.55)

    if wt_3 == mut_3:
        return 0.0, "Synonymous residue (wildtype)."

    reasons: List[str] = []

    g_wt = BURIAL_TRANSFER_PRIOR.get(wt_3, 0.0)
    g_mut = BURIAL_TRANSFER_PRIOR.get(mut_3, 0.0)
    wt_env = solv_burial * g_wt + exposure * max(0.0, -0.55 * g_wt)
    mut_env = solv_burial * g_mut + exposure * max(0.0, -0.55 * g_mut)
    d_solv = mut_env - wt_env

    if core_burial >= 0.45 and g_mut - g_wt > 1.2:
        reasons.append("reduces buried hydrophobic stabilization")
    elif exposure >= 0.65 and sap_val >= 0.10 and g_mut > 0.5 and g_wt < -1.0:
        reasons.append("replaces exposed hydrophobic patch residue with polar/charged side chain")

    vol_wt = RESIDUE_VOLUME.get(wt_3, 130.0)
    vol_mut = RESIDUE_VOLUME.get(mut_3, 130.0)
    d_vol = vol_mut - vol_wt
    d_pack = 0.0
    if core_burial > 0.05:
        if d_vol < -10.0:
            d_pack = 0.032 * abs(d_vol) * core_burial * (0.6 + 0.4 * packing_score)
        elif d_vol > 18.0:
            d_pack = 0.042 * (d_vol - 12.0) * (core_burial ** 1.2)

    d_ss = 0.0
    if ss == "helix":
        d_ss = (HELIX_PENALTY.get(mut_3, 0.5) - HELIX_PENALTY.get(wt_3, 0.5)) * 0.85
        if mut_3 == "PRO":
            d_ss += 1.25
            reasons.append("proline disrupts alpha-helix H-bond")
    elif ss == "sheet":
        d_ss = (SHEET_PENALTY.get(mut_3, 0.5) - SHEET_PENALTY.get(wt_3, 0.5)) * 0.75
        if mut_3 == "PRO":
            d_ss += 0.80
            reasons.append("proline disrupts beta-sheet H-bond")

    d_net = 0.0
    if in_disulfide and mut_3 != "CYS":
        d_net += 2.65
        reasons.append("breaks covalent disulfide bond")
    if in_salt_bridge and mut_3 not in {"ARG", "LYS", "HIS", "ASP", "GLU"}:
        d_net += 0.95
        reasons.append("disrupts electrostatic salt bridge")

    d_chg = 0.0
    charged = {"ARG", "LYS", "ASP", "GLU"}
    if core_burial > 0.30 and mut_3 in charged and wt_3 not in charged:
        d_chg += 2.5 * (core_burial ** 1.3)
        reasons.append("introduces uncompensated buried charge")

    total_score = d_solv + d_pack + d_ss + d_net + d_chg

    # Consistent rationale: never report cavity penalty if net score is stabilizing
    if d_pack > 0.20 and d_vol < -10.0 and total_score > 0.25:
        reasons.append("creates core packing cavity")
    elif d_pack > 0.20 and d_vol > 18.0 and total_score > 0.25:
        reasons.append("steric overpacking in buried core")
    elif d_pack > 0.20 and total_score <= -0.20:
        reasons.append("relieves unfavorable buried polar desolvation despite smaller side-chain volume")

    if not reasons:
        if total_score <= -0.5:
            reasons.append("favorable surface/solvation heuristic balance")
        elif total_score >= 0.5:
            reasons.append("unfavorable solvation or backbone propensity heuristic")
        else:
            reasons.append("conservative structural heuristic substitution")

    return round(float(total_score), 3), "; ".join(reasons)


def _run_foldx_ddg(
    structure: StructureData,
    variants: List[Tuple[str, str, int, str, str]],
    foldx_binary: str,
) -> Optional[Tuple[float, Dict[str, float]]]:
    """Run FoldX RepairPDB + Stability + BuildModel (true kcal/mol) on PDB or sanitized mmCIF."""
    exe = shutil.which(foldx_binary)
    if not exe:
        return None
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            pdb_copy = tmp / "input.pdb"
            write_sanitized_pdb(structure, pdb_copy)

            subprocess.run(
                [exe, "--command=RepairPDB", f"--pdb={pdb_copy.name}"],
                cwd=tmp,
                check=True,
                capture_output=True,
                text=True,
                timeout=180,
            )
            repaired = tmp / "input_Repair.pdb"
            target_pdb = repaired if repaired.exists() else pdb_copy

            subprocess.run(
                [exe, "--command=Stability", f"--pdb={target_pdb.name}"],
                cwd=tmp,
                check=True,
                capture_output=True,
                text=True,
                timeout=120,
            )

            dg_total = 0.0
            stab_fx = list(tmp.glob("*_0_ST.fxout"))
            if stab_fx:
                for line in reversed(stab_fx[0].read_text(encoding="utf-8", errors="replace").splitlines()):
                    parts = line.split("\t")
                    if len(parts) >= 2:
                        try:
                            dg_total = float(parts[1])
                            break
                        except ValueError:
                            continue

            mut_ddgs: Dict[str, float] = {}
            if variants:
                mut_lines = [f"{wt}{ch}{pos}{ins}{mut};" for ch, wt, pos, ins, mut in variants]
                (tmp / "individual_list.txt").write_text("\n".join(mut_lines) + "\n", encoding="utf-8")
                subprocess.run(
                    [exe, "--command=BuildModel", f"--pdb={target_pdb.name}", "--mutant-file=individual_list.txt"],
                    cwd=tmp,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=180,
                )
                dif_fx = list(tmp.glob("Dif_*.fxout"))
                if dif_fx:
                    lines = [l for l in dif_fx[0].read_text(encoding="utf-8", errors="replace").splitlines() if l.strip()]
                    data_rows = [l for l in lines if target_pdb.stem in l and not l.startswith("Pdb")]
                    if len(data_rows) == len(variants):
                        for (ch, wt, pos, ins, mut), row in zip(variants, data_rows):
                            parts = row.split("\t")
                            if len(parts) >= 2:
                                mut_ddgs[f"{ch}:{wt}{pos}{ins}{mut}"] = float(parts[1])

            return dg_total, mut_ddgs
    except (subprocess.SubprocessError, OSError, ValueError):
        return None


def _run_esm1v_zero_shot(
    structure: StructureData,
    variants: List[Tuple[str, str, int, str, str]],
    model_name: str,
) -> Optional[Dict[str, float]]:
    """Compute ESM-1v masked marginal log-likelihood ratio (log P(wt) - log P(mut), dimensionless)."""
    if not variants:
        return {}
    try:
        import torch
    except ImportError:
        return None

    chain_seqs = structure.sequence_by_chain
    chain_pos_map: Dict[Tuple[str, int, str], int] = {}
    chain_counters: Dict[str, int] = {}
    for res in structure.residues:
        idx_in_chain = chain_counters.get(res.chain_id, 0)
        chain_pos_map[res.key] = idx_in_chain
        chain_counters[res.chain_id] = idx_in_chain + 1

    llr_map: Dict[str, float] = {}

    # 1. Try HuggingFace transformers with local_files_only=True
    try:
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        model = AutoModelForMaskedLM.from_pretrained(model_name, local_files_only=True)
        model.eval()
        with torch.no_grad():
            for ch, wt, pos, ins, mut in variants:
                seq = chain_seqs.get(ch)
                seq_idx = chain_pos_map.get((ch, pos, ins))
                if seq is None or seq_idx is None:
                    continue
                masked_seq = seq[:seq_idx] + tokenizer.mask_token + seq[seq_idx + 1 :]
                inputs = tokenizer(masked_seq, return_tensors="pt")
                mask_positions = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)
                logits = model(**inputs).logits
                log_probs = torch.log_softmax(logits[0, mask_positions[1][0]], dim=-1)
                wt_id = tokenizer.convert_tokens_to_ids(wt)
                mut_id = tokenizer.convert_tokens_to_ids(mut)
                llr_map[f"{ch}:{wt}{pos}{ins}{mut}"] = round(float((log_probs[wt_id] - log_probs[mut_id]).item()), 3)
        return llr_map
    except Exception:
        pass

    # 2. Fallback to fair-esm (`import esm`) if locally cached
    try:
        import esm

        short_name = model_name.split("/")[-1]
        model, alphabet = esm.pretrained.load_model_and_alphabet(short_name)
        model.eval()
        batch_converter = alphabet.get_batch_converter()
        mask_idx = alphabet.mask_idx
        with torch.no_grad():
            for ch, wt, pos, ins, mut in variants:
                seq = chain_seqs.get(ch)
                seq_idx = chain_pos_map.get((ch, pos, ins))
                if seq is None or seq_idx is None:
                    continue
                _, _, tokens = batch_converter([("protein", seq)])
                tokens[0, seq_idx + 1] = mask_idx
                logits = model(tokens)["logits"]
                log_probs = torch.log_softmax(logits[0, seq_idx + 1], dim=-1)
                wt_id = alphabet.get_idx(wt)
                mut_id = alphabet.get_idx(mut)
                llr_map[f"{ch}:{wt}{pos}{ins}{mut}"] = round(float((log_probs[wt_id] - log_probs[mut_id]).item()), 3)
        return llr_map
    except Exception:
        return None


def compute_stability_metrics(
    structure: StructureData,
    atom_sasa: np.ndarray,
    res_rsasa: np.ndarray,
    res_sap_r5: np.ndarray,
    res_charges: np.ndarray,
    disulfide_pairs: List[Tuple[int, int]],
    salt_bridge_pairs: List[Tuple[int, int]],
    variants: Optional[List[str]] = None,
    backend_preference: Literal["auto", "foldx", "esm1v", "biophysical"] = "auto",
    run_alanine_scan: bool = True,
    foldx_binary: str = "foldx",
    esm_model_name: str = "facebook/esm1v_t33_650M_UR90S_1",
    rsasa_core_threshold: float = 0.18,
) -> StabilityResult:
    """Evaluate atomic BUNS, MolProbity clashes, packing, and calibrated/heuristic stability."""
    n_res = len(structure.residues)
    ss_list, dihedrals = _compute_backbone_ss(structure)
    available_backends = detect_available_stability_backends(foldx_binary)

    mean_plddt: Optional[float] = None
    low_plddt_fraction: Optional[float] = None
    if structure.plddt_detected:
        b_arr = np.array([r.mean_b_factor for r in structure.residues], dtype=np.float64)
        mean_plddt = round(float(np.mean(b_arr)), 2)
        low_plddt_fraction = round(float(np.mean(b_arr < 70.0)), 4)

    # Strict backbone-only refusal per Coventry / Baker Lab filtering principles
    if structure.is_backbone_only:
        zeros_f = np.zeros(n_res, dtype=np.float64)
        zeros_i = np.zeros(n_res, dtype=np.int32)
        metrics = StabilityMetrics(
            calculable=False,
            status_reason=(
                f"INVALID_BACKBONE_ONLY: sidechain completeness {structure.sidechain_completeness:.1%} (<80%). "
                f"Atomic BUNS, sidechain clashscore, core packing, and variant ddG cannot be evaluated without sidechains."
            ),
            backend_used="none",
            available_backends=available_backends,
            score_value=None,
            score_per_residue=None,
            score_unit="not_applicable",
            is_calibrated_free_energy=False,
            delta_g_fold_kcal_mol=None,
            mean_plddt=mean_plddt,
            low_plddt_fraction=low_plddt_fraction,
            packing_score=0.0,
            hbond_satisfaction_rate=0.0,
            steric_clash_count=0,
            molprobity_heavy_clashscore=0.0,
            buns_count=0,
            buns_atoms=[],
            ramachandran_outlier_count=0,
            disulfide_bond_count=len(disulfide_pairs),
            salt_bridge_count=len(salt_bridge_pairs),
            stable_core_residues=[],
            destabilized_residues=[],
            alanine_scan=[],
            variant_scores=[],
        )
        return StabilityResult(
            secondary_structures=ss_list,
            res_stability_score=zeros_f,
            res_ala_score=zeros_f,
            res_clashes=zeros_i,
            res_buns=zeros_i,
            res_packing=zeros_f,
            stable_core_indices=[],
            destabilized_indices=[],
            metrics=metrics,
        )

    disulfide_set = {idx for pair in disulfide_pairs for idx in pair}
    salt_bridge_set = {idx for pair in salt_bridge_pairs for idx in pair}

    mol_frame = compute_intrinsic_molecular_frame(structure.coords, structure.center_of_mass)
    pocket_sasa = compute_shrake_rupley_sasa(
        structure.coords,
        structure.vdw_radii,
        probe_radius=1.05,
        n_sphere_points=256,
        molecular_frame=mol_frame,
    )
    donors, acceptors, buns_eligible, pseudo_h = _classify_polar_atoms(structure, disulfide_set)
    hbond_sat_rate, bb_satisfied, buns_atoms, res_buns = _compute_atomic_buns_and_bb_hbond(
        structure, atom_sasa, pocket_sasa, donors, acceptors, buns_eligible, pseudo_h
    )
    res_clashes, total_clash_pairs, clashscore, res_packing, res_lj = _compute_molprobity_clashes_and_packing(
        structure, res_rsasa, donors, acceptors, pseudo_h
    )
    rama_outliers, rama_strain = _ramachandran_outlier_flags(dihedrals, structure.residues)

    # Dimensionless per-residue heuristic stability index (negative = stabilizing, positive = strain)
    res_stab = np.zeros(n_res, dtype=np.float64)
    for i, res in enumerate(structure.residues):
        burial = max(0.0, 1.0 - min(1.0, float(res_rsasa[i])))
        exposure = min(1.0, float(res_rsasa[i]))
        g_transfer = BURIAL_TRANSFER_PRIOR.get(res.resname, 0.0)
        g_solv = 0.65 * burial * g_transfer + 0.35 * exposure * max(0.0, -g_transfer)
        g_hb = -0.35 if bb_satisfied[i] else +0.55
        g_hb += 1.20 * float(res_buns[i])
        g_ss = 0.0
        if ss_list[i] == "helix":
            g_ss = 0.25 * (HELIX_PENALTY.get(res.resname, 0.4) - 0.35)
        elif ss_list[i] == "sheet":
            g_ss = 0.25 * (SHEET_PENALTY.get(res.resname, 0.4) - 0.35)
        g_elec = 0.0
        if i in salt_bridge_set:
            g_elec -= 0.35
        if i in disulfide_set:
            g_elec -= 1.25
        if burial > 0.72 and abs(res_charges[i]) > 0.5 and i not in salt_bridge_set:
            g_elec += 1.65 * (burial ** 1.5)
        res_stab[i] = g_solv + float(res_lj[i]) + g_hb + float(rama_strain[i]) + g_ss + g_elec

    heuristic_total = float(np.sum(res_stab))
    heuristic_per_res = heuristic_total / max(n_res, 1)

    # Full-length in silico Alanine scan (heuristic_au)
    res_ala_score = np.zeros(n_res, dtype=np.float64)
    ala_scan_items: List[VariantScore] = []
    for i, res in enumerate(structure.residues):
        if res.aa1 == "A":
            continue
        score_ala, rationale = _score_point_mutation_heuristic(
            res=res,
            mutant_aa1="A",
            rsasa=float(res_rsasa[i]),
            ss=ss_list[i],
            in_salt_bridge=(i in salt_bridge_set),
            in_disulfide=(i in disulfide_set),
            packing_score=float(res_packing[i]),
            sap_val=float(res_sap_r5[i]),
        )
        res_ala_score[i] = score_ala
        if run_alanine_scan:
            ala_scan_items.append(
                VariantScore(
                    mutation=f"{res.chain_id}:{res.aa1}{res.seq_tag}A",
                    chain_id=res.chain_id,
                    auth_seq_id=res.auth_seq_id,
                    ins_code=res.ins_code,
                    wildtype=res.aa1,
                    mutant="A",
                    score_value=score_ala,
                    score_unit="heuristic_au",
                    is_calibrated_free_energy=False,
                    ddg_kcal_mol=None,
                    backend="biophysical_heuristic_ala_scan",
                    classification=_classify_variant(score_ala),
                    sasa_burial=round(max(0.0, 1.0 - min(1.0, float(res_rsasa[i]))), 4),
                    rationale=rationale,
                )
            )

    default_chain = structure.chain_ids[0] if structure.chain_ids else "A"
    parsed_variants: List[Tuple[str, str, int, str, str]] = []
    res_lookup: Dict[Tuple[str, int, str], ResidueData] = {r.key: r for r in structure.residues}

    for v_str in variants or []:
        ch, wt, pos, ins, mut = parse_mutation_spec(v_str, default_chain=default_chain)
        res_obj = res_lookup.get((ch, pos, ins))
        if res_obj is None:
            raise ValueError(f"Variant '{v_str}' targets missing residue {ch}:{pos}{ins}.")
        if res_obj.aa1 != wt:
            raise ValueError(
                f"Variant '{v_str}' specifies wildtype '{wt}' at {ch}:{pos}{ins}, but structure has '{res_obj.aa1}'."
            )
        parsed_variants.append((ch, wt, pos, ins, mut))

    backend_used = "biophysical"
    score_value = round(heuristic_total, 2)
    score_per_res = round(heuristic_per_res, 4)
    score_unit: Literal["kcal/mol", "heuristic_au", "not_applicable"] = "heuristic_au"
    is_calibrated = False
    delta_g_fold_kcal_mol: Optional[float] = None
    external_scores: Dict[str, float] = {}

    if backend_preference == "foldx" or (backend_preference == "auto" and "foldx" in available_backends):
        fx_res = _run_foldx_ddg(structure, parsed_variants, foldx_binary)
        if fx_res is not None:
            dg_fx, external_scores = fx_res
            backend_used = "foldx"
            score_value = round(dg_fx, 2)
            score_per_res = round(dg_fx / max(n_res, 1), 4)
            score_unit = "kcal/mol"
            is_calibrated = True
            delta_g_fold_kcal_mol = score_value
        elif backend_preference == "foldx":
            raise RuntimeError("FoldX backend was explicitly requested but foldx binary is unavailable or failed.")

    if backend_used == "biophysical" and (
        backend_preference == "esm1v" or (backend_preference == "auto" and "esm1v" in available_backends)
    ):
        esm_res = _run_esm1v_zero_shot(structure, parsed_variants, esm_model_name)
        if esm_res is not None:
            external_scores = esm_res
            backend_used = "esm1v"
        elif backend_preference == "esm1v":
            raise RuntimeError(
                f"ESM-1v backend was explicitly requested but local weights for '{esm_model_name}' could not be loaded."
            )

    variant_scores: List[VariantScore] = []
    for ch, wt, pos, ins, mut in parsed_variants:
        res_obj = res_lookup[(ch, pos, ins)]
        idx = res_obj.index
        mut_key = f"{ch}:{wt}{pos}{ins}{mut}"

        if backend_used == "foldx" and mut_key in external_scores:
            val = round(float(external_scores[mut_key]), 3)
            v_unit: Literal["kcal/mol", "log_likelihood_ratio", "heuristic_au"] = "kcal/mol"
            v_cal = True
            v_ddg: Optional[float] = val
            rationale = "FoldX BuildModel all-atom rotamer-repacked free energy difference."
        elif backend_used == "esm1v" and mut_key in external_scores:
            val = round(float(external_scores[mut_key]), 3)
            v_unit = "log_likelihood_ratio"
            v_cal = False
            v_ddg = None
            rationale = "ESM-1v masked marginal evolutionary log-likelihood ratio log P(wt) - log P(mut)."
        else:
            val, rationale = _score_point_mutation_heuristic(
                res=res_obj,
                mutant_aa1=mut,
                rsasa=float(res_rsasa[idx]),
                ss=ss_list[idx],
                in_salt_bridge=(idx in salt_bridge_set),
                in_disulfide=(idx in disulfide_set),
                packing_score=float(res_packing[idx]),
                sap_val=float(res_sap_r5[idx]),
            )
            v_unit = "heuristic_au"
            v_cal = False
            v_ddg = None

        variant_scores.append(
            VariantScore(
                mutation=mut_key,
                chain_id=ch,
                auth_seq_id=pos,
                ins_code=ins,
                wildtype=wt,
                mutant=mut,
                score_value=val,
                score_unit=v_unit,
                is_calibrated_free_energy=v_cal,
                ddg_kcal_mol=v_ddg,
                backend=backend_used,
                classification=_classify_variant(val),
                sasa_burial=round(max(0.0, 1.0 - min(1.0, float(res_rsasa[idx]))), 4),
                rationale=rationale,
            )
        )

    stable_core_indices: List[int] = []
    destabilized_indices: List[int] = []
    for i, res in enumerate(structure.residues):
        is_buried = res_rsasa[i] <= rsasa_core_threshold
        is_stabilizing = res_stab[i] <= -0.35 or (is_buried and res_ala_score[i] >= 0.55)
        if is_buried and is_stabilizing and res_clashes[i] == 0 and res_buns[i] == 0 and not rama_outliers[i]:
            stable_core_indices.append(i)
        elif res_stab[i] >= 0.85 or res_clashes[i] > 0 or res_buns[i] > 0 or rama_outliers[i]:
            destabilized_indices.append(i)

    metrics = StabilityMetrics(
        calculable=True,
        backend_used=backend_used,
        available_backends=available_backends,
        score_value=score_value,
        score_per_residue=score_per_res,
        score_unit=score_unit,
        is_calibrated_free_energy=is_calibrated,
        delta_g_fold_kcal_mol=delta_g_fold_kcal_mol,
        mean_plddt=mean_plddt,
        low_plddt_fraction=low_plddt_fraction,
        packing_score=round(float(np.mean(res_packing)) if n_res else 0.0, 4),
        hbond_satisfaction_rate=round(hbond_sat_rate, 4),
        steric_clash_count=total_clash_pairs,
        molprobity_heavy_clashscore=clashscore,
        buns_count=len(buns_atoms),
        buns_atoms=buns_atoms,
        ramachandran_outlier_count=int(np.sum(rama_outliers)),
        disulfide_bond_count=len(disulfide_pairs),
        salt_bridge_count=len(salt_bridge_pairs),
        stable_core_residues=[structure.residues[i].short_label for i in stable_core_indices],
        destabilized_residues=[structure.residues[i].short_label for i in destabilized_indices],
        alanine_scan=ala_scan_items,
        variant_scores=variant_scores,
    )

    return StabilityResult(
        secondary_structures=ss_list,
        res_stability_score=res_stab,
        res_ala_score=res_ala_score,
        res_clashes=res_clashes,
        res_buns=res_buns,
        res_packing=res_packing,
        stable_core_indices=stable_core_indices,
        destabilized_indices=destabilized_indices,
        metrics=metrics,
    )
