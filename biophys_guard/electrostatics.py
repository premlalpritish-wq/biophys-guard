"""Classical electrodynamics, coupled PROPKA3-style titration, Agrawal SCM, and OPIG TAP charge metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from scipy.spatial import cKDTree

from .constants import (
    C_TERM_PKA,
    DEBYE_PER_E_ANGSTROM,
    FORMAL_CHARGE_ATOMS_BY_RES,
    IONIZABLE_PKA,
    N_TERM_PKA,
    SIDECHAIN_HBOND_ACCEPTORS,
    SIDECHAIN_HBOND_DONORS,
)
from .io_parser import ResidueData, StructureData, compute_intrinsic_molecular_frame
from .schema import ChargeMetrics


@dataclass
class IonizableGroup:
    """Discrete titratable chemical group placed at its functional-group atom(s)."""

    group_id: str
    res_index: int
    chain_id: str
    kind: str  # "acid" or "base"
    group_role: str  # "sidechain", "N_TERM", "C_TERM"
    base_pka: float
    center: np.ndarray
    atom_indices: List[int]
    desolv_shift: float = 0.0
    hb_shift: float = 0.0
    effective_pka: float = 7.0
    charge: float = 0.0


@dataclass
class ChargeResult:
    """Internal per-residue and global electrostatic outputs."""

    res_charges: np.ndarray
    res_effective_pka: List[Optional[float]]
    res_scm_r10: np.ndarray
    disulfide_pairs: List[Tuple[int, int]]
    salt_bridge_pairs: List[Tuple[int, int]]
    groups: List[IonizableGroup]
    metrics: ChargeMetrics


def _fibonacci_directions(n_dirs: int = 92) -> np.ndarray:
    """Uniform unit vectors on S^2 for rotation-invariant hemispheric charge search."""
    idx = np.arange(n_dirs, dtype=np.float64) + 0.5
    phi = np.arccos(1.0 - 2.0 * idx / n_dirs)
    theta = np.pi * (1.0 + np.sqrt(5.0)) * idx
    return np.column_stack((np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)))


def detect_disulfide_pairs(structure: StructureData) -> List[Tuple[int, int]]:
    """Return pairs of disulfide-bonded CYS residue indices (SG-SG <= 2.35 A)."""
    cys_sg: List[Tuple[int, np.ndarray]] = []
    for res in structure.residues:
        if res.resname != "CYS":
            continue
        for atom in res.atoms:
            if atom.name == "SG":
                cys_sg.append((res.index, atom.coord))
                break

    pairs: List[Tuple[int, int]] = []
    for i in range(len(cys_sg)):
        for j in range(i + 1, len(cys_sg)):
            idx_a, coord_a = cys_sg[i]
            idx_b, coord_b = cys_sg[j]
            if float(np.linalg.norm(coord_a - coord_b)) <= 2.35:
                pairs.append((idx_a, idx_b))
    return pairs


def _terminal_residue_sets(structure: StructureData) -> Tuple[Set[int], Set[int]]:
    n_terms: Set[int] = set()
    c_terms: Set[int] = set()
    by_chain: Dict[str, List[ResidueData]] = {}
    for res in structure.residues:
        by_chain.setdefault(res.chain_id, []).append(res)
    for chain_res in by_chain.values():
        if chain_res:
            n_terms.add(chain_res[0].index)
            c_terms.add(chain_res[-1].index)
    return n_terms, c_terms


def _build_ionizable_groups(
    structure: StructureData,
    disulfide_cys: Set[int],
    n_terms: Set[int],
    c_terms: Set[int],
) -> List[IonizableGroup]:
    """Build all titratable groups positioned strictly at chemical functional-group atoms."""
    groups: List[IonizableGroup] = []

    for res in structure.residues:
        atom_idx_by_name = {structure.atoms[idx].name: idx for idx in res.atom_indices}

        # Sidechain titratable group
        if res.index not in disulfide_cys and res.resname in IONIZABLE_PKA:
            base_pka, kind = IONIZABLE_PKA[res.resname]
            target_names = FORMAL_CHARGE_ATOMS_BY_RES.get(res.resname, ())
            matched_idx = [atom_idx_by_name[nm] for nm in target_names if nm in atom_idx_by_name]
            if not matched_idx:
                # Fallback to terminal sidechain atom or CB if minor truncation exists
                matched_idx = res.sidechain_atom_indices[-1:] if res.sidechain_atom_indices else res.atom_indices[:1]
            center = np.mean(structure.coords[matched_idx], axis=0)
            groups.append(
                IonizableGroup(
                    group_id=f"{res.short_label}:SC",
                    res_index=res.index,
                    chain_id=res.chain_id,
                    kind=kind,
                    group_role="sidechain",
                    base_pka=base_pka,
                    center=center,
                    atom_indices=matched_idx,
                )
            )

        # N-terminal alpha-amino group (distinct from sidechain)
        if res.index in n_terms and "N" in atom_idx_by_name:
            n_idx = atom_idx_by_name["N"]
            groups.append(
                IonizableGroup(
                    group_id=f"{res.short_label}:N_TERM",
                    res_index=res.index,
                    chain_id=res.chain_id,
                    kind="base",
                    group_role="N_TERM",
                    base_pka=N_TERM_PKA,
                    center=structure.coords[n_idx].copy(),
                    atom_indices=[n_idx],
                )
            )

        # C-terminal carboxylate group (distinct from sidechain)
        if res.index in c_terms:
            c_idx_list = [atom_idx_by_name[nm] for nm in ("O", "OXT") if nm in atom_idx_by_name]
            if not c_idx_list and "C" in atom_idx_by_name:
                c_idx_list = [atom_idx_by_name["C"]]
            if c_idx_list:
                groups.append(
                    IonizableGroup(
                        group_id=f"{res.short_label}:C_TERM",
                        res_index=res.index,
                        chain_id=res.chain_id,
                        kind="acid",
                        group_role="C_TERM",
                        base_pka=C_TERM_PKA,
                        center=np.mean(structure.coords[c_idx_list], axis=0),
                        atom_indices=c_idx_list,
                    )
                )

    return groups


def _compute_static_pka_perturbations(
    structure: StructureData,
    groups: List[IonizableGroup],
) -> None:
    """Compute PROPKA3-style smooth neighbor-density desolvation and polar H-bond pKa shifts."""
    coords = structure.coords
    tree = cKDTree(coords)

    # Identify all donor and acceptor heavy atom indices for H-bond stabilization term
    donor_atoms: Set[int] = set()
    acceptor_atoms: Set[int] = set()
    for res in structure.residues:
        sc_don = SIDECHAIN_HBOND_DONORS.get(res.resname, frozenset())
        sc_acc = SIDECHAIN_HBOND_ACCEPTORS.get(res.resname, frozenset())
        for idx in res.atom_indices:
            aname = structure.atoms[idx].name
            if aname == "N" and res.resname != "PRO":
                donor_atoms.add(idx)
            elif aname in sc_don:
                donor_atoms.add(idx)
            if aname in {"O", "OXT"} or aname in sc_acc:
                acceptor_atoms.add(idx)

    for g in groups:
        # 1. Desolvation from smooth local heavy-atom neighbor density (6.0 .. 15.5 A)
        nbrs = tree.query_ball_point(g.center, r=15.5)
        n_local = 0.0
        for a_idx in nbrs:
            if structure.atom_residue_idx[a_idx] == g.res_index:
                continue
            d = float(np.linalg.norm(coords[a_idx] - g.center))
            if d <= 6.0:
                n_local += 1.0
            elif d < 15.5:
                n_local += (15.5 - d) / 9.5

        # Reference interior neighbor count (~50 heavy atoms in 6-15.5 A shell)
        burial_frac = min(1.0, max(0.0, (n_local - 10.0) / 42.0))
        shift_desolv = 2.4 * (burial_frac ** 1.4)
        g.desolv_shift = +shift_desolv if g.kind == "acid" else -shift_desolv

        # 2. Hydrogen-bond stabilization of the charged state within 3.5 A
        hb_partners: Set[int] = set()
        for g_atom in g.atom_indices:
            near = tree.query_ball_point(coords[g_atom], r=3.5)
            for p_idx in near:
                if structure.atom_residue_idx[p_idx] == g.res_index:
                    continue
                if g.kind == "acid" and p_idx in donor_atoms:
                    hb_partners.add(int(structure.atom_residue_idx[p_idx]))
                elif g.kind == "base" and p_idx in acceptor_atoms:
                    hb_partners.add(int(structure.atom_residue_idx[p_idx]))

        n_hb = min(3, len(hb_partners))
        g.hb_shift = (-0.60 * n_hb) if g.kind == "acid" else (+0.60 * n_hb)


def _henderson_hasselbalch_charge(ph: float, pka: float, kind: str) -> float:
    if kind == "acid":
        return -1.0 / (1.0 + 10.0 ** (pka - ph))
    return +1.0 / (1.0 + 10.0 ** (ph - pka))


def _solve_coupled_charges_at_ph(
    groups: List[IonizableGroup],
    ph: float,
    n_iters: int = 12,
) -> np.ndarray:
    """Self-consistently solve coupled PROPKA3-style pKa and Henderson-Hasselbalch charges."""
    n_g = len(groups)
    if n_g == 0:
        return np.zeros(0, dtype=np.float64)

    centers = np.vstack([g.center for g in groups])
    dist_mat = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)

    pka = np.array([g.base_pka + g.desolv_shift + g.hb_shift for g in groups], dtype=np.float64)
    q = np.array(
        [_henderson_hasselbalch_charge(ph, pka[i], groups[i].kind) for i in range(n_g)],
        dtype=np.float64,
    )

    # Iterative damped Coulombic pKa coupling within 12.0 A
    # Unified sign law: d_pKa_coul(g) = - sum_{h != g} q_h * K / (d^2 + 4.0) * (1 - d / 12.0)
    k_coul = 30.0
    for _ in range(n_iters):
        new_pka = np.empty(n_g, dtype=np.float64)
        for i in range(n_g):
            coul_shift = 0.0
            for j in range(n_g):
                if i == j or groups[i].res_index == groups[j].res_index:
                    continue
                d = float(dist_mat[i, j])
                if d <= 12.0:
                    coul_shift -= q[j] * (k_coul / (d * d + 4.0)) * (1.0 - d / 12.0)
            # Clamp extreme local electrostatic shifts physically to [-4.5, +4.5]
            coul_shift = max(-4.5, min(4.5, coul_shift))
            new_pka[i] = groups[i].base_pka + groups[i].desolv_shift + groups[i].hb_shift + coul_shift

        new_q = np.array(
            [_henderson_hasselbalch_charge(ph, new_pka[i], groups[i].kind) for i in range(n_g)],
            dtype=np.float64,
        )
        max_delta = float(np.max(np.abs(new_q - q)))
        q = 0.6 * new_q + 0.4 * q
        pka = new_pka
        if max_delta < 1e-4:
            break

    for i, g in enumerate(groups):
        g.effective_pka = float(pka[i])
        g.charge = float(q[i])
    return q


def _solve_isoelectric_point(groups: List[IonizableGroup]) -> float:
    """Solve net_charge(pH) = 0 on [1.0, 14.0] via bisection using coupled titration."""
    if not groups:
        return 7.0

    def net_q(ph_val: float) -> float:
        return float(np.sum(_solve_coupled_charges_at_ph(groups, ph_val, n_iters=8)))

    lo, hi = 1.0, 14.0
    if net_q(lo) <= 0.0:
        return lo
    if net_q(hi) >= 0.0:
        return hi
    for _ in range(28):
        mid = 0.5 * (lo + hi)
        if net_q(mid) > 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _detect_salt_bridges(
    groups: List[IonizableGroup],
    cutoff: float = 4.5,
) -> List[Tuple[int, int]]:
    """Detect oppositely charged functional-group pairs (including termini) within 4.5 A."""
    pairs: Set[Tuple[int, int]] = set()
    for i in range(len(groups)):
        ga = groups[i]
        if abs(ga.charge) < 0.25:
            continue
        for j in range(i + 1, len(groups)):
            gb = groups[j]
            if ga.res_index == gb.res_index or abs(gb.charge) < 0.25:
                continue
            if ga.charge * gb.charge < 0.0:
                if float(np.linalg.norm(ga.center - gb.center)) <= cutoff:
                    pairs.add((min(ga.res_index, gb.res_index), max(ga.res_index, gb.res_index)))
    return sorted(pairs)


def compute_charge_metrics(
    structure: StructureData,
    res_sasa_sidechain: np.ndarray,
    res_rsasa: np.ndarray,
    total_sasa_a2: float,
    ph: float = 7.4,
    scm_radius: float = 10.0,
) -> ChargeResult:
    """Compute coupled PROPKA3-style titration, CoM dipole, Agrawal SCM, and OPIG TAP charge metrics."""
    n_res = len(structure.residues)
    disulfide_pairs = detect_disulfide_pairs(structure)
    com = structure.center_of_mass

    if structure.is_backbone_only:
        zero_res = np.zeros(n_res, dtype=np.float64)
        metrics = ChargeMetrics(
            calculable=False,
            status_reason=(
                f"INVALID_BACKBONE_ONLY: sidechain completeness {structure.sidechain_completeness:.1%} (<80%). "
                f"Chemical charge centers, coupled pKa, SCM, PPC/PNC require all-atom sidechain coordinates."
            ),
            ph=ph,
            center_of_mass_angstrom=(round(float(com[0]), 3), round(float(com[1]), 3), round(float(com[2]), 3)),
            is_origin_dependent_dipole=False,
            net_charge=0.0,
            net_charge_by_chain={ch: 0.0 for ch in structure.chain_ids},
            positive_charge=0.0,
            negative_charge=0.0,
            isoelectric_point=7.0,
            charge_density_per_1000a2=0.0,
            dipole_moment_debye=0.0,
            dipole_vector_debye=(0.0, 0.0, 0.0),
            charge_symmetry_score=1.0,
            hemispheric_charge_asymmetry=0.0,
            dipole_normalized_asymmetry=0.0,
            interchain_sfvcsp=None,
            ppc_positive_patch=0.0,
            pnc_negative_patch=0.0,
            scm_pos_sum=0.0,
            scm_neg_sum=0.0,
            max_scm_positive=0.0,
            min_scm_negative=0.0,
            salt_bridge_count=0,
            ionizable_residue_count=0,
        )
        return ChargeResult(
            res_charges=zero_res,
            res_effective_pka=[None] * n_res,
            res_scm_r10=zero_res,
            disulfide_pairs=disulfide_pairs,
            salt_bridge_pairs=[],
            groups=[],
            metrics=metrics,
        )

    disulfide_cys = {idx for pair in disulfide_pairs for idx in pair}
    n_terms, c_terms = _terminal_residue_sets(structure)
    groups = _build_ionizable_groups(structure, disulfide_cys, n_terms, c_terms)
    _compute_static_pka_perturbations(structure, groups)

    # Solve pI first, then solve target pH so group.effective_pka and group.charge reflect target pH
    pi_val = _solve_isoelectric_point(groups)
    _solve_coupled_charges_at_ph(groups, ph, n_iters=12)
    salt_bridges = _detect_salt_bridges(groups, cutoff=4.5)

    res_charges = np.zeros(n_res, dtype=np.float64)
    res_effective_pka: List[Optional[float]] = [None] * n_res
    for g in groups:
        res_charges[g.res_index] += g.charge
        if g.group_role == "sidechain":
            res_effective_pka[g.res_index] = round(g.effective_pka, 2)

    net_charge = float(sum(g.charge for g in groups))
    pos_charge = float(sum(g.charge for g in groups if g.charge > 0.0))
    neg_charge = float(sum(g.charge for g in groups if g.charge < 0.0))
    charge_density = (net_charge / max(total_sasa_a2, 1.0)) * 1000.0

    net_charge_by_chain: Dict[str, float] = {ch: 0.0 for ch in structure.chain_ids}
    for g in groups:
        net_charge_by_chain[g.chain_id] = net_charge_by_chain.get(g.chain_id, 0.0) + g.charge
    net_charge_by_chain = {ch: round(val, 3) for ch, val in net_charge_by_chain.items()}

    interchain_sfvcsp: Optional[float] = None
    if len(structure.chain_ids) >= 2:
        ch_a, ch_b = structure.chain_ids[0], structure.chain_ids[1]
        interchain_sfvcsp = round(net_charge_by_chain[ch_a] * net_charge_by_chain[ch_b], 3)

    # Classical SE(3)-invariant dipole moment relative to true atomic mass-weighted CoM:
    # mu_CoM = 4.803204 * sum_g q_g * (r_g - r_CoM), with ZERO unphysical Q_net/N subtraction!
    if groups:
        g_centers = np.vstack([g.center for g in groups])
        g_charges = np.array([g.charge for g in groups], dtype=np.float64)
        rel_g = g_centers - com
        dipole_vec = DEBYE_PER_E_ANGSTROM * np.sum(g_charges[:, None] * rel_g, axis=0)
    else:
        g_centers = np.zeros((0, 3), dtype=np.float64)
        g_charges = np.zeros(0, dtype=np.float64)
        rel_g = g_centers
        dipole_vec = np.zeros(3, dtype=np.float64)

    dipole_mag = float(np.linalg.norm(dipole_vec))

    # Agrawal et al. (2016) Spatial Charge Map (SCM):
    # Strictly filter to solvent-exposed sidechain residues (SASA_sc >= 10.0 A^2)
    exposed_sc_atoms: List[int] = []
    exposed_sc_charges: List[float] = []
    exposed_sc_res: Set[int] = {
        r.index for r in structure.residues if res_sasa_sidechain[r.index] >= 10.0 and r.sidechain_atom_indices
    }

    atom_formal_charge = np.zeros(len(structure.atoms), dtype=np.float64)
    for g in groups:
        if g.group_role != "sidechain" or g.res_index not in exposed_sc_res:
            continue
        share = g.charge / max(len(g.atom_indices), 1)
        for a_idx in g.atom_indices:
            atom_formal_charge[a_idx] += share

    for r_idx in sorted(exposed_sc_res):
        for a_idx in structure.residues[r_idx].sidechain_atom_indices:
            exposed_sc_atoms.append(a_idx)
            exposed_sc_charges.append(float(atom_formal_charge[a_idx]))

    res_scm = np.zeros(n_res, dtype=np.float64)
    scm_pos_sum = 0.0
    scm_neg_sum = 0.0
    max_scm_pos = 0.0
    min_scm_neg = 0.0

    if exposed_sc_atoms:
        exp_idx_arr = np.asarray(exposed_sc_atoms, dtype=np.int32)
        exp_coords = structure.coords[exp_idx_arr]
        exp_q = np.asarray(exposed_sc_charges, dtype=np.float64)
        exp_tree = cKDTree(exp_coords)
        exp_nbrs = exp_tree.query_ball_point(exp_coords, r=scm_radius)
        atom_scm = np.zeros(len(exp_idx_arr), dtype=np.float64)
        for k, nbrs in enumerate(exp_nbrs):
            if nbrs:
                atom_scm[k] = float(np.sum(exp_q[nbrs]))

        scm_pos_sum = float(np.sum(np.maximum(0.0, atom_scm)))
        scm_neg_sum = float(np.sum(np.minimum(0.0, atom_scm)))
        max_scm_pos = float(np.max(atom_scm))
        min_scm_neg = float(np.min(atom_scm))

        exp_parent_res = structure.atom_residue_idx[exp_idx_arr]
        for r_idx in exposed_sc_res:
            vals = atom_scm[exp_parent_res == r_idx]
            if len(vals):
                res_scm[r_idx] = float(np.mean(vals))

    # Unscaled pairwise electrostatic cluster integrals sum_{1 < d <= 12 A} (|q_i * q_j| / d^2)
    # Note: Evaluated across all exposed ionizable centers (not restricted to antibody CDR loops).
    exp_pos_g = [(g.center, g.charge) for g in groups if g.charge >= 0.25 and (g.group_role != "sidechain" or res_rsasa[g.res_index] >= 0.15)]
    exp_neg_g = [(g.center, abs(g.charge)) for g in groups if g.charge <= -0.25 and (g.group_role != "sidechain" or res_rsasa[g.res_index] >= 0.15)]

    def _patch_sum(pts_q: List[Tuple[np.ndarray, float]]) -> float:
        if len(pts_q) < 2:
            return 0.0
        total = 0.0
        for a in range(len(pts_q)):
            ca, qa = pts_q[a]
            for b in range(a + 1, len(pts_q)):
                cb, qb = pts_q[b]
                d = float(np.linalg.norm(ca - cb))
                if 1.0 < d <= 12.0:
                    total += (qa * qb) / (d * d)
        return total

    ppc_val = _patch_sum(exp_pos_g)
    pnc_val = _patch_sum(exp_neg_g)

    # Smooth SO(3) rotation-invariant hemispheric charge asymmetry over 92 uniform directions
    total_abs_q = float(np.sum(np.abs(g_charges)))
    if total_abs_q < 0.5:
        hemispheric_asym = 0.0
        dipole_norm_asym = 0.0
        css = 1.0
    else:
        mol_frame = compute_intrinsic_molecular_frame(structure.coords, com)
        dirs = _fibonacci_directions(92) @ mol_frame.T
        if dipole_mag > 1e-6:
            dirs = np.vstack([dirs, dipole_vec / dipole_mag])
        proj = rel_g @ dirs.T  # (n_groups, n_dirs)
        hemi_diff = np.abs(np.sum(g_charges[:, None] * np.tanh(proj / 2.5), axis=0))
        hemispheric_asym = float(min(1.0, np.max(hemi_diff) / (total_abs_q + 1e-9)))

        rg = float(np.sqrt(np.mean(np.sum((structure.coords - com) ** 2, axis=1)))) + 1e-6
        ideal_max_dipole = DEBYE_PER_E_ANGSTROM * total_abs_q * rg
        dipole_norm_asym = float(min(1.0, dipole_mag / max(ideal_max_dipole, 1e-6)))
        css = float(np.clip(1.0 - (0.60 * hemispheric_asym + 0.40 * dipole_norm_asym), 0.0, 1.0))

    ionizable_count = sum(1 for g in groups if abs(g.charge) >= 0.25)

    metrics = ChargeMetrics(
        calculable=True,
        ph=ph,
        center_of_mass_angstrom=(round(float(com[0]), 3), round(float(com[1]), 3), round(float(com[2]), 3)),
        is_origin_dependent_dipole=(abs(net_charge) > 1e-3),
        net_charge=round(net_charge, 3),
        net_charge_by_chain=net_charge_by_chain,
        positive_charge=round(pos_charge, 3),
        negative_charge=round(neg_charge, 3),
        isoelectric_point=round(pi_val, 2),
        charge_density_per_1000a2=round(charge_density, 3),
        dipole_moment_debye=round(dipole_mag, 2),
        dipole_vector_debye=(
            round(float(dipole_vec[0]), 2),
            round(float(dipole_vec[1]), 2),
            round(float(dipole_vec[2]), 2),
        ),
        charge_symmetry_score=round(css, 4),
        hemispheric_charge_asymmetry=round(hemispheric_asym, 4),
        dipole_normalized_asymmetry=round(dipole_norm_asym, 4),
        interchain_sfvcsp=interchain_sfvcsp,
        ppc_positive_patch=round(ppc_val, 3),
        pnc_negative_patch=round(pnc_val, 3),
        scm_pos_sum=round(scm_pos_sum, 3),
        scm_neg_sum=round(scm_neg_sum, 3),
        max_scm_positive=round(max_scm_pos, 3),
        min_scm_negative=round(min_scm_neg, 3),
        salt_bridge_count=len(salt_bridges),
        ionizable_residue_count=ionizable_count,
    )

    return ChargeResult(
        res_charges=res_charges,
        res_effective_pka=res_effective_pka,
        res_scm_r10=res_scm,
        disulfide_pairs=disulfide_pairs,
        salt_bridge_pairs=salt_bridges,
        groups=groups,
        metrics=metrics,
    )
