"""Chennamsetty (2009) SAP, Coventry (2022) SapScoreMetric, and OPIG PSH SASA engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from .constants import (
    ALA_X_ALA_SIDECHAIN_MAX_SASA,
    BLACK_MOULD_HYDROPHOBICITY,
    TIEN_MAX_TOTAL_SASA,
)
from .io_parser import StructureData, compute_intrinsic_molecular_frame
from .schema import HydrophobicPatch, SAPMetrics


@dataclass
class SAPResult:
    """Internal per-atom and per-residue SASA / SAP calculation outputs."""

    atom_sasa: np.ndarray
    res_sasa_total: np.ndarray
    res_sasa_sidechain: np.ndarray
    res_rsasa: np.ndarray
    res_sap_r5: np.ndarray
    res_sap_r10: np.ndarray
    metrics: SAPMetrics
    hotspot_residue_indices: List[int]


def _fibonacci_sphere(n_points: int) -> np.ndarray:
    """Generate quasi-uniform unit sphere points via the golden-angle spiral."""
    indices = np.arange(n_points, dtype=np.float64) + 0.5
    phi = np.arccos(1.0 - 2.0 * indices / n_points)
    theta = np.pi * (1.0 + np.sqrt(5.0)) * indices
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return np.column_stack((x, y, z))


def compute_shrake_rupley_sasa(
    coords: np.ndarray,
    vdw_radii: np.ndarray,
    probe_radius: float = 1.4,
    n_sphere_points: int = 590,
    molecular_frame: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute per-atom Solvent Accessible Surface Area (A^2) via Shrake-Rupley."""
    n_atoms = len(coords)
    if n_atoms == 0:
        return np.zeros(0, dtype=np.float64)

    unit_pts = _fibonacci_sphere(n_sphere_points)
    if molecular_frame is not None:
        unit_pts = unit_pts @ molecular_frame.T
    eff_radii = vdw_radii + probe_radius
    eff_r2 = eff_radii ** 2
    max_eff_r = float(np.max(eff_radii))

    tree = cKDTree(coords)
    atom_sasa = np.zeros(n_atoms, dtype=np.float64)

    for i in range(n_atoms):
        r_i = eff_radii[i]
        nbr_indices = tree.query_ball_point(coords[i], r=r_i + max_eff_r)
        nbrs = [j for j in nbr_indices if j != i]
        if not nbrs:
            atom_sasa[i] = 4.0 * np.pi * (r_i ** 2)
            continue

        nbr_arr = np.asarray(nbrs, dtype=np.int32)
        test_pts = coords[i] + r_i * unit_pts
        diff = test_pts[:, None, :] - coords[nbr_arr][None, :, :]
        dist2 = np.einsum("ijk,ijk->ij", diff, diff)
        occluded = np.any(dist2 < eff_r2[nbr_arr][None, :], axis=1)
        frac_accessible = 1.0 - float(np.mean(occluded))
        atom_sasa[i] = 4.0 * np.pi * (r_i ** 2) * frac_accessible

    return atom_sasa


def _compute_chennamsetty_sap(
    structure: StructureData,
    atom_sasa: np.ndarray,
    radius: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute exact Chennamsetty et al. (PNAS 2009) atomic and residue-averaged SAP at radius R."""
    n_res = len(structure.residues)
    n_atoms = len(structure.atoms)
    coords = structure.coords
    tree = cKDTree(coords)

    res_hbm = np.array(
        [BLACK_MOULD_HYDROPHOBICITY.get(r.resname, 0.0) for r in structure.residues],
        dtype=np.float64,
    )
    res_max_sc = np.array(
        [ALA_X_ALA_SIDECHAIN_MAX_SASA.get(r.resname, 0.0) for r in structure.residues],
        dtype=np.float64,
    )

    sc_mask = structure.atom_is_sidechain
    atom_sap = np.zeros(n_atoms, dtype=np.float64)
    all_nbrs = tree.query_ball_point(coords, r=radius)

    for a_idx in range(n_atoms):
        nbrs = all_nbrs[a_idx]
        if not nbrs:
            continue
        nbr_arr = np.asarray(nbrs, dtype=np.int32)
        nbr_arr = nbr_arr[sc_mask[nbr_arr]]
        if len(nbr_arr) == 0:
            continue

        nbr_res = structure.atom_residue_idx[nbr_arr]
        nbr_sasa = atom_sasa[nbr_arr]
        unique_res = np.unique(nbr_res)
        sap_val = 0.0
        for r_idx in unique_res:
            denom = res_max_sc[r_idx]
            if denom <= 0.0:
                continue
            sasa_in_sphere = float(np.sum(nbr_sasa[nbr_res == r_idx]))
            frac_exposed = min(1.0, sasa_in_sphere / denom)
            sap_val += frac_exposed * res_hbm[r_idx]
        atom_sap[a_idx] = sap_val

    # Chennamsetty et al. (2009): residue SAP is averaged across the atoms of residue k
    res_sap = np.zeros(n_res, dtype=np.float64)
    for res in structure.residues:
        if res.atom_indices:
            res_sap[res.index] = float(np.mean(atom_sap[res.atom_indices]))
    return atom_sap, res_sap


def _min_sidechain_atom_distance(
    coords: np.ndarray,
    indices_a: List[int],
    indices_b: List[int],
) -> float:
    if not indices_a or not indices_b:
        return float("inf")
    pts_a = coords[indices_a]
    pts_b = coords[indices_b]
    diff = pts_a[:, None, :] - pts_b[None, :, :]
    dist2 = np.einsum("ijk,ijk->ij", diff, diff)
    return float(np.sqrt(np.min(dist2)))


def compute_sap_metrics(
    structure: StructureData,
    probe_radius: float = 1.4,
    n_sphere_points: int = 590,
    sap_radius: float = 5.0,
    sap_macro_radius: float = 10.0,
    hotspot_threshold: float = 0.15,
) -> SAPResult:
    """Compute Shrake-Rupley SASA, Chennamsetty SAP, Coventry SapScoreMetric, and OPIG PSH."""
    n_res = len(structure.residues)
    mol_frame = compute_intrinsic_molecular_frame(structure.coords, structure.center_of_mass)
    atom_sasa = compute_shrake_rupley_sasa(
        structure.coords,
        structure.vdw_radii,
        probe_radius=probe_radius,
        n_sphere_points=n_sphere_points,
        molecular_frame=mol_frame,
    )

    res_sasa_total = np.zeros(n_res, dtype=np.float64)
    res_sasa_sidechain = np.zeros(n_res, dtype=np.float64)
    res_rsasa = np.zeros(n_res, dtype=np.float64)

    for res in structure.residues:
        tot = float(np.sum(atom_sasa[res.atom_indices])) if res.atom_indices else 0.0
        sc = float(np.sum(atom_sasa[res.sidechain_atom_indices])) if res.sidechain_atom_indices else 0.0
        res_sasa_total[res.index] = tot
        res_sasa_sidechain[res.index] = sc

        if res.resname == "GLY":
            res_rsasa[res.index] = min(1.0, tot / TIEN_MAX_TOTAL_SASA["GLY"])
        else:
            denom = ALA_X_ALA_SIDECHAIN_MAX_SASA.get(res.resname, 140.0)
            res_rsasa[res.index] = min(1.0, sc / denom) if denom > 0.0 else 0.0

    # Strict backbone-only refusal per Brian Coventry / Baker Lab filtering principles
    if structure.is_backbone_only:
        zero_sap = np.zeros(n_res, dtype=np.float64)
        metrics = SAPMetrics(
            calculable=False,
            status_reason=(
                f"INVALID_BACKBONE_ONLY: sidechain completeness is {structure.sidechain_completeness:.1%} "
                f"(<80% threshold). SAP, sidechain SASA, and hydrophobic patches require all-atom sidechain "
                f"rotamers (run ProteinMPNN + FastRelax/AlphaFold2 before developability filtering)."
            ),
            radius_angstrom=sap_radius,
            sap_score=None,
            sap_score_by_chain={},
            max_sap=None,
            mean_positive_sap=None,
            sap_95th_percentile=None,
            psh_surface_hydrophobicity=None,
            total_sasa_a2=round(float(np.sum(res_sasa_total)), 2),
            hydrophobic_sidechain_sasa_a2=0.0,
            hydrophobic_surface_fraction=0.0,
            aggregation_hotspot_count=0,
            aggregation_hotspot_residues=[],
            hydrophobic_patches=[],
        )
        return SAPResult(
            atom_sasa=atom_sasa,
            res_sasa_total=res_sasa_total,
            res_sasa_sidechain=res_sasa_sidechain,
            res_rsasa=res_rsasa,
            res_sap_r5=zero_sap,
            res_sap_r10=zero_sap,
            metrics=metrics,
            hotspot_residue_indices=[],
        )

    atom_sap_r5, res_sap_r5 = _compute_chennamsetty_sap(structure, atom_sasa, radius=sap_radius)
    _, res_sap_r10 = _compute_chennamsetty_sap(structure, atom_sasa, radius=sap_macro_radius)

    total_sasa_a2 = float(np.sum(res_sasa_total))
    hydrophobic_sc_sasa_a2 = float(
        sum(
            res_sasa_sidechain[r.index]
            for r in structure.residues
            if BLACK_MOULD_HYDROPHOBICITY.get(r.resname, 0.0) > 0.0
        )
    )
    hydrophobic_surface_fraction = hydrophobic_sc_sasa_a2 / total_sasa_a2 if total_sasa_a2 > 1e-6 else 0.0

    # Brian Coventry Rosetta SapScoreMetric: whole-protein sum of positive heavy-atom SAP values
    pos_atom_sap = np.maximum(0.0, atom_sap_r5)
    sap_score = float(np.sum(pos_atom_sap))
    sap_score_by_chain: Dict[str, float] = {}
    for ch in structure.chain_ids:
        ch_sum = sum(
            float(pos_atom_sap[a_idx])
            for r in structure.residues
            if r.chain_id == ch
            for a_idx in r.atom_indices
        )
        sap_score_by_chain[ch] = round(ch_sum, 3)

    # OPIG TAP-style PSH: pairwise hydrophobic surface proximity over exposed residues (rSASA >= 0.15)
    # Using Black & Mould shifted to [0, 1] (h_bm + 0.5) * rSASA or positive hydrophobic surface pairs
    exposed_idx = [r.index for r in structure.residues if res_rsasa[r.index] >= 0.15 and r.resname != "GLY"]
    psh_val = 0.0
    if len(exposed_idx) >= 2:
        exp_pts = np.vstack([structure.residues[idx].sc_centroid for idx in exposed_idx])
        exp_h = np.array(
            [(BLACK_MOULD_HYDROPHOBICITY[structure.residues[idx].resname] + 0.5) * (0.5 + 0.5 * res_rsasa[idx]) for idx in exposed_idx],
            dtype=np.float64,
        )
        for a in range(len(exposed_idx)):
            dists = np.linalg.norm(exp_pts[a + 1 :] - exp_pts[a], axis=1)
            mask = dists > 1.0
            if np.any(mask):
                psh_val += float(np.sum((exp_h[a] * exp_h[a + 1 :][mask]) / (dists[mask] ** 2)))

    # Exposed aggregation hot-spots (SAP_r5 >= threshold and exposed hydrophobic side chain >= 10 A^2)
    hotspot_indices: List[int] = []
    for res in structure.residues:
        h_bm = BLACK_MOULD_HYDROPHOBICITY.get(res.resname, 0.0)
        if res_sap_r5[res.index] >= hotspot_threshold and res_sasa_sidechain[res.index] >= 10.0 and h_bm > 0.0:
            hotspot_indices.append(res.index)

    # Contiguous 3D surface hydrophobic patch segmentation via heavy-atom side-chain contact distance <= 5.0 A
    patch_candidates = [
        res.index
        for res in structure.residues
        if (
            res.index in hotspot_indices
            or (
                res_sap_r5[res.index] >= max(0.08, hotspot_threshold * 0.65)
                and res_sasa_sidechain[res.index] >= 10.0
                and BLACK_MOULD_HYDROPHOBICITY.get(res.resname, 0.0) > 0.0
            )
        )
    ]

    patches: List[HydrophobicPatch] = []
    if patch_candidates:
        m = len(patch_candidates)
        adj = np.zeros((m, m), dtype=np.int8)
        for a in range(m):
            ra = structure.residues[patch_candidates[a]]
            for b in range(a + 1, m):
                rb = structure.residues[patch_candidates[b]]
                if _min_sidechain_atom_distance(structure.coords, ra.sidechain_atom_indices, rb.sidechain_atom_indices) <= 5.0:
                    adj[a, b] = 1
                    adj[b, a] = 1

        n_comp, labels = connected_components(csr_matrix(adj), directed=False)
        raw_patches: List[Tuple[float, List[int]]] = []
        for comp_id in range(n_comp):
            members = [patch_candidates[k] for k in range(m) if labels[k] == comp_id]
            hydro_sc_area = float(sum(res_sasa_sidechain[idx] for idx in members))
            if len(members) >= 2 or (len(members) == 1 and res_sap_r5[members[0]] >= hotspot_threshold and hydro_sc_area >= 35.0):
                raw_patches.append((hydro_sc_area, members))

        raw_patches.sort(key=lambda item: item[0], reverse=True)
        for p_num, (hydro_sc_area, members) in enumerate(raw_patches, start=1):
            members_sorted = sorted(members)
            res_objs = [structure.residues[idx] for idx in members_sorted]
            patch_centroid = np.mean([r.sc_centroid for r in res_objs], axis=0)
            patches.append(
                HydrophobicPatch(
                    patch_id=f"HP-{p_num}",
                    chain_id=res_objs[0].chain_id,
                    residue_labels=[r.short_label for r in res_objs],
                    hydrophobic_sidechain_area_a2=round(hydro_sc_area, 2),
                    mean_sap=round(float(np.mean(res_sap_r5[members_sorted])), 4),
                    max_sap=round(float(np.max(res_sap_r5[members_sorted])), 4),
                    centroid=(
                        round(float(patch_centroid[0]), 3),
                        round(float(patch_centroid[1]), 3),
                        round(float(patch_centroid[2]), 3),
                    ),
                )
            )

    metrics = SAPMetrics(
        calculable=True,
        radius_angstrom=sap_radius,
        sap_score=round(sap_score, 3),
        sap_score_by_chain=sap_score_by_chain,
        max_sap=round(float(np.max(res_sap_r5)) if n_res else 0.0, 4),
        mean_positive_sap=round(float(np.mean(pos_atom_sap)) if len(pos_atom_sap) else 0.0, 4),
        sap_95th_percentile=round(float(np.percentile(res_sap_r5, 95)) if n_res else 0.0, 4),
        psh_surface_hydrophobicity=round(psh_val, 2),
        total_sasa_a2=round(total_sasa_a2, 2),
        hydrophobic_sidechain_sasa_a2=round(hydrophobic_sc_sasa_a2, 2),
        hydrophobic_surface_fraction=round(hydrophobic_surface_fraction, 4),
        aggregation_hotspot_count=len(hotspot_indices),
        aggregation_hotspot_residues=[structure.residues[idx].short_label for idx in hotspot_indices],
        hydrophobic_patches=patches,
    )

    return SAPResult(
        atom_sasa=atom_sasa,
        res_sasa_total=res_sasa_total,
        res_sasa_sidechain=res_sasa_sidechain,
        res_rsasa=res_rsasa,
        res_sap_r5=res_sap_r5,
        res_sap_r10=res_sap_r10,
        metrics=metrics,
        hotspot_residue_indices=hotspot_indices,
    )
