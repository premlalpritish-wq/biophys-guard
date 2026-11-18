"""Structured Pydantic v2 models and JSON Schema exporter for biophys-guard."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field

TrafficLight = Literal["GREEN", "AMBER", "RED", "INVALID_BACKBONE_ONLY"]


class BiophysGuardConfig(BaseModel):
    """Runtime parameters and literature-calibrated reference bounds for biophys-guard."""

    ph: float = Field(default=7.4, ge=0.0, le=14.0, description="Solution pH for coupled pKa and electrostatic calculations.")
    sap_radius: float = Field(default=5.0, gt=1.0, le=20.0, description="Chennamsetty SAP spatial cutoff radius (A).")
    sap_macro_radius: float = Field(default=10.0, gt=2.0, le=25.0, description="Macroscopic SAP neighborhood cutoff radius (A).")
    scm_radius: float = Field(default=10.0, gt=2.0, le=25.0, description="Agrawal et al. (2016) Spatial Charge Map cutoff radius (A).")
    probe_radius: float = Field(default=1.4, gt=0.1, le=5.0, description="Solvent probe radius for Shrake-Rupley SASA (A).")
    n_sphere_points: int = Field(default=590, ge=128, le=1200, description="Fibonacci sphere tessellation points per atom for rotationally converged SASA.")
    sap_hotspot_threshold: float = Field(default=0.15, description="Per-residue Chennamsetty SAP threshold for flagging exposed aggregation hot-spots.")
    rsasa_core_threshold: float = Field(default=0.18, ge=0.0, le=1.0, description="Relative side-chain SASA cutoff below which a residue is considered buried core.")
    coventry_sap_green_max: float = Field(default=25.0, description="Coventry SapScoreMetric green bound (sum of positive residue SAP).")
    coventry_sap_amber_max: float = Field(default=35.0, description="Coventry SapScoreMetric amber upper bound (Cao et al., Nature 2022).")
    min_charge_symmetry_amber: float = Field(default=0.55, ge=0.0, le=1.0, description="Single-chain charge symmetry amber lower bound.")
    min_charge_symmetry_red: float = Field(default=0.35, ge=0.0, le=1.0, description="Single-chain charge symmetry red lower bound.")
    stability_backend: Literal["auto", "foldx", "esm1v", "biophysical"] = Field(
        default="auto",
        description="Variant/stability scorer backend selector.",
    )
    run_alanine_scan: bool = Field(default=True, description="Whether to compute per-residue alanine scan scores.")
    foldx_binary: str = Field(default="foldx", description="Executable name or path for FoldX if installed.")
    esm_model_name: str = Field(default="facebook/esm1v_t33_650M_UR90S_1", description="ESM-1v model identifier for zero-shot variant scoring.")


class GateCheck(BaseModel):
    """Single OPIG-style empirically anchored developability gate check."""

    name: str
    metric: Optional[float] = None
    unit: str
    reference_source: str
    green_range: str
    amber_range: str
    red_range: str
    flag: TrafficLight
    detail: str


class DevelopabilityVerdict(BaseModel):
    """Traffic-light developability gating summary anchored to OPIG/Coventry standards."""

    status: TrafficLight
    is_backbone_only: bool = False
    sidechain_completeness: float = Field(ge=0.0, le=1.0)
    green_count: int = Field(default=0, ge=0)
    amber_count: int = Field(default=0, ge=0)
    red_count: int = Field(default=0, ge=0)
    ml_confidence_flag: Optional[Literal["GREEN", "AMBER", "RED"]] = None
    low_confidence_residue_count: int = Field(default=0, ge=0)
    low_confidence_surface_residues: List[str] = Field(default_factory=list)
    gate_checks: List[GateCheck] = Field(default_factory=list)
    flags: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)


class HydrophobicPatch(BaseModel):
    """Contiguous 3D exposed hydrophobic cluster measured over side-chain contacts."""

    patch_id: str
    chain_id: str
    residue_labels: List[str]
    hydrophobic_sidechain_area_a2: float = Field(ge=0.0)
    mean_sap: float
    max_sap: float
    centroid: Tuple[float, float, float]


class SAPMetrics(BaseModel):
    """Chennamsetty (2009) SAP & Coventry (2022) SapScoreMetric outputs."""

    calculable: bool = True
    status_reason: Optional[str] = None
    radius_angstrom: float
    sap_score: Optional[float] = Field(default=None, description="Coventry whole-protein sum of positive residue SAP_r5 values.")
    sap_score_by_chain: Dict[str, float] = Field(default_factory=dict)
    max_sap: Optional[float] = None
    mean_positive_sap: Optional[float] = None
    sap_95th_percentile: Optional[float] = None
    psh_surface_hydrophobicity: Optional[float] = Field(default=None, description="OPIG-style pairwise surface hydrophobic proximity integral (PSH).")
    total_sasa_a2: float = Field(ge=0.0)
    hydrophobic_sidechain_sasa_a2: float = Field(ge=0.0)
    hydrophobic_surface_fraction: float = Field(ge=0.0, le=1.0)
    aggregation_hotspot_count: int = Field(default=0, ge=0)
    aggregation_hotspot_residues: List[str] = Field(default_factory=list)
    hydrophobic_patches: List[HydrophobicPatch] = Field(default_factory=list)


class ChargeMetrics(BaseModel):
    """Classical electrodynamics, coupled PROPKA-style titration, Agrawal SCM, and OPIG TAP charge metrics."""

    calculable: bool = True
    status_reason: Optional[str] = None
    ph: float
    pka_model: str = "PROPKA3_coupled_desolvation_HB_Coulomb"
    reference_origin: str = "atomic_mass_weighted_center_of_mass"
    center_of_mass_angstrom: Tuple[float, float, float]
    is_origin_dependent_dipole: bool = Field(
        description="True whenever abs(net_charge) > 1e-3 e; dipole is strictly standardized to the mass-weighted CoM."
    )
    net_charge: float
    net_charge_by_chain: Dict[str, float] = Field(default_factory=dict)
    positive_charge: float = Field(ge=0.0)
    negative_charge: float = Field(le=0.0)
    isoelectric_point: float = Field(ge=0.0, le=14.0)
    charge_density_per_1000a2: float
    dipole_moment_debye: float = Field(ge=0.0)
    dipole_vector_debye: Tuple[float, float, float]
    charge_symmetry_score: float = Field(ge=0.0, le=1.0)
    hemispheric_charge_asymmetry: float = Field(ge=0.0, le=1.0)
    dipole_normalized_asymmetry: float = Field(ge=0.0, le=1.0)
    interchain_sfvcsp: Optional[float] = Field(
        default=None,
        description="OPIG TAP Structural Fv Charge Symmetry Parameter (q_A * q_B) for multi-chain assemblies.",
    )
    ppc_positive_patch: float = Field(ge=0.0, description="OPIG-style positive surface charge patch score (PPC).")
    pnc_negative_patch: float = Field(ge=0.0, description="OPIG-style negative surface charge patch score (PNC).")
    scm_method_note: str = "Agrawal_2016_exposed_sidechain_SASA_ge_10A2_10A_sphere (no APBS Poisson-Boltzmann continuum integral)"
    scm_pos_sum: float = Field(ge=0.0, description="Sum of positive atomic SCM values across exposed residues.")
    scm_neg_sum: float = Field(le=0.0, description="Sum of negative atomic SCM values across exposed residues (viscosity predictor).")
    max_scm_positive: float
    min_scm_negative: float
    salt_bridge_count: int = Field(ge=0)
    ionizable_residue_count: int = Field(ge=0)


class VariantScore(BaseModel):
    """Mutational effect prediction with explicit physical unit and calibration disclosure."""

    mutation: str
    chain_id: str
    auth_seq_id: int
    ins_code: str = ""
    wildtype: str
    mutant: str
    score_value: float = Field(description="Numeric variant effect score in `score_unit` (positive = destabilizing/unfavorable).")
    score_unit: Literal["kcal/mol", "log_likelihood_ratio", "heuristic_au"]
    is_calibrated_free_energy: bool = Field(
        description="True ONLY for FoldX BuildModel with 3D repacking; False for ESM-1v LLR and static heuristics."
    )
    ddg_kcal_mol: Optional[float] = Field(
        default=None,
        description="Populated ONLY when FoldX all-atom repacking produced calibrated kcal/mol energies.",
    )
    backend: str
    classification: Literal["favorable_or_stabilizing", "neutral", "mildly_destabilizing", "strongly_destabilizing"]
    sasa_burial: float = Field(ge=0.0, le=1.0)
    rationale: str


class StabilityMetrics(BaseModel):
    """Structural quality, atomic BUNS, MolProbity clashes, and stability scoring with epistemic unit honesty."""

    calculable: bool = True
    status_reason: Optional[str] = None
    backend_used: str
    available_backends: List[str] = Field(default_factory=list)
    score_value: Optional[float] = Field(default=None, description="Global stability score in `score_unit`.")
    score_per_residue: Optional[float] = None
    score_unit: Literal["kcal/mol", "heuristic_au", "not_applicable"]
    is_calibrated_free_energy: bool = False
    delta_g_fold_kcal_mol: Optional[float] = Field(
        default=None,
        description="Populated ONLY when FoldX Stability produced calibrated kcal/mol energy.",
    )
    mean_plddt: Optional[float] = Field(default=None, ge=0.0, le=100.0)
    low_plddt_fraction: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    packing_score: float = Field(ge=0.0, le=1.0)
    hbond_satisfaction_rate: float = Field(ge=0.0, le=1.0)
    steric_clash_count: int = Field(default=0, ge=0, description="MolProbity non-bonded heavy-atom overlaps >= 0.40 A.")
    molprobity_heavy_clashscore: float = Field(default=0.0, ge=0.0, description="Heavy-atom steric clashes per 1,000 heavy atoms.")
    buns_count: int = Field(default=0, ge=0, description="Atomic Buried Unsatisfied Polar atoms (SASA_atom < 0.5 A^2, 0 valid H-bonds).")
    buns_atoms: List[str] = Field(default_factory=list)
    ramachandran_outlier_count: int = Field(default=0, ge=0)
    disulfide_bond_count: int = Field(default=0, ge=0)
    salt_bridge_count: int = Field(default=0, ge=0)
    stable_core_residues: List[str] = Field(default_factory=list)
    destabilized_residues: List[str] = Field(default_factory=list)
    alanine_scan: List[VariantScore] = Field(default_factory=list)
    variant_scores: List[VariantScore] = Field(default_factory=list)


class ResidueRiskProfile(BaseModel):
    """Per-residue biophysical metrics, confidence metadata, and 3D risk classification."""

    chain_id: str
    auth_seq_id: int
    ins_code: str = ""
    auth_comp_id: str
    aa1: str
    secondary_structure: Literal["helix", "sheet", "coil"]
    plddt: Optional[float] = None
    low_confidence_rotamer_warning: bool = False
    sasa_total: float = Field(ge=0.0)
    sasa_sidechain: float = Field(ge=0.0)
    rsasa: float = Field(ge=0.0, le=1.0)
    sap_r5: Optional[float] = None
    sap_r10: Optional[float] = None
    effective_pka: Optional[float] = None
    charge_ph: float = 0.0
    scm_r10: Optional[float] = None
    stability_score_residue: Optional[float] = None
    ala_scan_score: Optional[float] = None
    buns_atom_count: int = 0
    clash_atom_count: int = 0
    biophysical_risk_score: float = Field(ge=-99.99, le=99.99)
    risk_category: Literal[
        "aggregation_hotspot",
        "destabilized",
        "neutral_surface",
        "stable_core",
        "not_applicable_backbone_only",
    ]
    hex_color: str


class ViewerOverlaySpec(BaseModel):
    """Generated 3D overlay file paths and categorical residue lists."""

    pymol_script_path: Optional[str] = None
    risk_pdb_path: Optional[str] = None
    ngl_viewer_path: Optional[str] = None
    molstar_sidecar_path: Optional[str] = None
    color_legend: Dict[str, str] = Field(
        default_factory=lambda: {
            "aggregation_hotspot": "#dc2626",
            "destabilized": "#f97316",
            "neutral_surface": "#e2e8f0",
            "stable_core": "#2563eb",
            "not_applicable_backbone_only": "#94a3b8",
        }
    )
    red_hotspot_residues: List[str] = Field(default_factory=list)
    blue_core_residues: List[str] = Field(default_factory=list)
    destabilized_residues: List[str] = Field(default_factory=list)


class BiophysGuardReport(BaseModel):
    """Root JSON report schema for biophys-guard developability analysis."""

    schema_version: str = Field(default="2.0.0")
    tool: str = Field(default="biophys-guard")
    generated_at_utc: str
    input_file: str
    structure_format: Literal["pdb", "mmcif"]
    detected_model_source: Literal["AlphaFold", "ESMFold", "ImmuneBuilder", "RFdiffusion", "Experimental", "Unknown"]
    is_backbone_only: bool
    sidechain_completeness: float = Field(ge=0.0, le=1.0)
    chain_ids: List[str]
    residue_count: int = Field(ge=1)
    atom_count: int = Field(ge=1)
    sequence_by_chain: Dict[str, str]
    config: BiophysGuardConfig
    verdict: DevelopabilityVerdict
    sap_metrics: SAPMetrics
    charge_metrics: ChargeMetrics
    stability_metrics: StabilityMetrics
    residues: List[ResidueRiskProfile]
    viewer_overlays: ViewerOverlaySpec


def export_json_schema(output_path: str | Path) -> Path:
    """Export the BiophysGuardReport Draft 2020-12 JSON Schema to disk."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    schema = BiophysGuardReport.model_json_schema()
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["$id"] = "https://biophys-guard.local/schemas/biophys_guard_report.schema.json"
    path.write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")
    return path
