"""End-to-end biophys-guard orchestration with OPIG/Coventry traffic-light gating."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import List, Literal, Optional

from .constants import (
    MOLPROBITY_CLASH_AMBER_MAX,
    MOLPROBITY_CLASH_GREEN_MAX,
    OPIG_TAP_SFVCSP_AMBER_MIN,
    OPIG_TAP_SFVCSP_GREEN_MIN,
)
from .electrostatics import compute_charge_metrics
from .io_parser import StructureData, load_structure_file
from .sap import SAPResult, compute_sap_metrics
from .schema import (
    BiophysGuardConfig,
    BiophysGuardReport,
    ChargeMetrics,
    DevelopabilityVerdict,
    GateCheck,
    TrafficLight,
    ViewerOverlaySpec,
)
from .stability import StabilityResult, compute_stability_metrics
from .visualization import classify_residues_and_scores, generate_all_overlays


def _build_verdict(
    structure: StructureData,
    sap_res: SAPResult,
    charge_metrics: ChargeMetrics,
    stab_res: StabilityResult,
    config: BiophysGuardConfig,
) -> DevelopabilityVerdict:
    """Evaluate OPIG/Coventry empirical Green/Amber/Red developability gates."""
    checks: List[GateCheck] = []
    flags: List[str] = []
    recommendations: List[str] = []

    if structure.is_backbone_only:
        checks.append(
            GateCheck(
                name="All-Atom Sidechain Completeness",
                metric=round(structure.sidechain_completeness * 100.0, 1),
                unit="% non-Gly residues with heavy sidechains beyond CB",
                reference_source="Coventry / Baker Lab RFdiffusion->ProteinMPNN->AF2 protocol (Watson et al., Nature 2023)",
                green_range=">= 95%",
                amber_range="80% - 95%",
                red_range="< 80% (Refused as INVALID_BACKBONE_ONLY)",
                flag="INVALID_BACKBONE_ONLY",
                detail=(
                    f"Sidechain completeness is {structure.sidechain_completeness:.1%}; "
                    f"SAP, BUNS, SCM, and sidechain electrostatics cannot be computed on backbone-only or CB-stubbed models."
                ),
            )
        )
        flags.append("INVALID_BACKBONE_ONLY: missing atomic sidechain rotamers")
        recommendations.append(
            "Run inverse sequence design and full-atom rotamer packing (ProteinMPNN + FastRelax / AlphaFold2) prior to biophysical gating."
        )
        return DevelopabilityVerdict(
            status="INVALID_BACKBONE_ONLY",
            is_backbone_only=True,
            sidechain_completeness=structure.sidechain_completeness,
            green_count=0,
            amber_count=0,
            red_count=0,
            gate_checks=checks,
            flags=flags,
            recommendations=recommendations,
        )

    sap_m = sap_res.metrics
    stab_m = stab_res.metrics

    # 0. Partial Sidechain Truncation Check (when 80% <= completeness < 95%, e.g. low-res cryo-EM)
    if structure.sidechain_completeness < 0.95:
        checks.append(
            GateCheck(
                name="All-Atom Sidechain Completeness",
                metric=round(structure.sidechain_completeness * 100.0, 1),
                unit="% non-Gly residues with heavy sidechains beyond CB",
                reference_source="Coventry / Baker Lab & OPIG surface modeling requirements",
                green_range=">= 95%",
                amber_range="80% - 95%",
                red_range="< 80% (Refused as INVALID_BACKBONE_ONLY)",
                flag="AMBER",
                detail=(
                    f"Sidechain completeness is {structure.sidechain_completeness:.1%}; "
                    f"some solvent-facing sidechain atoms are truncated in coordinates, adding uncertainty to local SASA/SAP/SCM."
                ),
            )
        )
        flags.append(
            f"PARTIAL_SIDECHAIN_TRUNCATION_AMBER: {structure.sidechain_completeness:.1%} sidechain completeness (<95%)"
        )
        recommendations.append(
            "Build/repack missing sidechain atoms on truncated surface residues before relying on residue-level SAP or SCM."
        )

    # 1. Coventry SapScoreMetric + Chennamsetty Hot-spot Gate
    # Note: SapScoreMetric (<25 Green, 25-35 Amber, >35 Red) was calibrated on ~55-75 aa miniproteins (Cao et al., 2022).
    # For larger proteins (>90 aa, e.g. Fab/mAb), also evaluate mean_positive_sap (<=0.06 Green, <=0.10 Amber, >0.10 Red).
    sap_score_val = sap_m.sap_score or 0.0
    mean_pos_sap = sap_m.mean_positive_sap or 0.0
    max_sap_val = sap_m.max_sap or 0.0

    n_res = len(structure.residues)
    if n_res <= 90:
        if sap_score_val <= config.coventry_sap_green_max and max_sap_val < 0.35:
            sap_flag: TrafficLight = "GREEN"
        elif sap_score_val <= config.coventry_sap_amber_max and max_sap_val < 0.45:
            sap_flag = "AMBER"
        else:
            sap_flag = "RED"
        sap_green_desc = f"sap_score <= {config.coventry_sap_green_max:.0f} and max_sap < 0.35"
        sap_amber_desc = f"{config.coventry_sap_green_max:.0f} < sap_score <= {config.coventry_sap_amber_max:.0f}"
        sap_red_desc = f"sap_score > {config.coventry_sap_amber_max:.0f} or max_sap >= 0.45"
    else:
        if mean_pos_sap <= 0.045 and max_sap_val < 0.35:
            sap_flag = "GREEN"
        elif mean_pos_sap <= 0.070 and max_sap_val < 0.45:
            sap_flag = "AMBER"
        else:
            sap_flag = "RED"
        sap_green_desc = "mean_atom_sap <= 0.045 and max_sap < 0.35"
        sap_amber_desc = "0.045 < mean_atom_sap <= 0.070 or 0.35 <= max_sap < 0.45"
        sap_red_desc = "mean_atom_sap > 0.070 or max_sap >= 0.45"

    checks.append(
        GateCheck(
            name="Spatial Aggregation Propensity (Coventry SapScore / Chennamsetty SAP)",
            metric=sap_score_val if n_res <= 90 else mean_pos_sap,
            unit="SAP atom-sum" if n_res <= 90 else "SAP/atom",
            reference_source="Chennamsetty et al. PNAS 2009; Cao et al. Nature 2022",
            green_range=sap_green_desc,
            amber_range=sap_amber_desc,
            red_range=sap_red_desc,
            flag=sap_flag,
            detail=(
                f"sap_score={sap_score_val:.2f}, mean_pos_sap={mean_pos_sap:.4f}, "
                f"max_sap={max_sap_val:.3f}, hotspots={sap_m.aggregation_hotspot_count}, patches={len(sap_m.hydrophobic_patches)}."
            ),
        )
    )
    if sap_flag != "GREEN":
        flags.append(
            f"HYDROPHOBIC_PATCH_{sap_flag}: sap_score={sap_score_val:.2f}, max_sap={max_sap_val:.3f} across {sap_m.aggregation_hotspot_count} hot-spots"
        )
        if sap_m.aggregation_hotspot_residues:
            recommendations.append(
                f"Inspect surface hydrophobic patch residues ({', '.join(sap_m.aggregation_hotspot_residues[:5])}); "
                f"if not required for target binding, resurface with polar/charged residues."
            )

    # 2. Local Surface Electrostatic Patch Gates (Agrawal 2016 SCM_max/min & whole-protein cluster integrals)
    # Peak 10 A SCM >= +3.5 e or <= -3.5 e indicates strong localized electrostatic clustering (viscosity / polyspecificity)
    max_scm_pos = charge_metrics.max_scm_positive
    min_scm_neg = charge_metrics.min_scm_negative
    if max_scm_pos <= 3.2:
        ppc_flag: TrafficLight = "GREEN"
    elif max_scm_pos <= 4.5:
        ppc_flag = "AMBER"
    else:
        ppc_flag = "RED"
    checks.append(
        GateCheck(
            name="Surface Positive Charge Clustering (Agrawal 10 A SCM_max & Pair Integral)",
            metric=max_scm_pos,
            unit="e (peak 10 A exposed SCM)",
            reference_source="Agrawal et al. mAbs 2016 (exposed sidechain SASA >= 10 A^2)",
            green_range="<= +3.20 e",
            amber_range="+3.20 e to +4.50 e",
            red_range="> +4.50 e",
            flag=ppc_flag,
            detail=f"max_SCM_pos={max_scm_pos:+.2f} e, pair_integral={charge_metrics.ppc_positive_patch:.3f}, SCM_pos_sum={charge_metrics.scm_pos_sum:.1f}.",
        )
    )
    if ppc_flag != "GREEN":
        flags.append(f"POSITIVE_CHARGE_PATCH_{ppc_flag}: max_SCM_pos={max_scm_pos:+.2f} e (non-specific/heparin binding risk)")
        recommendations.append("Attenuate clustered positive surface charges (Arg/Lys) to lower non-specific binding and clearance risk.")

    abs_min_scm_neg = abs(min_scm_neg)
    if abs_min_scm_neg <= 3.5:
        pnc_flag: TrafficLight = "GREEN"
    elif abs_min_scm_neg <= 4.8:
        pnc_flag = "AMBER"
    else:
        pnc_flag = "RED"
    checks.append(
        GateCheck(
            name="Surface Negative Charge Clustering (Agrawal 10 A SCM_min & Pair Integral)",
            metric=min_scm_neg,
            unit="e (min 10 A exposed SCM)",
            reference_source="Agrawal et al. mAbs 2016 (high-concentration viscosity & self-association)",
            green_range=">= -3.50 e",
            amber_range="-4.80 e to -3.50 e",
            red_range="< -4.80 e",
            flag=pnc_flag,
            detail=f"min_SCM_neg={min_scm_neg:+.2f} e, pair_integral={charge_metrics.pnc_negative_patch:.3f}, SCM_neg_sum={charge_metrics.scm_neg_sum:.1f}.",
        )
    )
    if pnc_flag != "GREEN":
        flags.append(f"NEGATIVE_CHARGE_PATCH_{pnc_flag}: min_SCM_neg={min_scm_neg:+.2f} e, SCM_neg_sum={charge_metrics.scm_neg_sum:.1f}")
        recommendations.append("Disperse dense negative surface charge clusters (Asp/Glu) to reduce high-concentration self-association viscosity.")

    # 3. Charge Symmetry Gate: apply OPIG SFvCSP strictly to paired antibody-like chains (100..260 aa each, ratio <= 1.6);
    # for single chains or asymmetric binder-target complexes, evaluate global Spatial Charge Symmetry (CSS).
    chain_lengths = [len(seq) for seq in structure.sequence_by_chain.values()]
    is_antibody_like_pair = (
        len(chain_lengths) == 2
        and min(chain_lengths) >= 95
        and max(chain_lengths) <= 260
        and (max(chain_lengths) / max(min(chain_lengths), 1)) <= 1.6
    )
    if is_antibody_like_pair and charge_metrics.interchain_sfvcsp is not None:
        sfv = charge_metrics.interchain_sfvcsp
        if sfv > OPIG_TAP_SFVCSP_GREEN_MIN:
            sym_flag: TrafficLight = "GREEN"
        elif sfv >= OPIG_TAP_SFVCSP_AMBER_MIN:
            sym_flag = "AMBER"
        else:
            sym_flag = "RED"
        checks.append(
            GateCheck(
                name=f"Inter-Chain Charge Symmetry (OPIG SFvCSP at pH {charge_metrics.ph:.1f})",
                metric=sfv,
                unit="e^2",
                reference_source="Raybould et al. PNAS 2019 (paired VH/VL Fv domain symmetry)",
                green_range=f"> {OPIG_TAP_SFVCSP_GREEN_MIN:.2f}",
                amber_range=f"{OPIG_TAP_SFVCSP_AMBER_MIN:.2f} to {OPIG_TAP_SFVCSP_GREEN_MIN:.2f}",
                red_range=f"< {OPIG_TAP_SFVCSP_AMBER_MIN:.2f}",
                flag=sym_flag,
                detail=f"SFvCSP={sfv:+.2f}, chain charges={charge_metrics.net_charge_by_chain}, CoM dipole={charge_metrics.dipole_moment_debye:.1f} D.",
            )
        )
    else:
        css = charge_metrics.charge_symmetry_score
        if css >= config.min_charge_symmetry_amber:
            sym_flag = "GREEN"
        elif css >= config.min_charge_symmetry_red:
            sym_flag = "AMBER"
        else:
            sym_flag = "RED"
        extra_note = (
            f", inter-chain qA*qB={charge_metrics.interchain_sfvcsp:+.2f} (binder-target electrostatic complementarity)"
            if charge_metrics.interchain_sfvcsp is not None
            else ""
        )
        checks.append(
            GateCheck(
                name=f"Spatial Charge Symmetry (CSS at pH {charge_metrics.ph:.1f})",
                metric=css,
                unit="dimensionless [0..1]",
                reference_source="SE(3)-invariant CoM dipole & hemispheric charge mixing",
                green_range=f">= {config.min_charge_symmetry_amber:.2f}",
                amber_range=f"{config.min_charge_symmetry_red:.2f} - {config.min_charge_symmetry_amber:.2f}",
                red_range=f"< {config.min_charge_symmetry_red:.2f}",
                flag=sym_flag,
                detail=f"CSS={css:.3f}, CoM dipole={charge_metrics.dipole_moment_debye:.1f} D, pI={charge_metrics.isoelectric_point:.2f}{extra_note}.",
            )
        )
    if sym_flag != "GREEN":
        flags.append(f"CHARGE_SYMMETRY_{sym_flag}: pI={charge_metrics.isoelectric_point:.2f}, dipole={charge_metrics.dipole_moment_debye:.1f} D")
        recommendations.append("Rebalance oppositely charged surface faces to reduce electrostatic dipole self-association.")

    # 4. Coventry Atomic BUNS Gate
    buns_per_100 = (stab_m.buns_count * 100.0) / max(n_res, 1)
    if buns_per_100 <= 3.0:
        buns_flag: TrafficLight = "GREEN"
    elif buns_per_100 <= 7.0:
        buns_flag = "AMBER"
    else:
        buns_flag = "RED"
    checks.append(
        GateCheck(
            name="Atomic Buried Unsatisfied Polar Atoms (BUNS)",
            metric=float(stab_m.buns_count),
            unit="atoms (SASA < 0.5 A^2, 0 H-bonds)",
            reference_source="Coventry & Baker (Cao et al., Nature 2022)",
            green_range="<= 3.0 per 100 residues",
            amber_range="3.0 - 7.0 per 100 residues",
            red_range="> 7.0 per 100 residues",
            flag=buns_flag,
            detail=f"buns_count={stab_m.buns_count} ({buns_per_100:.1f}/100 res), H-bond satisfaction={stab_m.hbond_satisfaction_rate:.1%}.",
        )
    )
    if buns_flag != "GREEN":
        flags.append(f"ATOMIC_BUNS_{buns_flag}: {stab_m.buns_count} buried unsatisfied polar atoms ({', '.join(stab_m.buns_atoms[:5])})")
        recommendations.append("Satisfy buried polar atoms via internal H-bond networks or replace with hydrophobic packing.")

    # 5. MolProbity Heavy-Atom Clashscore Gate
    cs = stab_m.molprobity_heavy_clashscore
    if cs <= MOLPROBITY_CLASH_GREEN_MAX:
        clash_flag: TrafficLight = "GREEN"
    elif cs <= MOLPROBITY_CLASH_AMBER_MAX:
        clash_flag = "AMBER"
    else:
        clash_flag = "RED"
    checks.append(
        GateCheck(
            name="MolProbity Heavy-Atom Clashscore (overlap >= 0.40 A)",
            metric=cs,
            unit="clashes / 1,000 heavy atoms",
            reference_source="Word et al. J. Mol. Biol. 1999 (MolProbity)",
            green_range=f"<= {MOLPROBITY_CLASH_GREEN_MAX:.1f}",
            amber_range=f"{MOLPROBITY_CLASH_GREEN_MAX:.1f} - {MOLPROBITY_CLASH_AMBER_MAX:.1f}",
            red_range=f"> {MOLPROBITY_CLASH_AMBER_MAX:.1f}",
            flag=clash_flag,
            detail=f"clashscore={cs:.2f} ({stab_m.steric_clash_count} overlapping pairs), packing={stab_m.packing_score:.3f}.",
        )
    )
    if clash_flag != "GREEN":
        flags.append(f"MOLPROBITY_CLASH_{clash_flag}: clashscore={cs:.2f} ({stab_m.steric_clash_count} pairs)")
        recommendations.append("Relax sidechain rotamers (e.g. OpenMM / FastRelax / FoldX RepairPDB) before trusting surface patch integrals.")

    # 6. OPIG ML Structure Confidence & Rotamer Uncertainty Gate (AF2 / ESMFold / ImmuneBuilder)
    ml_conf_flag: Optional[Literal["GREEN", "AMBER", "RED"]] = None
    low_conf_surface = [
        r.short_label
        for r in structure.residues
        if structure.plddt_detected and r.mean_b_factor < 70.0 and sap_res.res_rsasa[r.index] >= 0.15
    ]
    low_conf_count = (
        sum(1 for r in structure.residues if r.mean_b_factor < 70.0) if structure.plddt_detected else 0
    )

    if structure.plddt_detected and stab_m.mean_plddt is not None and stab_m.low_plddt_fraction is not None:
        if stab_m.mean_plddt >= 80.0 and stab_m.low_plddt_fraction <= 0.10:
            ml_conf_flag = "GREEN"
        elif stab_m.mean_plddt >= 70.0 and stab_m.low_plddt_fraction <= 0.25:
            ml_conf_flag = "AMBER"
        else:
            ml_conf_flag = "RED"
        checks.append(
            GateCheck(
                name="ML Model Confidence & Rotamer Uncertainty (pLDDT < 70)",
                metric=stab_m.mean_plddt,
                unit="pLDDT [0..100]",
                reference_source="Abanades et al. Comm. Biol. 2023; Raybould et al. Comm. Biol. 2024",
                green_range="mean >= 80 & <10% residues below pLDDT 70",
                amber_range="mean >= 70 & <=25% residues below pLDDT 70",
                red_range="mean < 70 or >25% residues below pLDDT 70",
                flag=ml_conf_flag,
                detail=(
                    f"mean_pLDDT={stab_m.mean_plddt:.1f}, {stab_m.low_plddt_fraction:.1%} ({low_conf_count} res) < 70 pLDDT, "
                    f"{len(low_conf_surface)} low-confidence surface residues ({structure.detected_model_source})."
                ),
            )
        )
        if ml_conf_flag != "GREEN":
            flags.append(
                f"ML_ROTAMER_UNCERTAINTY_{ml_conf_flag}: {stab_m.low_plddt_fraction:.1%} residues have pLDDT < 70; surface SAP/SCM may fluctuate with rotamer jitter"
            )
            recommendations.append(
                "Sample conformational/rotamer ensembles on low-pLDDT (<70) surface regions before acting on single-frame SAP or charge patches."
            )

    flag_list = [c.flag for c in checks]
    green_n = sum(1 for f in flag_list if f == "GREEN")
    amber_n = sum(1 for f in flag_list if f == "AMBER")
    red_n = sum(1 for f in flag_list if f == "RED")

    if red_n > 0:
        overall: TrafficLight = "RED"
    elif amber_n > 0:
        overall = "AMBER"
    else:
        overall = "GREEN"

    if overall == "GREEN" and not recommendations:
        recommendations.append("All evaluated biophysical gates fall within GREEN empirical reference bounds.")
    elif not recommendations:
        recommendations.append("Review flagged AMBER/RED gates before advancing candidate to expression.")

    return DevelopabilityVerdict(
        status=overall,
        is_backbone_only=False,
        sidechain_completeness=structure.sidechain_completeness,
        green_count=green_n,
        amber_count=amber_n,
        red_count=red_n,
        ml_confidence_flag=ml_conf_flag,
        low_confidence_residue_count=low_conf_count,
        low_confidence_surface_residues=low_conf_surface,
        gate_checks=checks,
        flags=flags,
        recommendations=recommendations,
    )


class BiophysGuardPipeline:
    """Main pipeline orchestrator for evaluating protein developability."""

    def __init__(self, config: Optional[BiophysGuardConfig] = None) -> None:
        self.config = config or BiophysGuardConfig()

    def analyze(
        self,
        structure_path: str | Path,
        output_dir: Optional[str | Path] = None,
        prefix: Optional[str] = None,
        variants: Optional[List[str]] = None,
        write_outputs: bool = True,
    ) -> BiophysGuardReport:
        """Run full biophysical filtering pipeline and optionally emit JSON & 3D overlays."""
        structure = load_structure_file(structure_path)
        cfg = self.config

        sap_res = compute_sap_metrics(
            structure=structure,
            probe_radius=cfg.probe_radius,
            n_sphere_points=cfg.n_sphere_points,
            sap_radius=cfg.sap_radius,
            sap_macro_radius=cfg.sap_macro_radius,
            hotspot_threshold=cfg.sap_hotspot_threshold,
        )

        charge_res = compute_charge_metrics(
            structure=structure,
            res_sasa_sidechain=sap_res.res_sasa_sidechain,
            res_rsasa=sap_res.res_rsasa,
            total_sasa_a2=sap_res.metrics.total_sasa_a2,
            ph=cfg.ph,
            scm_radius=cfg.scm_radius,
        )

        stab_res = compute_stability_metrics(
            structure=structure,
            atom_sasa=sap_res.atom_sasa,
            res_rsasa=sap_res.res_rsasa,
            res_sap_r5=sap_res.res_sap_r5,
            res_charges=charge_res.res_charges,
            disulfide_pairs=charge_res.disulfide_pairs,
            salt_bridge_pairs=charge_res.salt_bridge_pairs,
            variants=variants,
            backend_preference=cfg.stability_backend,
            run_alanine_scan=cfg.run_alanine_scan,
            foldx_binary=cfg.foldx_binary,
            esm_model_name=cfg.esm_model_name,
            rsasa_core_threshold=cfg.rsasa_core_threshold,
        )

        profiles, risk_map = classify_residues_and_scores(
            structure=structure,
            sap_res=sap_res,
            stab_res=stab_res,
            res_charges=charge_res.res_charges,
            res_effective_pka=charge_res.res_effective_pka,
            res_scm=charge_res.res_scm_r10,
        )

        verdict = _build_verdict(
            structure=structure,
            sap_res=sap_res,
            charge_metrics=charge_res.metrics,
            stab_res=stab_res,
            config=cfg,
        )

        report = BiophysGuardReport(
            generated_at_utc=datetime.now(timezone.utc).isoformat(),
            input_file=str(structure.input_path),
            structure_format=structure.structure_format,
            detected_model_source=structure.detected_model_source,
            is_backbone_only=structure.is_backbone_only,
            sidechain_completeness=structure.sidechain_completeness,
            chain_ids=structure.chain_ids,
            residue_count=len(structure.residues),
            atom_count=len(structure.atoms),
            sequence_by_chain=structure.sequence_by_chain,
            config=cfg,
            verdict=verdict,
            sap_metrics=sap_res.metrics,
            charge_metrics=charge_res.metrics,
            stability_metrics=stab_res.metrics,
            residues=profiles,
            viewer_overlays=ViewerOverlaySpec(
                red_hotspot_residues=[
                    f"{r.chain_id}:{r.aa1}{r.auth_seq_id}{r.ins_code}"
                    for r in profiles
                    if r.risk_category == "aggregation_hotspot"
                ],
                blue_core_residues=[
                    f"{r.chain_id}:{r.aa1}{r.auth_seq_id}{r.ins_code}"
                    for r in profiles
                    if r.risk_category == "stable_core"
                ],
                destabilized_residues=[
                    f"{r.chain_id}:{r.aa1}{r.auth_seq_id}{r.ins_code}"
                    for r in profiles
                    if r.risk_category == "destabilized"
                ],
            ),
        )

        if write_outputs:
            out_dir = Path(output_dir) if output_dir else Path("outputs")
            out_prefix = prefix or structure.input_path.stem
            overlays = generate_all_overlays(
                structure=structure,
                report=report,
                risk_map=risk_map,
                output_dir=out_dir,
                prefix=out_prefix,
            )
            report.viewer_overlays = overlays
            report_path = out_dir / f"{out_prefix}_report.json"
            report_path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")

        return report


def evaluate_structure(
    structure_path: str | Path,
    output_dir: Optional[str | Path] = None,
    variants: Optional[List[str]] = None,
    config: Optional[BiophysGuardConfig] = None,
    write_outputs: bool = True,
) -> BiophysGuardReport:
    """Convenience functional API for single-structure evaluation."""
    return BiophysGuardPipeline(config=config).analyze(
        structure_path=structure_path,
        output_dir=output_dir,
        variants=variants,
        write_outputs=write_outputs,
    )
