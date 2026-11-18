"""Command-line interface for biophys-guard."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import List, Optional

from .pipeline import BiophysGuardPipeline
from .schema import BiophysGuardConfig, BiophysGuardReport, export_json_schema


def _print_summary(report: BiophysGuardReport) -> None:
    v = report.verdict
    s = report.sap_metrics
    c = report.charge_metrics
    st = report.stability_metrics

    print("=" * 76)
    print(f"biophys-guard Report: {Path(report.input_file).name}")
    print(
        f"Source: {report.detected_model_source} ({report.structure_format.upper()}) | "
        f"Residues: {report.residue_count} | Chains: {', '.join(report.chain_ids)} | "
        f"Sidechain completeness: {report.sidechain_completeness:.1%}"
    )
    print("-" * 76)
    print(f"TRAFFIC-LIGHT VERDICT: {v.status} (GREEN={v.green_count}, AMBER={v.amber_count}, RED={v.red_count})")

    if v.is_backbone_only:
        print(f"  [GATED]     {s.status_reason}")
    else:
        print(
            f"  [SAP]       Coventry SapScore: {s.sap_score:.2f} | Max SAP(R=5A): {s.max_sap:.4f} | "
            f"Hot-spots: {s.aggregation_hotspot_count} | Patches: {len(s.hydrophobic_patches)} | PSH: {s.psh_surface_hydrophobicity:.2f}"
        )
        print(
            f"  [Charge]    pH {c.ph:.1f} Net Q: {c.net_charge:+.2f} | pI: {c.isoelectric_point:.2f} | "
            f"CoM Dipole: {c.dipole_moment_debye:.1f} D | CSS: {c.charge_symmetry_score:.3f} | "
            f"PPC={c.ppc_positive_patch:.2f} PNC={c.pnc_negative_patch:.2f} SCM_neg={c.scm_neg_sum:.2f}"
        )
        if c.interchain_sfvcsp is not None:
            print(f"              OPIG SFvCSP (inter-chain): {c.interchain_sfvcsp:+.2f} | Chain charges: {c.net_charge_by_chain}")
        print(
            f"  [Structure] Backend: {st.backend_used} | Score: {st.score_value:+.2f} {st.score_unit} "
            f"(calibrated={st.is_calibrated_free_energy}) | Packing: {st.packing_score:.3f} | "
            f"BUNS: {st.buns_count} | MolProbity Clashscore: {st.molprobity_heavy_clashscore:.2f} ({st.steric_clash_count} pairs)"
        )

    for g in v.gate_checks:
        print(f"    [{g.flag:<21s}] {g.name}: {g.detail}")

    if st.variant_scores:
        print("  [Variants]")
        for vs in st.variant_scores:
            print(f"    - {vs.mutation}: {vs.score_value:+.3f} {vs.score_unit} ({vs.classification}) [{vs.rationale}]")

    if report.viewer_overlays.risk_pdb_path:
        print("  [Artifacts]")
        print(f"    * PyMOL script : {report.viewer_overlays.pymol_script_path}")
        print(f"    * Risk PDB     : {report.viewer_overlays.risk_pdb_path}")
        print(f"    * NGL Viewer   : {report.viewer_overlays.ngl_viewer_path}")
        print(f"    * Mol* Spec    : {report.viewer_overlays.molstar_sidecar_path}")
    print("=" * 76)


def _cmd_analyze(args: argparse.Namespace) -> int:
    variants: List[str] = []
    if args.variants:
        for item in args.variants:
            variants.extend([x.strip() for x in item.split(",") if x.strip()])

    cfg = BiophysGuardConfig(
        ph=args.ph,
        sap_radius=args.sap_radius,
        stability_backend=args.stability_backend,
        run_alanine_scan=not args.no_ala_scan,
    )
    pipeline = BiophysGuardPipeline(config=cfg)
    report = pipeline.analyze(
        structure_path=args.structure,
        output_dir=args.output_dir,
        prefix=args.prefix,
        variants=variants,
        write_outputs=True,
    )
    _print_summary(report)
    if args.schema_out:
        export_json_schema(args.schema_out)
    return 0


def _cmd_export_schema(args: argparse.Namespace) -> int:
    path = export_json_schema(args.output)
    print(f"Exported JSON Schema to {path}")
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    repo_root = Path(__file__).resolve().parent.parent
    examples_dir = repo_root / "examples"
    ubq_pdb = examples_dir / "1UBQ_ubiquitin.pdb"
    lcb1_mono = examples_dir / "7JZU_LCB1_monomer.pdb"
    lcb1_cpx = examples_dir / "7JZU_LCB1_RBD_complex.pdb"
    fab_pdb = examples_dir / "1N8Z_Trastuzumab_Fab.pdb"
    af_p53 = examples_dir / "AF_P04637_p53.cif"
    bb_only = examples_dir / "7JZU_LCB1_backbone_only.pdb"

    if not ubq_pdb.exists() or not lcb1_mono.exists():
        sys.path.insert(0, str(repo_root))
        from examples.prepare_real_benchmarks import prepare_benchmarks

        prepare_benchmarks(examples_dir)

    out_dir = Path(args.output_dir)
    schema_path = export_json_schema(repo_root / "schemas" / "biophys_guard_report.schema.json")
    print(f"Schema exported: {schema_path}")

    pipeline = BiophysGuardPipeline()
    r_ubq = pipeline.analyze(ubq_pdb, output_dir=out_dir, write_outputs=True)
    _print_summary(r_ubq)

    r_lcb1 = pipeline.analyze(lcb1_mono, output_dir=out_dir, variants=["A:M13K", "A:L31E", "A:I12P"], write_outputs=True)
    _print_summary(r_lcb1)

    r_cpx = pipeline.analyze(lcb1_cpx, output_dir=out_dir, write_outputs=True)
    _print_summary(r_cpx)

    if fab_pdb.exists():
        r_fab = pipeline.analyze(fab_pdb, output_dir=out_dir, write_outputs=True)
        _print_summary(r_fab)

    if af_p53.exists():
        r_af = pipeline.analyze(af_p53, output_dir=out_dir, write_outputs=True)
        _print_summary(r_af)

    if bb_only.exists():
        r_bb = pipeline.analyze(bb_only, output_dir=out_dir, write_outputs=True)
        _print_summary(r_bb)

    print(
        json.dumps(
            {
                "1UBQ_status": r_ubq.verdict.status,
                "1UBQ_sap_score": r_ubq.sap_metrics.sap_score,
                "7JZU_LCB1_monomer_status": r_lcb1.verdict.status,
                "7JZU_LCB1_monomer_sap_score": r_lcb1.sap_metrics.sap_score,
                "7JZU_LCB1_complex_sap_by_chain": r_cpx.sap_metrics.sap_score_by_chain,
            },
            indent=2,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="biophys-guard",
        description="Biophysical Constraint-Filtering Harness for Generated Proteins.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_an = sub.add_parser("analyze", help="Analyze a PDB or mmCIF protein structure.")
    p_an.add_argument("structure", help="Path to input .pdb or .cif/.mmcif file.")
    p_an.add_argument("-o", "--output-dir", default="outputs", help="Directory for JSON report and 3D overlays.")
    p_an.add_argument("--prefix", default=None, help="Output filename prefix (defaults to input stem).")
    p_an.add_argument("--ph", type=float, default=7.4, help="Solution pH for electrostatics (default: 7.4).")
    p_an.add_argument("--sap-radius", type=float, default=5.0, help="SAP neighborhood cutoff radius in A (default: 5.0).")
    p_an.add_argument(
        "--stability-backend",
        choices=["auto", "foldx", "esm1v", "biophysical"],
        default="auto",
        help="Thermodynamic stability / variant scoring backend.",
    )
    p_an.add_argument(
        "--variants",
        nargs="*",
        default=[],
        help="Comma- or space-separated mutations to score, e.g. A:M13K,A:I31E.",
    )
    p_an.add_argument("--no-ala-scan", action="store_true", help="Omit full alanine scan list from JSON output.")
    p_an.add_argument("--schema-out", default=None, help="Optional path to also write JSON Schema.")
    p_an.set_defaults(func=_cmd_analyze)

    p_sc = sub.add_parser("export-schema", help="Export the BiophysGuardReport JSON Schema.")
    p_sc.add_argument(
        "-o",
        "--output",
        default="schemas/biophys_guard_report.schema.json",
        help="Output JSON Schema file path.",
    )
    p_sc.set_defaults(func=_cmd_export_schema)

    p_demo = sub.add_parser("demo", help="Run the real experimental & de novo benchmark suite.")
    p_demo.add_argument("-o", "--output-dir", default="outputs", help="Output directory for demo artifacts.")
    p_demo.set_defaults(func=_cmd_demo)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
