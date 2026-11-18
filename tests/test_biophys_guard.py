"""Rigorous primary-source verification & SE(3) invariance test suite for biophys-guard."""

from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from biophys_guard import (
    BiophysGuardConfig,
    BiophysGuardPipeline,
    BiophysGuardReport,
    evaluate_structure,
    export_json_schema,
)
from biophys_guard.io_parser import load_structure_file, write_sanitized_pdb


EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"


def _rodrigues_matrix(axis: np.ndarray, angle_deg: float) -> np.ndarray:
    k = axis / np.linalg.norm(axis)
    kx = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    theta = math.radians(angle_deg)
    return np.eye(3) + math.sin(theta) * kx + (1.0 - math.cos(theta)) * (kx @ kx)


class TestBiophysGuardPrimaryPrinciples(unittest.TestCase):
    """Test invariance, backbone gating, coupled pKa, atomic BUNS, MolProbity clashes, and real PDBs."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmpdir = tempfile.TemporaryDirectory()
        cls.tmp_path = Path(cls._tmpdir.name)
        cls.ubq_pdb = EXAMPLES_DIR / "1UBQ_ubiquitin.pdb"
        cls.ubq_cif = EXAMPLES_DIR / "1UBQ_ubiquitin.cif"
        cls.lcb1_mono = EXAMPLES_DIR / "7JZU_LCB1_monomer.pdb"
        cls.lcb1_cpx = EXAMPLES_DIR / "7JZU_LCB1_RBD_complex.pdb"
        cls.lcb1_bb = EXAMPLES_DIR / "7JZU_LCB1_backbone_only.pdb"
        cls.lcb1_stub = EXAMPLES_DIR / "7JZL_LCB1_cb_stub.pdb"
        cls.s5b_pdb = EXAMPLES_DIR / "7S5B_il7r_binder.pdb"
        cls.fab_pdb = EXAMPLES_DIR / "1N8Z_Trastuzumab_Fab.pdb"
        cls.af_p53 = EXAMPLES_DIR / "AF_P04637_p53.cif"

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmpdir.cleanup()

    def test_se3_translation_and_rotation_invariance_on_ubiquitin(self) -> None:
        """All physical metrics must be strictly invariant under arbitrary SO(3) rotation + R^3 translation."""
        from biophys_guard.electrostatics import compute_charge_metrics
        from biophys_guard.sap import compute_sap_metrics
        from biophys_guard.stability import compute_stability_metrics

        self.assertTrue(self.ubq_pdb.exists(), "Real 1UBQ_ubiquitin.pdb missing.")
        rot = _rodrigues_matrix(np.array([1.0, -2.3, 3.1]), 67.4)
        trans = np.array([142.5, -88.1, 319.7])

        # 1. Exact float64 in-memory SE(3) invariance (machine precision)
        s_orig = load_structure_file(self.ubq_pdb)
        s_mem = load_structure_file(self.ubq_pdb)
        s_mem.coords = (s_mem.coords @ rot.T) + trans
        s_mem.center_of_mass = (rot @ s_mem.center_of_mass) + trans
        for res in s_mem.residues:
            for attr in ("ca_coord", "cb_coord", "n_coord", "c_coord", "o_coord", "oxt_coord", "sc_centroid"):
                val = getattr(res, attr)
                if val is not None:
                    setattr(res, attr, (rot @ val) + trans)
            for atom in res.atoms:
                atom.coord = (rot @ atom.coord) + trans

        sap_o = compute_sap_metrics(s_orig)
        sap_m = compute_sap_metrics(s_mem)
        chg_o = compute_charge_metrics(s_orig, sap_o.res_sasa_sidechain, sap_o.res_rsasa, sap_o.metrics.total_sasa_a2)
        chg_m = compute_charge_metrics(s_mem, sap_m.res_sasa_sidechain, sap_m.res_rsasa, sap_m.metrics.total_sasa_a2)
        st_o = compute_stability_metrics(
            s_orig, sap_o.atom_sasa, sap_o.res_rsasa, sap_o.res_sap_r5, chg_o.res_charges,
            chg_o.disulfide_pairs, chg_o.salt_bridge_pairs, run_alanine_scan=False,
        )
        st_m = compute_stability_metrics(
            s_mem, sap_m.atom_sasa, sap_m.res_rsasa, sap_m.res_sap_r5, chg_m.res_charges,
            chg_m.disulfide_pairs, chg_m.salt_bridge_pairs, run_alanine_scan=False,
        )

        self.assertAlmostEqual(sap_o.metrics.total_sasa_a2, sap_m.metrics.total_sasa_a2, delta=1e-2)
        self.assertAlmostEqual(sap_o.metrics.sap_score, sap_m.metrics.sap_score, delta=1e-3)
        self.assertAlmostEqual(sap_o.metrics.max_sap, sap_m.metrics.max_sap, delta=1e-4)
        self.assertAlmostEqual(chg_o.metrics.net_charge, chg_m.metrics.net_charge, delta=1e-3)
        self.assertAlmostEqual(chg_o.metrics.isoelectric_point, chg_m.metrics.isoelectric_point, delta=1e-2)
        self.assertAlmostEqual(chg_o.metrics.dipole_moment_debye, chg_m.metrics.dipole_moment_debye, delta=1e-2)
        self.assertAlmostEqual(chg_o.metrics.charge_symmetry_score, chg_m.metrics.charge_symmetry_score, delta=1e-4)
        self.assertAlmostEqual(chg_o.metrics.scm_neg_sum, chg_m.metrics.scm_neg_sum, delta=1e-3)
        self.assertEqual(st_o.metrics.buns_count, st_m.metrics.buns_count)
        self.assertEqual(st_o.metrics.steric_clash_count, st_m.metrics.steric_clash_count)
        self.assertAlmostEqual(st_o.metrics.score_value, st_m.metrics.score_value, delta=1e-2)
        np.testing.assert_allclose(
            np.array(chg_m.metrics.dipole_vector_debye),
            rot @ np.array(chg_o.metrics.dipole_vector_debye),
            atol=0.05,
        )

        # 2. PDB %8.3f text round-trip invariance
        rot_pdb = self.tmp_path / "1ubq_rotated.pdb"
        write_sanitized_pdb(s_mem, rot_pdb, remarks=["EXPDTA    X-RAY DIFFRACTION"])
        cfg = BiophysGuardConfig(run_alanine_scan=False)
        rep_orig = BiophysGuardPipeline(cfg).analyze(self.ubq_pdb, write_outputs=False)
        rep_rot = BiophysGuardPipeline(cfg).analyze(rot_pdb, write_outputs=False)
        self.assertAlmostEqual(rep_orig.sap_metrics.total_sasa_a2, rep_rot.sap_metrics.total_sasa_a2, delta=1.5)
        self.assertAlmostEqual(rep_orig.sap_metrics.sap_score, rep_rot.sap_metrics.sap_score, delta=0.05)
        self.assertAlmostEqual(rep_orig.charge_metrics.dipole_moment_debye, rep_rot.charge_metrics.dipole_moment_debye, delta=0.5)
        self.assertEqual(rep_orig.stability_metrics.buns_count, rep_rot.stability_metrics.buns_count)
        self.assertEqual(rep_orig.stability_metrics.steric_clash_count, rep_rot.stability_metrics.steric_clash_count)

    def test_backbone_only_and_cb_stub_strict_refusal(self) -> None:
        """Backbone-only RFdiffusion and CB-stub cryo-EM models must be gated as INVALID_BACKBONE_ONLY."""
        for path in (self.lcb1_bb, self.lcb1_stub):
            rep = evaluate_structure(path, write_outputs=False)
            self.assertTrue(rep.is_backbone_only, f"Failed backbone-only detection for {path}")
            self.assertEqual(rep.verdict.status, "INVALID_BACKBONE_ONLY")
            self.assertFalse(rep.sap_metrics.calculable)
            self.assertIsNone(rep.sap_metrics.sap_score)
            self.assertIsNone(rep.sap_metrics.max_sap)
            self.assertFalse(rep.charge_metrics.calculable)
            self.assertFalse(rep.stability_metrics.calculable)
            self.assertEqual(rep.stability_metrics.score_unit, "not_applicable")
            self.assertTrue(all(r.risk_category == "not_applicable_backbone_only" for r in rep.residues))

    def test_real_experimental_and_de_novo_benchmarks(self) -> None:
        """Verify real PDB/mmCIF consistency, interface patch burial, multi-chain Fab, and AF2 pLDDT warnings."""
        cfg = BiophysGuardConfig(run_alanine_scan=False)
        r_ubq_pdb = BiophysGuardPipeline(cfg).analyze(self.ubq_pdb, write_outputs=False)
        r_ubq_cif = BiophysGuardPipeline(cfg).analyze(self.ubq_cif, write_outputs=False)
        self.assertEqual(r_ubq_pdb.detected_model_source, "Experimental")
        self.assertEqual(r_ubq_cif.detected_model_source, "Experimental")
        self.assertAlmostEqual(r_ubq_pdb.sap_metrics.sap_score, r_ubq_cif.sap_metrics.sap_score, delta=0.2)
        self.assertAlmostEqual(r_ubq_pdb.charge_metrics.net_charge, r_ubq_cif.charge_metrics.net_charge, delta=0.05)

        # Unbound de novo binder LCB1 exposes hydrophobic target-binding interface vs bound complex
        r_lcb1_mono = BiophysGuardPipeline(cfg).analyze(self.lcb1_mono, write_outputs=False)
        r_lcb1_cpx = BiophysGuardPipeline(cfg).analyze(self.lcb1_cpx, write_outputs=False)
        self.assertGreater(
            r_lcb1_mono.sap_metrics.sap_score,
            r_lcb1_cpx.sap_metrics.sap_score_by_chain["A"],
        )

        # High-resolution 1.50 A de novo IL-7R binder (7S5B)
        r_7s5b = BiophysGuardPipeline(cfg).analyze(self.s5b_pdb, write_outputs=False)
        self.assertEqual(r_7s5b.detected_model_source, "Experimental")
        self.assertFalse(r_7s5b.is_backbone_only)

        # Clinical Trastuzumab Fab (1N8Z chains A+B): multi-chain SFvCSP and disulfides
        r_fab = BiophysGuardPipeline(cfg).analyze(self.fab_pdb, write_outputs=False)
        self.assertIsNotNone(r_fab.charge_metrics.interchain_sfvcsp)
        self.assertGreaterEqual(r_fab.stability_metrics.disulfide_bond_count, 2)

        # Real AlphaFold DB p53 (AF_P04637): global pLDDT ~ 75.1 yet >40% residues < 70 must warn
        r_p53 = BiophysGuardPipeline(cfg).analyze(self.af_p53, write_outputs=False)
        self.assertEqual(r_p53.detected_model_source, "AlphaFold")
        self.assertEqual(r_p53.verdict.ml_confidence_flag, "RED")
        self.assertGreater(len(r_p53.verdict.low_confidence_surface_residues), 20)

    def test_propka_coupled_salt_bridge_shifts_pka_toward_ionized_state(self) -> None:
        """Salt-bridged buried ASP-LYS pair must shift ASP pKa downward relative to isolated buried ASP."""
        # Build a dense hydrophobic shell around ASP at origin, with or without LYS NZ at 2.9 A
        shell_coords = []
        idx = 1
        for dx in (-4.5, 0.0, 4.5):
            for dy in (-4.5, 0.0, 4.5):
                for dz in (-4.5, 4.5):
                    shell_coords.append((idx, dx, dy, dz))
                    idx += 1

        def write_buried_case(path: Path, with_salt_bridge: bool) -> None:
            lines = ["EXPDTA    X-RAY DIFFRACTION"]
            serial = 1
            # Res 1..18 LEU hydrophobic shell
            for r_id, x, y, z in shell_coords:
                for aname, off in [("N", (-0.8, 0, 0)), ("CA", (0, 0, 0)), ("C", (0.8, 0, 0)), ("O", (0.8, 0.8, 0)), ("CB", (0, -0.8, 0)), ("CG", (0, -1.6, 0))]:
                    lines.append(
                        f"ATOM  {serial:5d}  {aname:<3s} LEU A{r_id:4d}    "
                        f"{x+off[0]:8.3f}{y+off[1]:8.3f}{z+off[2]:8.3f}  1.00 15.00           {aname[0]:>2s}"
                    )
                    serial += 1
            # Central ASP (residue 100) with OD1/OD2 near (0, 0, 0)
            asp_atoms = [
                ("N", (-2.2, 0.0, 0.0), "N"),
                ("CA", (-1.8, 0.5, 0.0), "C"),
                ("C", (-1.8, 1.5, 0.0), "C"),
                ("O", (-2.5, 2.0, 0.0), "O"),
                ("CB", (-1.0, 0.0, 0.0), "C"),
                ("CG", (-0.4, 0.0, 0.0), "C"),
                ("OD1", (0.0, 0.5, 0.0), "O"),
                ("OD2", (0.0, -0.5, 0.0), "O"),
            ]
            for aname, (x, y, z), elem in asp_atoms:
                lines.append(
                    f"ATOM  {serial:5d}  {aname:<3s} ASP A 100    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 15.00           {elem:>2s}"
                )
                serial += 1
            if with_salt_bridge:
                lys_atoms = [
                    ("N", (5.5, 0.0, 0.0), "N"),
                    ("CA", (4.8, 0.0, 0.0), "C"),
                    ("C", (4.8, 1.4, 0.0), "C"),
                    ("O", (5.5, 1.9, 0.0), "O"),
                    ("CB", (4.0, 0.0, 0.0), "C"),
                    ("CG", (3.4, 0.0, 0.0), "C"),
                    ("CD", (2.9, 0.0, 0.0), "C"),
                    ("CE", (2.5, 0.0, 0.0), "C"),
                    ("NZ", (2.8, 0.0, 0.0), "N"),
                ]
                for aname, (x, y, z), elem in lys_atoms:
                    lines.append(
                        f"ATOM  {serial:5d}  {aname:<3s} LYS A 101    "
                        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 15.00           {elem:>2s}"
                    )
                    serial += 1
            lines.append("END")
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        iso_pdb = self.tmp_path / "asp_isolated.pdb"
        sb_pdb = self.tmp_path / "asp_saltbridge.pdb"
        write_buried_case(iso_pdb, with_salt_bridge=False)
        write_buried_case(sb_pdb, with_salt_bridge=True)

        rep_iso = evaluate_structure(iso_pdb, config=BiophysGuardConfig(run_alanine_scan=False), write_outputs=False)
        rep_sb = evaluate_structure(sb_pdb, config=BiophysGuardConfig(run_alanine_scan=False), write_outputs=False)

        asp_iso = next(r for r in rep_iso.residues if r.auth_seq_id == 100)
        asp_sb = next(r for r in rep_sb.residues if r.auth_seq_id == 100)
        self.assertGreater(asp_iso.effective_pka, 3.80)  # Desolvation raises acid pKa
        self.assertLess(asp_sb.effective_pka, asp_iso.effective_pka)  # Salt bridge lowers acid pKa back down
        self.assertGreaterEqual(rep_sb.charge_metrics.salt_bridge_count, 1)

    def test_molprobity_clash_detection_and_epistemic_unit_honesty(self) -> None:
        """Detect 0.90 A non-bonded overlap (d=2.50 A C-C) and enforce non-kcal/mol heuristic honesty."""
        clash_pdb = self.tmp_path / "clash_250.pdb"
        clash_pdb.write_text(
            "\n".join(
                [
                    "EXPDTA    X-RAY DIFFRACTION",
                    "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 20.00           N",
                    "ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00 20.00           C",
                    "ATOM      3  C   ALA A   1       2.005   1.420   0.000  1.00 20.00           C",
                    "ATOM      4  O   ALA A   1       1.250   2.380   0.000  1.00 20.00           O",
                    "ATOM      5  CB  ALA A   1       1.980  -0.760   1.210  1.00 20.00           C",
                    "ATOM      6  N   ALA A   3      10.000   0.000   0.000  1.00 20.00           N",
                    "ATOM      7  CA  ALA A   3       8.540   0.000   0.000  1.00 20.00           C",
                    "ATOM      8  C   ALA A   3       8.000   1.420   0.000  1.00 20.00           C",
                    "ATOM      9  O   ALA A   3       8.750   2.380   0.000  1.00 20.00           O",
                    # Placed at 2.50 A from res 1 CB (vdW sum 3.40 -> overlap 0.90 A >= 0.40 A)
                    "ATOM     10  CB  ALA A   3       4.480  -0.760   1.210  1.00 20.00           C",
                    "END",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        rep = evaluate_structure(
            clash_pdb,
            variants=["A:A1K"],
            config=BiophysGuardConfig(run_alanine_scan=True),
            write_outputs=False,
        )
        self.assertGreaterEqual(rep.stability_metrics.steric_clash_count, 1)
        self.assertGreater(rep.stability_metrics.molprobity_heavy_clashscore, 0.0)
        self.assertEqual(rep.stability_metrics.score_unit, "heuristic_au")
        self.assertFalse(rep.stability_metrics.is_calibrated_free_energy)
        self.assertIsNone(rep.stability_metrics.delta_g_fold_kcal_mol)
        self.assertEqual(rep.stability_metrics.variant_scores[0].score_unit, "heuristic_au")
        self.assertIsNone(rep.stability_metrics.variant_scores[0].ddg_kcal_mol)

    def test_inscode_preservation_and_schema_roundtrip(self) -> None:
        """Verify insertion codes (100A), HTML offline fallback, and schema validation."""
        ins_pdb = self.tmp_path / "kabat_inscode.pdb"
        ins_pdb.write_text(
            "\n".join(
                [
                    "EXPDTA    X-RAY DIFFRACTION",
                    "ATOM      1  N   TRP A 100A      0.000   0.000   0.000  1.00 20.00           N",
                    "ATOM      2  CA  TRP A 100A      1.458   0.000   0.000  1.00 20.00           C",
                    "ATOM      3  C   TRP A 100A      2.005   1.420   0.000  1.00 20.00           C",
                    "ATOM      4  O   TRP A 100A      1.250   2.380   0.000  1.00 20.00           O",
                    "ATOM      5  CB  TRP A 100A      1.980  -0.760   1.210  1.00 20.00           C",
                    "ATOM      6  CG  TRP A 100A      3.450  -0.760   1.210  1.00 20.00           C",
                    "END",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        out_dir = self.tmp_path / "ins_out"
        rep = evaluate_structure(
            ins_pdb,
            output_dir=out_dir,
            variants=["A:W100AF"],
            config=BiophysGuardConfig(run_alanine_scan=False),
            write_outputs=True,
        )
        self.assertEqual(rep.residues[0].ins_code, "A")
        self.assertEqual(rep.stability_metrics.variant_scores[0].mutation, "A:W100AF")
        html_text = Path(rep.viewer_overlays.ngl_viewer_path).read_text(encoding="utf-8")
        self.assertIn('"100^A"', html_text)
        self.assertIn('"A:100A"', html_text)
        self.assertIn("offline-msg", html_text)

        schema_path = self.tmp_path / "schema.json"
        export_json_schema(schema_path)
        schema_obj = json.loads(schema_path.read_text(encoding="utf-8"))
        self.assertEqual(schema_obj["title"], "BiophysGuardReport")

        loaded = BiophysGuardReport.model_validate_json(
            (out_dir / "kabat_inscode_report.json").read_text(encoding="utf-8")
        )
        self.assertEqual(loaded.residues[0].ins_code, "A")


if __name__ == "__main__":
    unittest.main()
