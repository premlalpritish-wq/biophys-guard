# `biophys-guard` Adversarial Verification Audit Trail

---

## Iteration 0 — Baseline Audit (`2026-11-18`)

- **Auditor:** `@deane_coventry_twin` (`bbabcb83-0400-4023-9dcd-5ea1e589deec`, read-only)
- **Verdict:** **`NO-GO`**
- **Score:** **`1.90 / 10.00`**

### Rubric Breakdown

| # | Audit Rubric | Score | Max | Summary |
|---:|---|---:|---:|---|
| 1 | Empirical Grounding & Real Benchmark Validity | 0.30 | 2.00 | Synthetic 3-helix toy PDBs with broken $>6.3\text{ \AA}$ loop bonds (`ASP 17`, `ASP 36`), spoofed `ALPHAFOLD` headers & synthetic pLDDT values, zero real PDB/AF2 benchmarks |
| 2 | SAP, SASA & Backbone-Only Gating Rigor | 0.55 | 2.00 | Virtual-$C_\beta$ spoofing + `atom_sasa=0` zero-SAP bug on RFdiffusion backbones, Gly-X-Gly + spurious `GLY=60.0` + clipped `rSASA=1.2`, missing Coventry `sap_score`, 96-pt coarse SASA |
| 3 | Electrodynamics, Charge Symmetry & $\text{p}K_a$ Coupling | 0.35 | 2.00 | Forbidden $Q_{\text{net}}/N_{\text{res}}$ subtraction & unweighted `sc_centroid` CoM in dipole, merged terminal/sidechain charges on `sc_centroid`, missing Coulombic/HB $\text{p}K_a$ coupling, unmasked core `SCM` |
| 4 | Thermodynamic Epistemic Honesty, BUNS & Clashes | 0.30 | 2.00 | `kcal/mol` stamped on ESM-1v LLR & uncalibrated static heuristics, contradictory Ala-scan cavities (`ddG < 0`), per-residue chemistry-blind BUNS, non-MolProbity `0.72*r_sum` clash cutoff |
| 5 | OPIG Calibration, Confidence Awareness & Invariance Tests | 0.40 | 2.00 | Arbitrary composite viability score, missing Green/Amber/Red empirical quantiles & TAP descriptors (`PSH/PPC/PNC/SFvCSP`), missing `pLDDT < 70` rotamer uncertainty flags, headerless crystal misclassification, zero SE(3) invariance tests |

### Critical Blockers to Resolve in Remediation Pass 1
1. Replace synthetic toy benchmarks with real RCSB PDB (`1UBQ`, `7JZL`/`LCB1`, `1N8Z` Fab with Kabat insertion codes) and real AlphaFold DB (`P04637` / `P00533`) structures.
2. Enforce strict backbone-only gating (`is_backbone_only=True`, `sap_calculable=False`, `status="INVALID_BACKBONE_ONLY"`); remove virtual-$C_\beta$ metric spoofing.
3. Implement exact Chennamsetty (2009) SAP (Ala-X-Ala side-chain maxima, `GLY=0.0`, all-residue heavy-atom averaging, `>= 590`-pt SASA tessellation) + Coventry `sap_score` ($\sum \max(0, \text{SAP}_r)$) + hydrophobic-only patch area.
4. Implement classical origin-invariant dipole around atomic mass-weighted CoM (`4.803204`, no $Q_{\text{net}}/N$ subtraction), chemical functional-group charge centers, iterative PROPKA3-inspired coupled $\text{p}K_a$ (smooth neighbor density desolvation + Coulombic + H-bond), exposure-filtered atomic `SCM` ($\text{SASA}_{\text{sc}} \ge 10\text{ \AA}^2$), and TAP `PPC`/`PNC`/`SFvCSP`.
5. Enforce strict unit honesty (`kcal/mol` ONLY for FoldX, `log_likelihood_ratio` for ESM-1v, `heuristic_au` for static structural heuristic without false thermodynamic claims), atomic BUNS (`SASA_atom < 0.5 A^2` + typed donor/acceptor H-bond rules), and MolProbity heavy-atom clashes (overlap $\ge 0.40\text{ \AA}$, 1-2/1-3/1-4 + H-bond exclusions, `clashscore` per 1,000 atoms).
6. Replace composite viability theater with OPIG empirical Green/Amber/Red traffic-light gates, preserve `ins_code` everywhere, flag `pLDDT < 70` surface rotamer uncertainty, and pass strict SE(3) translation/rotation invariance tests.

---

## Iteration 1 — Adversarial Re-Audit (`2026-11-18`)

- **Auditor:** `@deane_coventry_twin` (`bbabcb83-0400-4023-9dcd-5ea1e589deec`, read-only)
- **Verdict:** **`GO`**
- **Score:** **`9.00 / 10.00`** *(Up from `1.90 / 10.00` in Iteration 0)*

### Rubric Progression

| # | Audit Rubric | Iter 0 | Iter 1 | Max | Outcome |
|---:|---|---:|---:|---:|---|
| 1 | Empirical Grounding & Real Benchmark Validity | 0.30 | **1.85** | 2.00 | Real RCSB PDB/mmCIF & AFDB structures (`1UBQ`, `7S5B`, `7JZU`, `7JZL`, `1N8Z`, `AF_P04637`) replace synthetic toys |
| 2 | SAP, SASA & Backbone-Only Gating Rigor | 0.55 | **1.85** | 2.00 | Exact Ala-X-Ala SAP, `GLY=0.0`, Coventry `SapScoreMetric`, 590-pt $SO(3)$-equivariant SASA, strict `INVALID_BACKBONE_ONLY` refusal |
| 3 | Electrodynamics, Charge Symmetry & $\text{p}K_a$ Coupling | 0.35 | **1.80** | 2.00 | Mass-weighted CoM dipole (`4.803204`, zero $Q_{\text{net}}/N$), chemical charge centers, PROPKA3-style coupled titration, exposed-only SCM |
| 4 | Thermodynamic Epistemic Honesty, BUNS & Clashes | 0.30 | **1.80** | 2.00 | Unit separation (`kcal/mol` FoldX vs `log_likelihood_ratio` ESM-1v vs `heuristic_au`), atomic BUNS + MolProbity clashscore |
| 5 | OPIG Calibration, Confidence Awareness & Invariance Tests | 0.40 | **1.70** | 2.00 | Traffic-light flags, `pLDDT < 70` surface warnings on `AF_P04637`, `ins_code` preservation, 6/6 SE(3) & physical tests passing |
| | **TOTAL** | **1.90** | **9.00** | **10.00** | **`GO`** |

### Post-Audit Polish Applied (Iteration 1.1 Hardening)
Resolved all remaining `[MAJOR]` and `[MINOR]` items flagged by `@deane_coventry_twin`:
1. Added explicit electrostatic remediation recommendations on `AMBER`/`RED` `PPC`, `PNC`, and `CSS`/`SFvCSP` flags (`pipeline.py`), ensuring non-`GREEN` reports never emit an `"All evaluated biophysical gates fall within GREEN"` recommendation.
2. Added explicit `AMBER` sidechain-truncation gate (`80% <= sidechain_completeness < 95%`, flagging `7JZU_LCB1_monomer.pdb` at `86.8%`).
3. Restricted TAP `SFvCSP` gating strictly to paired antibody-like chains (`1N8Z_Trastuzumab_Fab`), preventing false warnings on asymmetric binder–antigen complexes (`7JZU_LCB1_RBD_complex`).
4. Removed unused `OPIG_TAP_PPC/PNC_*` dead constants (`constants.py`), removed arbitrary `* 10.0` `PSH` scaling (`sap.py`), fixed `"SAP/atom"` unit label and `Agrawal` SCM citation (`pipeline.py`), checked `EXPDTA` prior to ML keywords and removed bare `"RFD"` substring (`io_parser.py`), added `fair-esm` fallback alongside `transformers` (`stability.py`), and updated `README.md`.

---

## Iteration 2 — Final Verification Audit (`2026-11-18`)

- **Auditor:** `@deane_coventry_twin` (`bbabcb83-0400-4023-9dcd-5ea1e589deec`, read-only)
- **Verdict:** **`GO`** *(Zero `[CRITICAL]` or `[MAJOR]` blockers remaining)*
- **Final Score:** **`9.55 / 10.00`** *(Progression: `1.90` Iter 0 $\rightarrow$ `9.00` Iter 1 $\rightarrow$ `9.55` Iter 2)*

### Final Rubric Breakdown

| # | Audit Rubric | Iter 0 | Iter 1 | Iter 2 | Max | Final Verification Outcome |
|---:|---|---:|---:|---:|---:|---|
| 1 | Empirical Grounding & Real Benchmark Validity | 0.30 | 1.85 | **1.90** | 2.00 | Real RCSB/AFDB suite verified across 7 benchmarks + `README.md` synced (`-0.10`: no standalone ProThermDB/CST correlation table script) |
| 2 | SAP, SASA & Backbone-Only Gating Rigor | 0.55 | 1.85 | **1.95** | 2.00 | `80–95%` `PARTIAL_SIDECHAIN_TRUNCATION_AMBER` gate active (`7JZU_LCB1` `86.8%`, `7S5B` `94.3%`); `"SAP/atom"` unit fixed; `* 10.0` removed from `PSH` |
| 3 | Electrodynamics, Charge Symmetry & $\text{p}K_a$ Coupling | 0.35 | 1.80 | **1.90** | 2.00 | `SFvCSP` gate scoped to paired antibody-like chains (`1N8Z` `-0.34 GREEN`); binder–target complex (`7JZU`) evaluates `CSS=0.854` with complementarity disclosure (`-0.10`: length-based `is_antibody_like_pair` heuristic) |
| 4 | Thermodynamic Epistemic Honesty, BUNS & Clashes | 0.30 | 1.80 | **1.85** | 2.00 | `fair-esm` fallback implemented; strict `kcal/mol` vs `log_likelihood_ratio` vs `heuristic_au` unit separation (`-0.15`: intra-residue clashes skipped & uncalibrated static `heuristic_au` weights) |
| 5 | OPIG Calibration, Confidence Awareness & Invariance Tests | 0.40 | 1.70 | **1.95** | 2.00 | Electrostatic/sidechain remediation aligned across all non-`GREEN` verdicts; `EXPDTA` precedence enforced; 6/6 SE(3) & physical tests passing (`49.235s`) |
| | **TOTAL** | **1.90** | **9.00** | **9.55** | **10.00** | **`GO`** |

### Non-Blocking Observations (Tracked for Future Extensions)
- `is_antibody_like_pair` (`biophys_guard/pipeline.py:213-218`) uses a two-chain `[95, 260]` length and `<= 1.6` ratio heuristic; non-antibody homodimers/heterodimers in that length window would route through `SFvCSP` unless IMGT/CDR annotations are provided.
- `has_full_sidechain` (`biophys_guard/io_parser.py:433`) treats any sidechain atom beyond `CB` as present (offset by the `80–95%` `AMBER` truncation gate); `stability.py:359` skips intra-residue (`ra == rb`) contacts; `fair-esm` fallback (`stability.py:669`) assumes local hub cache and ESM-1v's 1022-residue limit.

---

## Iteration 3 / Phase 3 — Distill.pub Web Monograph Preflight Audit (`2026-11-18`)

- **Auditor:** `@deane_coventry_twin` (`bbabcb83-0400-4023-9dcd-5ea1e589deec`, read-only)
- **Preflight Progression:** `7.50 / 10.00 (NO-GO)` $\rightarrow$ **`9.65 / 10.00 (GO)`**
- **Scope Verified:** `scripts/export_web_monograph_assets.py`, `docs/index.html`, `docs/styles.css`, `docs/app.js`, `docs/assets/monograph_data.js`, `docs/assets/reports/*`, `docs/assets/structures/*`, and `README.md`.

### Preflight Rubric Progression

| # | Web Preflight Rubric | Preflight 0 | Preflight 1 | Max | Outcome |
|---:|---|---:|---:|---:|---|
| 1 | Telemetry Fidelity vs. `outputs/*_report.json` & `AUDIT_TRAIL.md` | 1.55 | **1.95** | 2.00 | Fixed `C_TERM`-contaminated Figure 4.1 charge via neutral `ALA 102` cap (`q = -0.941 -> -1.000`, $\text{p}K_a = 6.20 \rightarrow 3.66$) and verified exact parity across all 8 benchmark KPIs & SE(3) checks |
| 2 | Equation Parity & Epistemic Unit Honesty | 1.55 | **1.95** | 2.00 | Corrected `ESM-1v` LLR sign convention ($\log P(x_{\text{wt}} \mid x_{-i}) - \log P(x_{\text{mut}} \mid x_{-i})$) matching `stability.py:659,911`; double-escaped KaTeX strings; strict `FoldX` / `ESM-1v` / `heuristic_au` unit separation |
| 3 | Benchmark Provenance & Primary-Source Citations | 1.30 | **1.95** | 2.00 | Separated sequential `1N8Z` Trastuzumab Fab from the `A:W100AF` / `100^A` insertion-code unit-test fixture (`tests/test_biophys_guard.py`); verified all primary citations & RCSB/AFDB policy links |
| 4 | 3D Workbench, Risk Color Semantics & Insertion-Code Mechanics | 1.65 | **1.85** | 2.00 | Verified `#dc2626`/`#2563eb`/`#f97316`/`#e2e8f0`/`#94a3b8` palette parity, `INVALID_BACKBONE_ONLY` refusal banner, fallback 3D recoloring, and explicit 250-row / 6-patch display disclosures |
| 5 | Asset Export Determinism, Schema Metadata & `README.md` Sync | 1.45 | **1.95** | 2.00 | Fixed `BiophysGuardReport v2.0.0` hero badge, Pillar 2 `iter0_flaw` wording matching `AUDIT_TRAIL.md:16`, Figure 3.2 filter label, and `README.md` alignment |
| | **TOTAL** | **7.50** | **9.65** | **10.00** | **`GO` — Cleared for GitHub Pages Deployment** |

