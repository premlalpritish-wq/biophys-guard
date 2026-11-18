# `biophys-guard`: Biophysical Constraint-Filtering Harness for Generated Proteins

> **Live Interactive Distill.pub Monograph & 3D Structural Workbench:** https://premlalpritish-wq.github.io/biophys-guard/  
> **Source Repository:** https://github.com/premlalpritish-wq/biophys-guard  
> **Adversarial Audit Trajectory:** `Iteration 0: 1.90/10 (NO-GO)` $\rightarrow$ `Iteration 1: 9.00/10 (GO)` $\rightarrow$ `Iteration 2: 9.55/10 (GO)` ([AUDIT_TRAIL.md](AUDIT_TRAIL.md))

`biophys-guard` is a primary-source-grounded biophysical filtering and developability benchmarking pipeline for `.pdb` and `.cif`/`.mmcif` structures from **AlphaFold**, **ESMFold**, **ImmuneBuilder**, **RFdiffusion** (post-packing), or experimental structures.

## Primary-Source Methods & Physical Foundations

1. **Hydrophobic Patch Exposure & Aggregation Propensity (`biophys_guard/sap.py`)**
   - Strictly $SO(3)$-equivariant **590-point Fibonacci Shrake-Rupley SASA** (`probe_radius=1.4 Å`, rotated rigidly via an internal molecular frame for machine-precision SE(3) invariance).
   - **Chennamsetty et al. (PNAS 2009) Spatial Aggregation Propensity (SAP)** at $R = 5.0\text{ \AA}$ and $10.0\text{ \AA}$ using Black & Mould (1991) hydrophobicity centered at `GLY = 0.0` (`[-0.500, +0.500]`) and self-consistent **Ala-X-Ala** extended tripeptide side-chain max SASA (`GLY = 0.0`, $\text{rSASA} \in [0, 1]$).
   - **Coventry / Baker Lab `SapScoreMetric`** (`sap_score = sum(max(0, atom_SAP))`; Cao et al., *Nature* 2022) and unscaled surface hydrophobic proximity (`PSH`).
   - **Strict Backbone-Only Refusal:** Backbone-only (`N, CA, C, O`) or $C_\beta$-stubbed models (`< 80%` sidechain heavy-atom completeness beyond $C_\beta$) are explicitly refused as `INVALID_BACKBONE_ONLY` instead of fabricating virtual $C_\beta$ metrics.

2. **Classical Electrodynamics, Coupled $\text{p}K_a$, and Charge Symmetry (`biophys_guard/electrostatics.py`)**
   - Formal ionizable centers positioned on chemical functional-group atoms (`LYS:NZ`, `ARG:CZ`, `ASP:OD1/OD2`, `GLU:OE1/OE2`, `HIS:ND1/NE2`, `CYS:SG`, `TYR:OH`, `N_TERM:N`, `C_TERM:O/OXT`).
   - Iterative **PROPKA3-inspired coupled $\text{p}K_a$** solver combining smooth local neighbor-density desolvation ($6.0\text{–}15.5\text{ \AA}$), polar H-bond stabilization, and damped Coulombic salt-bridge coupling, plus bisection $\text{pI}$.
   - **SE(3)-Invariant Dipole Moment** ($\vec{\mu}_{\text{CoM}} = 4.803204 \sum_g q_g (\vec{r}_g - \vec{r}_{\text{CoM}})$) relative to the atomic mass-weighted Center of Mass (no artificial $Q_{\text{net}}/N$ subtraction).
   - **Agrawal et al. (mAbs 2016) Spatial Charge Map (`SCM`)** filtered strictly to solvent-exposed side chains ($\text{SASA}_{\text{sc}} \ge 10.0\text{ \AA}^2$), unscaled positive/negative cluster pair integrals, smooth 92-direction $SO(3)$-invariant Charge Symmetry Score (`CSS`), and **OPIG TAP `SFvCSP`** for paired antibody-like chains.

3. **Atomic BUNS, MolProbity Clashes & Epistemic Unit Honesty (`biophys_guard/stability.py`)**
   - **Atomic BUNS (`buns_count`, `buns_atoms`):** Heavy polar donors/acceptors with $\text{SASA}_{1.4\text{ \AA}} < 0.50\text{ \AA}^2$ and water-pocket $\text{SASA}_{1.05\text{ \AA}} < 1.25\text{ \AA}^2$ lacking valid directional H-bonds (disulfide `CYS:SG`, `MET:SD`, and `PRO:N` donor exclusions; crystallographic `ASN/GLN/HIS` flip tolerance).
   - **MolProbity Heavy-Atom Clashes:** Non-bonded overlaps $\ge 0.40\text{ \AA}$ ($\ge 0.65\text{ \AA}$ for valid H-bonds) with verified `C–N <= 1.85 Å` peptide-bond topology exclusions, reported as `clashscore` per 1,000 heavy atoms.
   - **Unit Honesty:** `kcal/mol` (`is_calibrated_free_energy=True`) is emitted **only** by `FoldX`; `ESM-1v` emits dimensionless `log_likelihood_ratio`; built-in static 3D heuristics emit dimensionless `heuristic_au`.

4. **OPIG Traffic-Light Gating & 3D Overlays (`biophys_guard/pipeline.py`, `visualization.py`, `schema.py`)**
   - Empirical per-axis **`GREEN` / `AMBER` / `RED` / `INVALID_BACKBONE_ONLY`** flags plus per-residue `pLDDT < 70` ML rotamer uncertainty warnings.
   - Draft 2020-12 JSON Schema (`schemas/biophys_guard_report.schema.json`), risk-encoded PDB (`[-99.99, +99.99]`), PyMOL `.pml`, NGL.js `.html` (with insertion-code support and offline fallback), and Mol* sidecar specs.

## Usage

```bash
# Run the real RCSB PDB & AlphaFold DB benchmark suite
python -m biophys_guard.cli demo -o outputs

# Analyze a custom PDB or mmCIF structure
python -m biophys_guard.cli analyze examples/7JZU_LCB1_monomer.pdb -o outputs --ph 7.4 --variants A:M13K,A:L31E,A:I12P

# Export JSON Schema
python -m biophys_guard.cli export-schema -o schemas/biophys_guard_report.schema.json

# Run verification and SE(3) invariance tests
python -m unittest discover -s tests -v

# Regenerate the static Distill.pub monograph & 3D workbench bundle (docs/assets/)
python scripts/export_web_monograph_assets.py
python -m http.server 8765 --directory docs
```
