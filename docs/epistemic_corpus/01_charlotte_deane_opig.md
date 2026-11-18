# Primary-Source Epistemic Synthesis: Prof. Charlotte M. Deane (Oxford Protein Informatics Group — OPIG)

## 1. Foundational Literature & Tools
- **Raybould, M. I. J., Marks, C., Krawczyk, K., Taddese, B., Nowak, J., Lewis, A. P., Bujotzek, A., Shi, J., & Deane, C. M. (2019).** *Five computational developability guidelines for therapeutic antibody profiling.* **PNAS**, 116(10), 4025–4030.
- **Raybould, M. I. J., Turnbull, O. M., Suter, A., & Deane, C. M. (2024).** *Contextualising the developability risk of antibodies with lambda light chains using enhanced therapeutic antibody profiling.* **Communications Biology**, 7, 62.
- **Abanades, B., Wong, W. K., Boyles, F., Georges, G., Bujotzek, A., & Deane, C. M. (2023).** *ImmuneBuilder: Deep-learning models for predicting the structures of immune proteins.* **Communications Biology**, 6, 575.
- **Hoerschinger, V. J., Waibl, F., Pomarici, N. D., Loeffler, J. R., Deane, C. M., Georges, G., Kettenberger, H., Fernandez-Quintero, M. L., & Liedl, K. R. (2023).** *PEP-Patch: Electrostatics in Protein–Protein Recognition, Specificity, and Antibody Developability.* **J. Chem. Inf. Model.** (`surface_analyses` / APBS electrostatic surface integrals).

---

## 2. OPIG Developability Philosophy: Empirical Reference Distributions vs. Hand-Tuned Thresholds
Prof. Deane's core methodological principle is **empirical calibration against real experimental/clinical distributions** rather than arbitrary scalar composite percentages or opaque "viability scores":
1. **No Uncalibrated Composite Theater:** Combining heterogeneous physical signals (hydrophobicity, charge, clashes) into an uncalibrated `0–100` "expression viability score" creates false precision. Each biophysical axis must retain its own interpretable, empirically anchored **Green / Amber / Red** traffic-light flag:
   - **Green:** Safely inside the central empirical distribution (e.g., 5th–95th percentile) of well-behaved soluble native or post-Phase-I Clinical-Stage Therapeutic (CST) reference structures.
   - **Amber:** Outer tail / extreme quantiles (typically outside the 5th/95th percentile or 1st–5th / 95th–99th percentile tails of clinical-stage / soluble benchmarks).
   - **Red:** Completely outside observed viable clinical/soluble bounds — high risk of aggregation, viscosity, polyspecificity, or expression failure.

---

## 3. The Five TAP Metrics & Mathematical Formulation (Raybould et al., *PNAS* 2019 & *Comm. Biol.* 2024)
In the **Therapeutic Antibody Profiler (TAP)**, 3D structures are modeled with ABodyBuilder / ImmuneBuilder and five physicochemical descriptors are evaluated across the surface / CDR vicinity:
1. **Total CDR Length ($L_{\text{tot}}$):** Combined residue count across CDRs (`Green < 54`, `Amber 54–60`, `Red > 60`).
2. **Patches of Surface Hydrophobicity (`PSH`):**
   $$\text{PSH} = \sum_{i, j \in \text{surface}} \frac{H_i \cdot H_j}{d_{ij}^2}$$
   (weighted pairwise hydrophobic proximity over surface-exposed residues using Kyte-Doolittle / Black-Mould normalized hydrophobicity and inter-residue distance $d_{ij}$; Amber `83.84–100.71` or `156.20–173.85`, Red `< 83.84` or `> 173.85` for antibody CDR vicinity). Note that both *insufficient* and *excessive* surface hydrophobicity signal anomalies relative to clinical norms, while for general *de novo* miniproteins only **high** exposed hydrophobic patches (`PSH_high` / `SAP`) drive self-association.
3. **Patches of Positive Charge (`PPC`):**
   Sum of localized positive surface charge patch weights over exposed basic clusters ($\text{Arg}, \text{Lys}, \text{His}$; TAP Green $\le 1.25$, Amber $1.25\text{–}3.16$, Red $> 3.16$). High PPC correlates with nonspecific binding, heparin retention, and fast *in vivo* clearance.
4. **Patches of Negative Charge (`PNC`):**
   Sum of localized negative surface charge patch weights over exposed acidic clusters ($\text{Asp}, \text{Glu}$; TAP Green $\le 1.84$, Amber $1.84\text{–}3.50$, Red $> 3.50$). High PNC correlates with self-association and high solution viscosity.
5. **Structural Fv Charge Symmetry Parameter (`SFvCSP`):**
   $$\text{SFvCSP} = q_{\text{chain A}} \times q_{\text{chain B}}$$
   Product of the net surface charges of paired domains/chains ($V_H$ and $V_L$ in antibodies; TAP Green $> -6.30$, Amber $-20.40$ to $-6.30$, Red $< -20.40$). A large negative product signifies strong inter-chain opposite-charge polarity driving dimer/self-association viscosity; for single-chain *de novo* domains, spatial dipolar / hemispheric charge asymmetry measures the intra-domain analog.

---

## 4. Electrostatic Surface Integrals (`PEP-Patch`) & ML Structure Uncertainty (`ImmuneBuilder` / `AF2` / `ESMFold`)
1. **Surface Potential Integrals vs. Point-Charge Heuristics (`PEP-Patch`):**
   - Point-charge metrics (like naive residue charge sums) ignore screening length, burial depth, and contiguous surface topology.
   - `PEP-Patch` segments contiguous positive and negative electrostatic zones on the solvent-accessible / Gaussian molecular surface and integrates patch area and potential magnitude. When a continuum Poisson-Boltzmann solver (APBS) is absent, any distance-weighted spatial charge map (`SCM`) must:
     - Operate at atomic chemical charge centers on solvent-exposed side chains ($\text{SASA}_{\text{sc}} \ge 10\text{ \AA}^2$), NOT residue $C_\alpha$/centroids;
     - Explicitly distinguish continuous surface potential integrals (`APBS`/`PEP-Patch`) from Coulomb-screened/radial atomic charge maps (`SCM`), never pretending a point sum is a Poisson-Boltzmann potential.
2. **Side-Chain Rotamer Uncertainty in AI-Predicted Models (AlphaFold2, ESMFold, ImmuneBuilder):**
   - OPIG's structural modeling benchmarks (*Abanades et al.*, 2023; *Raybould et al.*, 2024) show that unrelaxed ML predictions exhibit side-chain rotamer jitter and local stereochemical clashes.
   - Because surface patch scores (`PSH`, `SAP`, `PPC`, `PNC`, `SCM`) depend sensitively on solvent exposure of flexible surface side chains ($\text{Arg}, \text{Lys}, \text{Trp}, \text{Phe}, \text{Glu}$), **low-pLDDT regions (`pLDDT < 70`)** and **unrelaxed steric clashes** degrade surface patch reliability.
   - Any developability pipeline analyzing an AlphaFold/ESMFold model MUST flag low-pLDDT surface/patch residues (`pLDDT < 70`) with an explicit rotamer-uncertainty warning rather than treating ML coordinates as ground-truth crystal geometry.
