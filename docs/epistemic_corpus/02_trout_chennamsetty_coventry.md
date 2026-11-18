# Primary-Source Epistemic Synthesis: Prof. Bernhardt L. Trout & Dr. Naresh Chennamsetty (MIT) + Dr. Brian Coventry (Baker Lab / UW IPD)

## 1. Foundational Literature
- **Chennamsetty, N., Voynov, V., Kayser, V., Helk, B., & Trout, B. L. (2009).** *Design of therapeutic proteins with enhanced stability.* **PNAS**, 106(29), 11937–11942.
- **Chennamsetty, N., Helk, B., Voynov, V., Kayser, V., & Trout, B. L. (2010).** *Aggregation-prone motifs in human immunoglobulin G.* **J. Mol. Biol.**, 391(2), 404–413.
- **Agrawal, N. J., Helk, B., Kumar, S., Mody, N., Sathish, H. A., Samra, H. S., Buck, P. M., Li, L., & Trout, B. L. (2016).** *Computational tool for the early screening of monoclonal antibodies for their viscosities.* **mAbs**, 8(1), 43–48.
- **Tien, M. Z., Meyer, A. G., Sydykova, D. K., Spielman, S. J., & Wilke, C. O. (2013).** *Maximum allowed solvent accessibilities of residues in proteins.* **PLOS ONE**, 8(11), e80635.
- **Cao, L., Coventry, B., Goreshnik, I., Huang, B., Sheffler, W., Park, J. S., Jude, K. M., Marković, I., Kadam, R. U., Verschueren, K. H. G., Verstraete, K., Walsh, S. T. R., Bennett, N., Phal, A., Yang, A., Kozodoy, L., DeWitt, M., Picton, L., Miller, L., Strauch, E.-M., DeBouver, N. D., Pires, A., Bera, A. K., Halabiya, S., Hammerson, B., Yang, W., Moniz, S., Stewart, L., Wilson, I. A., Ruohola-Baker, H., Schlessinger, J., Lee, S., Savvides, S. N., Garcia, K. C., & Baker, D. (2022).** *Design of protein-binding proteins from the target structure alone.* **Nature**, 605, 551–560.
- **Cao, L., Goreshnik, I., Coventry, B., Case, J. B., Miller, L., Kozodoy, L., Chen, R. E., Carter, L., Walls, A. C., Park, Y.-J., Strauch, E.-M., Stewart, L., Diamond, M. S., Veesler, D., & Baker, D. (2020).** *De novo design of picomolar SARS-CoV-2 miniprotein inhibitors.* **Science**, 370(6515), 426–431. (PDB `7JZL` / `LCB1`, `7JZM` / `LCB3`).
- **Watson, J. L., Juergens, D., Bennett, N. R., Trippe, B. L., Yim, J., Eisenach, H. E., Ahern, W., Borst, A. J., Ragotte, R. J., Milles, L. F., Wicky, B. I. M., Hanikel, N., Pellock, S. J., Courbet, A., Sheffler, W., Wang, J., Venkatesh, P., Sappington, I., Torres, S. V., Lauko, A., Bortoli, V. D., Mathieu, E., Ovchinnikov, S., Barzilay, R., Jaakkola, T. S., DiMaio, F., Baek, M., & Baker, D. (2023).** *De novo design of protein structure and function with RFdiffusion.* **Nature**, 620, 1089–1100.

---

## 2. Exact Mathematical Formulation of Spatial Aggregation Propensity (SAP)
In **Chennamsetty et al. (*PNAS* 2009)**, the SAP score of atom $i$ within spatial cutoff radius $R$ (typically $R = 5.0\text{ \AA}$ for local hotspots and $10.0\text{ \AA}$ for macroscopic patches) is strictly defined over **side-chain atoms** as:

$$\text{SAP}_{\text{atom } i}(R) = \sum_{\substack{j \in \text{residues with } \ge 1 \\ \text{side-chain atom within } R \text{ of } i}} \left( \frac{\text{SAA}_{\text{sc}, j \in B(i, R)}}{\text{SAA}_{\text{sc, max}}(j)} \times H_{\text{Black-Mould}}(j) \right)$$

Key exact specifications from Chennamsetty et al. (2009):
1. **Numerator:** $\text{SAA}_{\text{sc}, j \in B(i, R)}$ is the sum of Shrake-Rupley solvent-accessible surface areas of **side-chain heavy atoms** of residue $j$ whose centers lie inside the sphere of radius $R$ centered at atom $i$.
2. **Denominator:** $\text{SAA}_{\text{sc, max}}(j)$ is the total side-chain SASA of amino acid $j$ in a fully extended tripeptide reference conformation (**Ala-X-Ala** in Chennamsetty et al.; **Gly-X-Gly** theoretical maxima in Tien et al., 2013).
   - **Glycine handling:** Glycine has **no side-chain heavy atoms** beyond $C_\alpha$ and has $H_{\text{Black-Mould}}(\text{GLY}) = 0.000$. Therefore Glycine contributes identically `0.0` to SAP (do NOT assign a spurious non-zero denominator/backbone fallback to Glycine).
3. **Black & Mould (1991) Hydrophobicity Scale Centered at Gly = 0.0:**
   Raw Black & Mould (1991) HPLC retention values span `[0.000 (Arg), 1.000 (Phe)]` with `Gly = 0.501`. Linearly shifting and normalizing relative to Glycine yields `[-0.500, +0.500]` where positive values are hydrophobic (`PHE: +0.500`, `LEU: +0.442`, `ILE: +0.442`, `TYR: +0.379`, `TRP: +0.377`, `VAL: +0.324`, `MET: +0.237`, `PRO: +0.210`, `CYS: +0.179`, `ALA: +0.115`), `GLY: 0.000`, and negative values are hydrophilic (`THR: -0.051`, `SER: -0.142`, `LYS: -0.218`, `GLN: -0.250`, `ASN: -0.265`, `HIS: -0.336`, `GLU: -0.457`, `ASP: -0.472`, `ARG: -0.500`).
4. **Residue SAP Aggregation:**
   $$\text{SAP}_{\text{res } k} = \frac{1}{|A_k|} \sum_{i \in A_k} \text{SAP}_{\text{atom } i}$$
   (averaged over all atoms $A_k$ of residue $k$, or side-chain + $C_\alpha$ atoms; note that Chennamsetty et al. average over MD frames or evaluate on the static 3D structure).

---

## 3. Exact Mathematical Formulation of Spatial Charge Map (SCM)
In **Agrawal et al. (*mAbs* 2016)**, high-concentration solution viscosity is driven by localized exposed electrostatic patches (especially exposed negative patches on Fv interacting with positive Fc or self-associating):
1. **Exposure Filter:** Only residues with side-chain SASA $\ge 10\text{ \AA}^2$ contribute exposed charges.
2. **Atomic SCM Score ($R = 10.0\text{ \AA}$):**
   $$\text{SCM}_{\text{atom } i} = \sum_{\substack{j \in \text{side-chain atoms of} \\ \text{exposed residues, } d_{ij} \le 10\text{ \AA}}} q_j$$
   where $q_j$ are formal/partial atomic charges at the target pH (no arbitrary `rSASA` rescaling inside the sum if already filtered by $\text{SASA}_{\text{sc}} \ge 10\text{ \AA}^2$, or evaluated at the true functional group charge-bearing atoms).
3. **Global Negative/Positive Patch Sums:**
   $$\text{SCM}_{\text{neg}} = \sum_{i \in \text{exposed}} \min(0, \text{SCM}_{\text{atom } i}), \qquad \text{SCM}_{\text{pos}} = \sum_{i \in \text{exposed}} \max(0, \text{SCM}_{\text{atom } i})$$
   Large $|\text{SCM}_{\text{neg}}|$ indicates clustered negative surface charge predicting elevated viscosity / self-association.

---

## 4. Brian Coventry's De Novo Binder Filtering Rules (Cao et al. 2022, Watson et al. 2023)
1. **Whole-Protein `sap_score` (`SapScoreMetric` in Rosetta):**
   - Coventry's `SapScoreMetric` computes the sum of positive residue/atom SAP contributions across the target chain/monomer:
     $$\text{SAP}_{\text{score}} = \sum_{k : \text{SAP}_k > 0} \text{SAP}_k$$
   - In *de novo* miniprotein binder campaigns (Cao et al., 2022; Bennett et al., 2023), monomer developability filtering typically requires **`sap_score < 35`** (with clean monomers `< 25`), alongside evaluating `sap_score` both on the **unbound binder monomer** (where the hydrophobic target-binding interface can expose a hotspot patch) and **bound complex** (where binding buries the interface patch).
2. **Buried Unsatisfied Polar Atoms (`BUNS`):**
   - A heavy polar atom (backbone `N`, `O`, or polar side-chain donor/acceptor `N`, `O`, `S` on Ser/Thr/Tyr/Asn/Gln/His/Lys/Arg/Asp/Glu/Trp/Cys) is **buried** if its atomic SASA is near zero ($\text{SASA}_{\text{atom}} < 0.5\text{ \AA}^2$ with probe radius $1.4\text{ \AA}$).
   - It is **unsatisfied** if it forms zero geometrically valid hydrogen bonds (donor–acceptor heavy-atom distance $\le 3.5\text{ \AA}$ and directional donor–H···acceptor / C=O···donor geometry where available; proline backbone `N` has no amide H and is not a donor).
   - Note that counting **per-residue** `unsat_polar` and conflating `SASA >= 0.22` with buried atomic SASA is physically sloppy; BUNS MUST be tracked at **atomic resolution** (`SASA_atom < 0.5 A^2` without H-bond partner) and reported explicitly as `buns_count` and `buns_atoms`.
3. **CRITICAL: Why Backbone-Only Structures Cannot Yield Valid SAP, SASA, or BUNS:**
   - Raw **RFdiffusion** outputs are backbone-only ($N, C_\alpha, C, O$ or poly-Gly coordinates) **prior to ProteinMPNN sequence design and all-atom rotamer packing**.
   - **You CANNOT reconstruct virtual $C_\beta$ atoms onto a poly-Gly or backbone-only PDB and pretend to calculate SAP, side-chain SASA, charge symmetry, or BUNS!** Side-chain surface exposure ($\text{SASA}_{\text{sc}}$), hydrophobic cluster contacts, electrostatic functional-group charge positions, and buried polar satisfaction depend entirely on all-atom side-chain rotamer geometry.
   - Therefore, any pipeline encountering a backbone-only design ($> 20\%$ of non-Gly residues missing side-chain heavy atoms beyond $C_\beta$, or $100\%$ poly-Gly RFdiffusion scaffold) MUST **refuse/gate** all-atom side-chain metrics (`is_backbone_only = True`, `sap_calculable = False`, `status = "INVALID_BACKBONE_ONLY"`) and direct the user to run sequence design + side-chain packing (`ProteinMPNN` + `FastRelax`/`AlphaFold2`) first.
