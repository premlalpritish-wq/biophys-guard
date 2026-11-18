# Primary-Source Epistemic Synthesis: Classical Protein Electrodynamics & Thermodynamics

## 1. Foundational Literature
- **Tanford, C., & Kirkwood, J. G. (1957).** *Theory of protein titration curves. I. General equations for impenetrable spheres.* **J. Am. Chem. Soc.**, 79(20), 5333–5339.
- **Olsson, M. H. M., Søndergaard, C. R., Rostkowski, M., & Jensen, J. H. (2011).** *PROPKA3: Consistent treatment of internal and surface residues in empirical $\text{p}K_a$ predictions.* **J. Chem. Theory Comput.**, 7(2), 525–537.
- **Felder, C. E., Prilusky, J., Silman, I., & Sussman, J. L. (2007).** *A server and database for dipole moments of proteins.* **Nucleic Acids Res.**, 35, W512–W521.
- **Word, J. M., Lovell, S. C., LaBean, T. H., Taylor, H. C., Zalis, M. E., Presley, B. K., & Richardson, J. S. (1999).** *Visualizing and quantifying molecular goodness-of-fit: small-probe contact dots with explicit hydrogen atoms.* **J. Mol. Biol.**, 285(4), 1711–1733. (MolProbity clashscore overlap $\ge 0.40\text{ \AA}$).
- **Meier, J., Rao, R., Verkuil, R., Liu, J., Sercu, T., & Rives, A. (2021).** *Language models enable zero-shot prediction of the effects of mutations on protein function.* **NeurIPS**, 34, 29287–29303 (ESM-1v).
- **Schymkowitz, J., Borg, J., Stricher, F., Nys, R., Rousseau, F., & Serrano, L. (2005).** *The FoldX web server: an online force field.* **Nucleic Acids Res.**, 33, W382–W388.

---

## 2. Origin Invariance & SE(3) Symmetry of Macromolecular Dipole Moments
1. **Fundamental Law (Jackson, *Classical Electrodynamics* / Felder et al., *NAR* 2007):**
   The electric dipole moment of a discrete charge distribution $\{q_a, \vec{r}_a\}$ relative to a reference origin $\vec{r}_0$ is:
   $$\vec{\mu}(\vec{r}_0) = 4.803204 \sum_{a} q_a (\vec{r}_a - \vec{r}_0) \quad [\text{Debye}]$$
   where $1\,e\cdot\text{\AA} = 4.803204\text{ D}$.
   - If net charge $Q_{\text{net}} = \sum_a q_a = 0$, $\vec{\mu}$ is origin-independent.
   - If $Q_{\text{net}} \ne 0$ (nearly all real proteins at $\text{pH} \ne \text{pI}$), shifting the origin shifts the dipole by $-Q_{\text{net}} \Delta \vec{r}_0$.
2. **Required Physical Convention (Protein Dipole Moments Server / Felder et al., 2007):**
   - Set $\vec{r}_0 = \vec{r}_{\text{CoM}} = \frac{\sum_a m_a \vec{r}_a}{\sum_a m_a}$, the **atomic mass-weighted Center of Mass** (NOT unweighted residue centroids).
   - **Never subtract $Q_{\text{net}} / N_{\text{res}}$ per residue!** Subtracting a uniform background charge $-Q_{\text{net}}/N$ distorts the physical charge distribution whenever residue density is non-uniform; by defining $\vec{r}_0 = \vec{r}_{\text{CoM}}$ directly on the true charges $q_a$, $\vec{\mu}_{\text{CoM}}$ is **strictly SE(3)-invariant** (invariant under any 3D rigid translation $\vec{r}_a \mapsto \vec{r}_a + \vec{t}$ and preserves vector magnitude $\|\vec{\mu}_{\text{CoM}}\|$ under any 3D rotation $R \in SO(3)$).
3. **Chemical Charge Centers (Never Assign Charge to Side-Chain Geometric Centroid):**
   Formal ionizable charges physically reside on the terminal titratable functional-group atoms:
   - `LYS`: `NZ` (fallback `CE` / `CB`)
   - `ARG`: `CZ` (center of guanidinium resonance; fallback `NE`/`NH1`/`NH2`)
   - `ASP`: `CG` (center of carboxylate `OD1`/`OD2` resonance; or split `0.5` on `OD1`, `0.5` on `OD2`)
   - `GLU`: `CD` (center of carboxylate `OE1`/`OE2` resonance; or split `0.5` on `OE1`, `0.5` on `OE2`)
   - `HIS`: midpoint of `ND1` and `NE2` (or `0.5` each)
   - `CYS` (free thiol, not disulfide-bonded): `SG`
   - `TYR`: `OH`
   - `N_TERM`: backbone `N` of the chain's N-terminal residue
   - `C_TERM`: backbone `C` / `OXT` of the chain's C-terminal residue

---

## 3. Coupled Tanford-Kirkwood / PROPKA3 $\text{p}K_a$ Perturbation Physics
In **Olsson et al. (*JCTC* 2011, PROPKA3)**, shifting an ionizable group $i$ into a protein interior involves competing forces:
$$\text{p}K_{a, i} = \text{p}K_{a,\text{model}} + \Delta\text{p}K_{a,\text{desolv}} + \Delta\text{p}K_{a,\text{HB}} + \Delta\text{p}K_{a,\text{Coulomb}}$$
1. **Burial Desolvation ($\Delta\text{p}K_{a,\text{desolv}}$):** Removing a charge from high-dielectric water ($\varepsilon_w \approx 78.4$) into the low-dielectric protein interior ($\varepsilon_p \approx 4\text{–}10$) favors the **neutral** species:
   - **Acids** (`ASP`, `GLU`, `CYS`, `TYR`, `C_TERM`): $\Delta\text{p}K_{a,\text{desolv}} > 0$ (harder to deprotonate).
   - **Bases** (`HIS`, `LYS`, `ARG`, `N_TERM`): $\Delta\text{p}K_{a,\text{desolv}} < 0$ (harder to protonate).
   - Modeled smoothly from local heavy-atom neighbor density $N_{\text{local}}$ (within $6\text{–}15.5\text{ \AA}$, rather than hard step cutoffs) or continuous burial fraction.
2. **Coulombic & Salt-Bridge Coupling ($\Delta\text{p}K_{a,\text{Coulomb}}$) + H-Bonds ($\Delta\text{p}K_{a,\text{HB}}$):**
   - A nearby oppositely charged residue (salt bridge, $d \le 4.5\text{ \AA}$) or stabilizing hydrogen bond stabilizes the **ionized** state of both partners:
     - For an **acid** near a positive group/H-bond donor: $\Delta\text{p}K_{a,\text{Coulomb}} < 0$ (lowers $\text{p}K_a$, making it easier to stay deprotonated/negative).
     - For a **base** near a negative group/H-bond acceptor: $\Delta\text{p}K_{a,\text{Coulomb}} > 0$ (raises $\text{p}K_a$, making it easier to stay protonated/positive).
   - Conversely, same-sign nearby charges raise acid $\text{p}K_a$ and lower base $\text{p}K_a$ (repulsive shift).
   - Omitting $\Delta\text{p}K_{a,\text{Coulomb}}$ and $\Delta\text{p}K_{a,\text{HB}}$ while applying only $\Delta\text{p}K_{a,\text{desolv}}$ causes buried salt-bridged residues to falsely appear neutral!

---

## 4. Epistemic Honesty in Thermodynamic Stability ($\Delta G$, $\Delta\Delta G$) & Steric Validation
1. **Units Must Never Be Falsified:**
   - **FoldX (`BuildModel` / `Stability`, Schymkowitz et al., *NAR* 2005)** performs all-atom rotamer repacking (`RepairPDB` + side-chain optimization) with a calibrated empirical force field and reports $\Delta G$ and $\Delta\Delta G$ in **`kcal/mol`**.
   - **ESM-1v (Meier et al., *NeurIPS* 2021)** evaluates masked marginal evolutionary log-likelihood ratios:
     $$\text{LLR}_i = \log P(x_i = \text{mut} \mid x_{\setminus i}) - \log P(x_i = \text{wt} \mid x_{\setminus i})$$
     which is a **dimensionless log-likelihood ratio (`log_likelihood_ratio`)**, NOT a physical `kcal/mol` Gibbs free energy (never multiply by an arbitrary $RT_{\text{eff}}$ factor and label it `kcal/mol`).
   - **Static Single-Structure Heuristics (No Rotamer Repacking):** Without full 3D side-chain rotamer packing and solvent calibration, an analytical/empirical descriptor (transfer energy + volume cavity + secondary structure propensity) is an **uncalibrated heuristic stability index (`heuristic_au`)**. Labeling uncalibrated heuristic point-mutation numbers as `kcal/mol` without 3D mutant side-chain repacking violates biophysical rigor.
2. **All-Atom Steric Clash Definition (MolProbity / Word et al., 1999):**
   - A non-bonded steric clash occurs when two non-bonded atoms (excluding 1-2 bonded, 1-3 angle, and 1-4 torsional neighbors, as well as valid hydrogen-bond donor/acceptor pairs) overlap by **$\ge 0.40\text{ \AA}$**, i.e., $d_{ab} \le (r_{\text{vdW}, a} + r_{\text{vdW}, b}) - 0.40\text{ \AA}$ (or heavy-atom proxy accounting for implicit H where H atoms are absent, reported cleanly as `clashscore = clashes / (n_atoms / 1000)`).
