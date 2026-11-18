"""Biophysical reference constants and empirical scales for biophys-guard."""

from __future__ import annotations

from typing import Dict, FrozenSet, Tuple

# Canonical 20 standard amino acids
THREE_TO_ONE: Dict[str, str] = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}

# Common protonation / non-standard aliases mapped to canonical 3-letter codes
RESIDUE_ALIASES: Dict[str, str] = {
    "HID": "HIS",
    "HIE": "HIS",
    "HIP": "HIS",
    "HSD": "HIS",
    "HSE": "HIS",
    "HSP": "HIS",
    "CYX": "CYS",
    "CYM": "CYS",
    "ASH": "ASP",
    "GLH": "GLU",
    "LYN": "LYS",
    "MSE": "MET",
    "SEC": "CYS",
    "PYL": "LYS",
}

ONE_TO_THREE: Dict[str, str] = {v: k for k, v in THREE_TO_ONE.items()}
CANONICAL_RESIDUES = tuple(sorted(THREE_TO_ONE.keys()))
BACKBONE_ATOMS = frozenset({"N", "CA", "C", "O", "OXT", "H", "HA", "HA2", "HA3"})

# Physical constants
DEBYE_PER_E_ANGSTROM = 4.803204

# Atomic masses (Da) for true mass-weighted Center of Mass calculation
ATOMIC_MASSES: Dict[str, float] = {
    "H": 1.008,
    "C": 12.011,
    "N": 14.007,
    "O": 15.999,
    "S": 32.060,
    "P": 30.974,
    "SE": 78.971,
    "F": 18.998,
    "CL": 35.450,
    "BR": 79.904,
    "I": 126.904,
    "FE": 55.845,
    "ZN": 65.380,
    "MG": 24.305,
    "CA": 40.078,
}
DEFAULT_ATOMIC_MASS = 12.011

# Black & Mould (1991) normalized hydrophobicity scale shifted so GLY = 0.000,
# strictly bounded in [-0.500, +0.500] per Chennamsetty et al. (PNAS 2009).
BLACK_MOULD_HYDROPHOBICITY: Dict[str, float] = {
    "PHE": 0.500,
    "LEU": 0.442,
    "ILE": 0.442,
    "TYR": 0.379,
    "TRP": 0.377,
    "VAL": 0.324,
    "MET": 0.237,
    "PRO": 0.210,
    "CYS": 0.179,
    "ALA": 0.115,
    "GLY": 0.000,
    "THR": -0.051,
    "SER": -0.142,
    "LYS": -0.218,
    "GLN": -0.250,
    "ASN": -0.265,
    "HIS": -0.336,
    "GLU": -0.457,
    "ASP": -0.472,
    "ARG": -0.500,
}

# Tien et al. (2013) PLoS ONE 8(11):e80635 theoretical max total SASA (A^2)
TIEN_MAX_TOTAL_SASA: Dict[str, float] = {
    "ALA": 129.0,
    "ARG": 274.0,
    "ASN": 195.0,
    "ASP": 193.0,
    "CYS": 167.0,
    "GLN": 225.0,
    "GLU": 223.0,
    "GLY": 104.0,
    "HIS": 224.0,
    "ILE": 197.0,
    "LEU": 201.0,
    "LYS": 236.0,
    "MET": 224.0,
    "PHE": 240.0,
    "PRO": 159.0,
    "SER": 155.0,
    "THR": 172.0,
    "TRP": 285.0,
    "TYR": 263.0,
    "VAL": 174.0,
}

# Extended Ala-X-Ala tripeptide side-chain max SASA (A^2) computed self-consistently
# using Bondi vdW radii, probe_radius=1.4 A, and 590-pt Shrake-Rupley tessellation
# across extended rotamers (Chennamsetty et al., PNAS 2009 convention).
# GLY has no heavy side-chain atoms beyond CA and contributes 0.0 to SAP.
ALA_X_ALA_SIDECHAIN_MAX_SASA: Dict[str, float] = {
    "ALA": 58.33,
    "ARG": 193.03,
    "ASN": 115.58,
    "ASP": 114.26,
    "CYS": 92.39,
    "GLN": 142.93,
    "GLU": 139.37,
    "GLY": 0.0,
    "HIS": 145.40,
    "ILE": 134.68,
    "LEU": 131.20,
    "LYS": 159.78,
    "MET": 148.69,
    "PHE": 168.25,
    "PRO": 102.14,
    "SER": 77.81,
    "THR": 102.49,
    "TRP": 200.99,
    "TYR": 183.85,
    "VAL": 111.35,
}

# Bondi van der Waals atomic radii (Angstroms)
BONDI_VDW_RADII: Dict[str, float] = {
    "H": 1.20,
    "C": 1.70,
    "N": 1.55,
    "O": 1.52,
    "S": 1.80,
    "P": 1.80,
    "SE": 1.90,
    "F": 1.47,
    "CL": 1.75,
    "BR": 1.85,
    "I": 1.98,
    "FE": 1.40,
    "ZN": 1.39,
    "MG": 1.73,
    "CA": 1.74,
}
DEFAULT_VDW_RADIUS = 1.70

# Ionizable residue intrinsic model pKa values and type ("acid" or "base")
# Pace et al. (2009) / PROPKA3 reference values
IONIZABLE_PKA: Dict[str, Tuple[float, str]] = {
    "ASP": (3.80, "acid"),
    "GLU": (4.50, "acid"),
    "CYS": (9.00, "acid"),
    "TYR": (10.00, "acid"),
    "HIS": (6.50, "base"),
    "LYS": (10.50, "base"),
    "ARG": (12.50, "base"),
}
N_TERM_PKA = 8.00
C_TERM_PKA = 3.20

# Exact chemical charge-bearing atom names for formal ionizable centers
FORMAL_CHARGE_ATOMS_BY_RES: Dict[str, Tuple[str, ...]] = {
    "LYS": ("NZ",),
    "ARG": ("CZ",),
    "ASP": ("OD1", "OD2"),
    "GLU": ("OE1", "OE2"),
    "HIS": ("ND1", "NE2"),
    "CYS": ("SG",),
    "TYR": ("OH",),
}

# Hydrogen-bond chemistry for atomic BUNS and MolProbity clash filtering
# Note: PRO backbone N has no amide hydrogen and is NOT a donor.
SIDECHAIN_HBOND_DONORS: Dict[str, FrozenSet[str]] = {
    "ARG": frozenset({"NE", "NH1", "NH2"}),
    "ASN": frozenset({"ND2"}),
    "CYS": frozenset({"SG"}),
    "GLN": frozenset({"NE2"}),
    "HIS": frozenset({"ND1", "NE2"}),
    "LYS": frozenset({"NZ"}),
    "SER": frozenset({"OG"}),
    "THR": frozenset({"OG1"}),
    "TRP": frozenset({"NE1"}),
    "TYR": frozenset({"OH"}),
}

SIDECHAIN_HBOND_ACCEPTORS: Dict[str, FrozenSet[str]] = {
    "ASN": frozenset({"OD1"}),
    "ASP": frozenset({"OD1", "OD2"}),
    "CYS": frozenset({"SG"}),
    "GLN": frozenset({"OE1"}),
    "GLU": frozenset({"OE1", "OE2"}),
    "HIS": frozenset({"ND1", "NE2"}),
    "MET": frozenset({"SD"}),
    "SER": frozenset({"OG"}),
    "THR": frozenset({"OG1"}),
    "TYR": frozenset({"OH"}),
}

# Side-chain transfer free energy (heuristic structural stability prior, NOT FoldX kcal/mol)
BURIAL_TRANSFER_PRIOR: Dict[str, float] = {
    "ILE": -2.45,
    "LEU": -2.30,
    "VAL": -1.95,
    "PHE": -2.55,
    "TRP": -2.35,
    "MET": -1.65,
    "TYR": -1.30,
    "CYS": -1.25,
    "ALA": -0.65,
    "PRO": -0.35,
    "GLY": 0.00,
    "THR": 0.35,
    "SER": 0.55,
    "HIS": 0.90,
    "ASN": 1.25,
    "GLN": 1.15,
    "GLU": 1.85,
    "ASP": 1.95,
    "LYS": 2.15,
    "ARG": 2.35,
}

# Residue van der Waals volumes (A^3, Harpaz et al. 1994)
RESIDUE_VOLUME: Dict[str, float] = {
    "GLY": 66.4,
    "ALA": 91.5,
    "SER": 99.1,
    "CYS": 105.6,
    "ASP": 124.5,
    "PRO": 129.3,
    "ASN": 135.2,
    "THR": 122.1,
    "GLU": 155.1,
    "VAL": 141.7,
    "GLN": 161.1,
    "HIS": 167.3,
    "MET": 170.8,
    "ILE": 168.8,
    "LEU": 167.9,
    "LYS": 171.3,
    "PHE": 203.4,
    "TYR": 203.6,
    "ARG": 202.0,
    "TRP": 237.6,
}

# Secondary structure conformational propensity priors (Pace & Scholtz 1998; Kim & Berg 1993)
HELIX_PENALTY: Dict[str, float] = {
    "ALA": 0.00,
    "ARG": 0.21,
    "LYS": 0.26,
    "LEU": 0.21,
    "MET": 0.24,
    "GLU": 0.40,
    "GLN": 0.39,
    "TRP": 0.49,
    "PHE": 0.54,
    "TYR": 0.53,
    "ILE": 0.41,
    "VAL": 0.61,
    "HIS": 0.61,
    "ASP": 0.69,
    "ASN": 0.65,
    "CYS": 0.68,
    "THR": 0.66,
    "SER": 0.50,
    "GLY": 1.00,
    "PRO": 3.16,
}

SHEET_PENALTY: Dict[str, float] = {
    "VAL": 0.00,
    "ILE": 0.00,
    "TYR": 0.05,
    "PHE": 0.10,
    "THR": 0.12,
    "TRP": 0.20,
    "CYS": 0.25,
    "LEU": 0.35,
    "ARG": 0.40,
    "MET": 0.42,
    "GLN": 0.48,
    "SER": 0.48,
    "HIS": 0.55,
    "LYS": 0.60,
    "ALA": 0.68,
    "GLU": 0.72,
    "ASN": 0.78,
    "ASP": 0.85,
    "GLY": 1.15,
    "PRO": 2.85,
}

# Literature-calibrated Green / Amber / Red empirical thresholds
# Coventry / Baker Lab (Cao et al., Nature 2022; Bennett et al., Nat Commun 2023)
COVENTRY_SAP_SCORE_GREEN_MAX = 25.0
COVENTRY_SAP_SCORE_AMBER_MAX = 35.0
# Per-residue SAP hot-spot threshold (Chennamsetty et al., PNAS 2009)
CHENNAMSETTY_RESIDUE_SAP_HOTSPOT = 0.15

# OPIG TAP Structural Fv Charge Symmetry Parameter (Raybould et al., PNAS 2019)
OPIG_TAP_SFVCSP_GREEN_MIN = -6.30
OPIG_TAP_SFVCSP_AMBER_MIN = -20.40

# MolProbity clashscore (overlaps >= 0.40 A per 1,000 atoms; Word et al., JMB 1999)
MOLPROBITY_CLASH_GREEN_MAX = 5.0
MOLPROBITY_CLASH_AMBER_MAX = 15.0

# Atomic BUNS thresholds (Coventry & Baker: buried SASA < 0.5 A^2)
ATOMIC_BUNS_SASA_CUTOFF = 0.50
ATOMIC_HBOND_DIST_MAX = 3.50
MOLPROBITY_OVERLAP_CUTOFF = 0.40
