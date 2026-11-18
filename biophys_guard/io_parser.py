"""PDB and mmCIF structure parser, strict sidechain-completeness validator, and risk PDB writer."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import shlex
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np

from .constants import (
    ATOMIC_MASSES,
    BACKBONE_ATOMS,
    BONDI_VDW_RADII,
    DEFAULT_ATOMIC_MASS,
    DEFAULT_VDW_RADIUS,
    RESIDUE_ALIASES,
    THREE_TO_ONE,
)


@dataclass
class AtomRecord:
    """Single parsed heavy atom from PDB or mmCIF."""

    serial: int
    name: str
    resname: str
    chain_id: str
    auth_seq_id: int
    ins_code: str
    coord: np.ndarray
    occupancy: float
    b_factor: float
    element: str
    mass: float
    is_hetatm: bool = False


@dataclass
class ResidueData:
    """Container for one polymer residue and its constituent heavy atoms."""

    index: int
    chain_id: str
    auth_seq_id: int
    ins_code: str
    resname: str
    aa1: str
    atoms: List[AtomRecord] = field(default_factory=list)
    atom_indices: List[int] = field(default_factory=list)
    backbone_atom_indices: List[int] = field(default_factory=list)
    sidechain_atom_indices: List[int] = field(default_factory=list)
    ca_coord: Optional[np.ndarray] = None
    cb_coord: Optional[np.ndarray] = None
    n_coord: Optional[np.ndarray] = None
    c_coord: Optional[np.ndarray] = None
    o_coord: Optional[np.ndarray] = None
    oxt_coord: Optional[np.ndarray] = None
    sc_centroid: Optional[np.ndarray] = None
    mean_b_factor: float = 0.0
    has_full_sidechain: bool = False

    @property
    def key(self) -> Tuple[str, int, str]:
        """Unique residue key preserving insertion code."""
        return (self.chain_id, self.auth_seq_id, self.ins_code)

    @property
    def seq_tag(self) -> str:
        """Author sequence identifier including insertion code (e.g. '100A')."""
        return f"{self.auth_seq_id}{self.ins_code}".strip()

    @property
    def label(self) -> str:
        """Human-readable identifier such as A:ILE19 or A:TRP100A."""
        return f"{self.chain_id}:{self.resname}{self.seq_tag}"

    @property
    def short_label(self) -> str:
        """Compact identifier such as A:I19 or A:W100A."""
        return f"{self.chain_id}:{self.aa1}{self.seq_tag}"


@dataclass
class StructureData:
    """Parsed biomolecular structure ready for biophysical calculation."""

    input_path: Path
    structure_format: Literal["pdb", "mmcif"]
    detected_model_source: Literal["AlphaFold", "ESMFold", "ImmuneBuilder", "RFdiffusion", "Experimental", "Unknown"]
    residues: List[ResidueData]
    atoms: List[AtomRecord]
    coords: np.ndarray
    vdw_radii: np.ndarray
    atom_masses: np.ndarray
    center_of_mass: np.ndarray
    atom_residue_idx: np.ndarray
    atom_is_sidechain: np.ndarray
    chain_ids: List[str]
    sequence_by_chain: Dict[str, str]
    plddt_detected: bool
    is_backbone_only: bool
    sidechain_completeness: float
    raw_headers: List[str] = field(default_factory=list)


def _normalize_resname(resname: str) -> str:
    clean = resname.strip().upper()
    return RESIDUE_ALIASES.get(clean, clean)


def _infer_element(atom_name: str, raw_element: str) -> str:
    elem = raw_element.strip().upper()
    if elem and elem not in {"?", "."}:
        cleaned = "".join(ch for ch in elem if ch.isalpha())
        if cleaned in BONDI_VDW_RADII:
            return cleaned
        if len(cleaned) >= 1 and cleaned[0] in BONDI_VDW_RADII:
            return cleaned[0]
    stripped = atom_name.strip().upper()
    if stripped.startswith("SE"):
        return "SE"
    for ch in stripped:
        if ch.isalpha():
            return ch
    return "C"


def _detect_model_source(
    headers: List[str],
    b_factors: np.ndarray,
    is_backbone_only: bool,
) -> Tuple[Literal["AlphaFold", "ESMFold", "ImmuneBuilder", "RFdiffusion", "Experimental", "Unknown"], bool]:
    """Identify upstream generator and genuine pLDDT confidence provenance."""
    header_blob = "\n".join(headers).upper()

    # Check explicit experimental methods before ML model heuristics
    if (
        "EXPDTA" in header_blob
        or "_EXPTL." in header_blob
        or "X-RAY DIFFRACTION" in header_blob
        or "ELECTRON MICROSCOPY" in header_blob
        or "SOLUTION NMR" in header_blob
        or "CRYO-EM" in header_blob
    ):
        return "Experimental", False

    if "RFDIFFUSION" in header_blob:
        return "RFdiffusion", False
    if "IMMUNEBUILDER" in header_blob or "ABODYBUILDER" in header_blob:
        is_plddt = bool(len(b_factors) > 0 and 0.0 <= np.min(b_factors) and np.max(b_factors) <= 100.0)
        return "ImmuneBuilder", is_plddt
    if "ESMFOLD" in header_blob or "ESM-" in header_blob:
        is_plddt = bool(len(b_factors) > 0 and 0.0 <= np.min(b_factors) and np.max(b_factors) <= 100.0)
        return "ESMFold", is_plddt
    if (
        "ALPHAFOLD" in header_blob
        or "COLABFOLD" in header_blob
        or "_MA_QA_METRIC" in header_blob
        or "LOCAL_METRIC_VALUES" in header_blob
    ):
        is_plddt = bool(len(b_factors) > 0 and 0.0 <= np.min(b_factors) and np.max(b_factors) <= 100.0)
        return "AlphaFold", is_plddt

    if is_backbone_only:
        return "RFdiffusion", False

    # ESMFold normalized [0, 1] confidence scale without explicit header
    if len(b_factors) > 0:
        b_min, b_max = float(np.min(b_factors)), float(np.max(b_factors))
        b_mean = float(np.mean(b_factors))
        if 0.0 <= b_min and b_max <= 1.0 and b_mean > 0.45:
            return "ESMFold", True

    # Never misclassify headerless X-ray/cryo-EM structures whose B-factors sit in [20, 100] as AlphaFold
    return "Unknown", False


def _parse_pdb_lines(lines: List[str]) -> Tuple[List[AtomRecord], List[str]]:
    atoms: List[AtomRecord] = []
    headers: List[str] = []
    seen_model = False

    for line in lines:
        rec = line[:6].strip()
        if rec in {"HEADER", "TITLE", "COMPND", "REMARK", "EXPDTA", "MODEL"}:
            headers.append(line.rstrip())
            if rec == "MODEL":
                if seen_model:
                    break
                seen_model = True
            continue
        if rec == "ENDMDL":
            break
        if rec not in {"ATOM", "HETATM"}:
            continue

        alt_loc = line[16:17] if len(line) > 16 else " "
        if alt_loc not in {" ", "A", "1"}:
            continue

        raw_resname = line[17:20].strip()
        resname = _normalize_resname(raw_resname)
        if resname in {"HOH", "WAT", "DOD"} or resname not in THREE_TO_ONE:
            continue

        try:
            serial = int(line[6:11].strip() or len(atoms) + 1)
            atom_name = line[12:16].strip()
            chain_id = (line[21:22].strip() if len(line) > 21 else "") or "A"
            auth_seq_id = int(line[22:26].strip())
            ins_code = line[26:27].strip() if len(line) > 26 else ""
            x = float(line[30:38].strip())
            y = float(line[38:46].strip())
            z = float(line[46:54].strip())
            occupancy = float(line[54:60].strip()) if len(line) >= 60 and line[54:60].strip() else 1.0
            b_factor = float(line[60:66].strip()) if len(line) >= 66 and line[60:66].strip() else 0.0
            raw_elem = line[76:78].strip() if len(line) >= 78 else ""
        except ValueError:
            continue

        element = _infer_element(atom_name, raw_elem)
        if element == "H":
            continue

        atoms.append(
            AtomRecord(
                serial=serial,
                name=atom_name,
                resname=resname,
                chain_id=chain_id,
                auth_seq_id=auth_seq_id,
                ins_code=ins_code,
                coord=np.array([x, y, z], dtype=np.float64),
                occupancy=occupancy,
                b_factor=b_factor,
                element=element,
                mass=ATOMIC_MASSES.get(element, DEFAULT_ATOMIC_MASS),
                is_hetatm=(rec == "HETATM"),
            )
        )

    return atoms, headers


def _tokenize_cif_row(line: str) -> List[str]:
    """Tokenize a single mmCIF data line honoring quoted strings."""
    lexer = shlex.shlex(line, posix=True)
    lexer.quotes = "'\""
    lexer.whitespace_split = True
    lexer.commenters = ""
    return list(lexer)


def _parse_mmcif_lines(lines: List[str]) -> Tuple[List[AtomRecord], List[str]]:
    atoms: List[AtomRecord] = []
    headers: List[str] = []

    i = 0
    n = len(lines)
    while i < n:
        stripped = lines[i].strip()
        if (
            stripped.startswith("#")
            or stripped.startswith("_entry.")
            or stripped.startswith("_software.")
            or stripped.startswith("_exptl.")
            or stripped.startswith("_ma_qa_metric")
            or stripped.startswith("_struct.")
        ):
            headers.append(stripped)

        if stripped == "loop_":
            col_names: List[str] = []
            j = i + 1
            while j < n and lines[j].strip().startswith("_"):
                col_names.append(lines[j].strip())
                j += 1

            if col_names and col_names[0].startswith("_atom_site."):
                col_idx = {name.split(".", 1)[1]: idx for idx, name in enumerate(col_names)}
                row_tokens: List[str] = []
                while j < n:
                    row_line = lines[j].strip()
                    if not row_line or row_line.startswith("#") or row_line == "loop_" or row_line.startswith("_"):
                        break
                    row_tokens.extend(_tokenize_cif_row(row_line))
                    if len(row_tokens) >= len(col_names):
                        row = row_tokens[: len(col_names)]
                        row_tokens = row_tokens[len(col_names) :]
                        atom = _atom_from_cif_row(row, col_idx, len(atoms) + 1)
                        if atom is not None:
                            atoms.append(atom)
                    j += 1
                i = j
                continue
        i += 1

    return atoms, headers


def _atom_from_cif_row(
    row: List[str], col_idx: Dict[str, int], fallback_serial: int
) -> Optional[AtomRecord]:
    def get(col: str, default: str = "") -> str:
        idx = col_idx.get(col)
        return row[idx] if idx is not None and idx < len(row) else default

    model_num = get("pdbx_PDB_model_num", "1")
    if model_num not in {"1", ".", "?"}:
        return None

    alt_loc = get("label_alt_id", ".")
    if alt_loc not in {".", "?", "A", "1"}:
        return None

    group_pdb = get("group_PDB", "ATOM").upper()
    raw_resname = get("auth_comp_id") or get("label_comp_id")
    resname = _normalize_resname(raw_resname)
    if resname in {"HOH", "WAT", "DOD"} or resname not in THREE_TO_ONE:
        return None

    atom_name = get("auth_atom_id") or get("label_atom_id")
    chain_id = get("auth_asym_id") or get("label_asym_id") or "A"
    if chain_id in {".", "?"}:
        chain_id = "A"

    seq_str = get("auth_seq_id")
    if seq_str in {"", ".", "?"}:
        seq_str = get("label_seq_id", "1")
    ins_code = get("pdbx_PDB_ins_code", "")
    if ins_code in {".", "?"}:
        ins_code = ""

    raw_elem = get("type_symbol", "")
    element = _infer_element(atom_name, raw_elem)
    if element == "H":
        return None

    try:
        serial_str = get("id", str(fallback_serial))
        serial = int(serial_str) if serial_str not in {".", "?"} else fallback_serial
        auth_seq_id = int(seq_str)
        x = float(get("Cartn_x"))
        y = float(get("Cartn_y"))
        z = float(get("Cartn_z"))
        occ_str = get("occupancy", "1.0")
        occupancy = float(occ_str) if occ_str not in {".", "?"} else 1.0
        b_str = get("B_iso_or_equiv", "0.0")
        b_factor = float(b_str) if b_str not in {".", "?"} else 0.0
    except ValueError:
        return None

    return AtomRecord(
        serial=serial,
        name=atom_name,
        resname=resname,
        chain_id=chain_id,
        auth_seq_id=auth_seq_id,
        ins_code=ins_code,
        coord=np.array([x, y, z], dtype=np.float64),
        occupancy=occupancy,
        b_factor=b_factor,
        element=element,
        mass=ATOMIC_MASSES.get(element, DEFAULT_ATOMIC_MASS),
        is_hetatm=(group_pdb == "HETATM"),
    )


def load_structure_file(file_path: str | Path) -> StructureData:
    """Parse a .pdb or .cif/.mmcif structure without fabricating missing sidechains."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Structure file not found: {path}")

    suffix = path.suffix.lower()
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()

    if suffix in {".cif", ".mmcif"}:
        structure_format: Literal["pdb", "mmcif"] = "mmcif"
        raw_atoms, headers = _parse_mmcif_lines(lines)
    else:
        structure_format = "pdb"
        raw_atoms, headers = _parse_pdb_lines(lines)

    if not raw_atoms:
        raise ValueError(f"No canonical polymer heavy atoms parsed from {path}")

    residues: List[ResidueData] = []
    res_map: Dict[Tuple[str, int, str], ResidueData] = {}

    for atom in raw_atoms:
        key = (atom.chain_id, atom.auth_seq_id, atom.ins_code)
        if key not in res_map:
            res = ResidueData(
                index=len(residues),
                chain_id=atom.chain_id,
                auth_seq_id=atom.auth_seq_id,
                ins_code=atom.ins_code,
                resname=atom.resname,
                aa1=THREE_TO_ONE[atom.resname],
            )
            res_map[key] = res
            residues.append(res)
        res_map[key].atoms.append(atom)

    all_atoms: List[AtomRecord] = []
    non_gly_count = 0
    non_gly_complete_count = 0

    for res in residues:
        atom_by_name = {a.name: a for a in res.atoms}
        if "N" in atom_by_name:
            res.n_coord = atom_by_name["N"].coord
        if "CA" in atom_by_name:
            res.ca_coord = atom_by_name["CA"].coord
        if "C" in atom_by_name:
            res.c_coord = atom_by_name["C"].coord
        if "O" in atom_by_name:
            res.o_coord = atom_by_name["O"].coord
        if "OXT" in atom_by_name:
            res.oxt_coord = atom_by_name["OXT"].coord

        sc_atoms = [a for a in res.atoms if a.name not in BACKBONE_ATOMS]
        if res.resname == "GLY":
            res.has_full_sidechain = True
        elif res.resname == "ALA":
            res.has_full_sidechain = "CB" in atom_by_name
        else:
            # Must possess real sidechain heavy atoms beyond CB (catches both N/CA/C/O and CB-stubbed models)
            res.has_full_sidechain = any(a.name != "CB" for a in sc_atoms)

        if res.resname != "GLY":
            non_gly_count += 1
            if res.has_full_sidechain:
                non_gly_complete_count += 1

        if "CB" in atom_by_name:
            res.cb_coord = atom_by_name["CB"].coord
        elif res.ca_coord is not None:
            res.cb_coord = res.ca_coord
        else:
            res.cb_coord = np.mean([a.coord for a in res.atoms], axis=0)

        if sc_atoms:
            res.sc_centroid = np.mean([a.coord for a in sc_atoms], axis=0)
        else:
            res.sc_centroid = res.cb_coord

        b_vals = [a.b_factor for a in res.atoms]
        res.mean_b_factor = float(np.mean(b_vals)) if b_vals else 0.0

        for atom in res.atoms:
            atom_idx = len(all_atoms)
            all_atoms.append(atom)
            res.atom_indices.append(atom_idx)
            if atom.name in BACKBONE_ATOMS:
                res.backbone_atom_indices.append(atom_idx)
            else:
                res.sidechain_atom_indices.append(atom_idx)

    coords = np.vstack([a.coord for a in all_atoms]).astype(np.float64)
    vdw_radii = np.array(
        [BONDI_VDW_RADII.get(a.element, DEFAULT_VDW_RADIUS) for a in all_atoms],
        dtype=np.float64,
    )
    atom_masses = np.array([a.mass for a in all_atoms], dtype=np.float64)
    center_of_mass = np.sum(coords * atom_masses[:, None], axis=0) / np.sum(atom_masses)

    atom_residue_idx = np.empty(len(all_atoms), dtype=np.int32)
    atom_is_sidechain = np.zeros(len(all_atoms), dtype=bool)
    for res in residues:
        for idx in res.atom_indices:
            atom_residue_idx[idx] = res.index
        for idx in res.sidechain_atom_indices:
            atom_is_sidechain[idx] = True

    chain_ids: List[str] = []
    sequence_by_chain: Dict[str, str] = {}
    for res in residues:
        if res.chain_id not in sequence_by_chain:
            chain_ids.append(res.chain_id)
            sequence_by_chain[res.chain_id] = ""
        sequence_by_chain[res.chain_id] += res.aa1

    all_gly = len(residues) > 0 and all(r.resname == "GLY" for r in residues)
    sidechain_completeness = (non_gly_complete_count / non_gly_count) if non_gly_count > 0 else 0.0
    is_backbone_only = all_gly or (sidechain_completeness < 0.80)

    raw_b = np.array([r.mean_b_factor for r in residues], dtype=np.float64)
    model_source, plddt_detected = _detect_model_source(headers, raw_b, is_backbone_only)

    if plddt_detected and len(raw_b) > 0 and np.max(raw_b) <= 1.0:
        for res in residues:
            res.mean_b_factor *= 100.0
        for atom in all_atoms:
            atom.b_factor *= 100.0

    return StructureData(
        input_path=path,
        structure_format=structure_format,
        detected_model_source=model_source,
        residues=residues,
        atoms=all_atoms,
        coords=coords,
        vdw_radii=vdw_radii,
        atom_masses=atom_masses,
        center_of_mass=center_of_mass,
        atom_residue_idx=atom_residue_idx,
        atom_is_sidechain=atom_is_sidechain,
        chain_ids=chain_ids,
        sequence_by_chain=sequence_by_chain,
        plddt_detected=plddt_detected,
        is_backbone_only=is_backbone_only,
        sidechain_completeness=round(float(sidechain_completeness), 4),
        raw_headers=headers,
    )


def compute_intrinsic_molecular_frame(coords: np.ndarray, origin: np.ndarray) -> np.ndarray:
    """Build a strictly SO(3)-equivariant right-handed orthonormal frame Q from internal coordinates."""
    n = len(coords)
    if n < 3:
        return np.eye(3, dtype=np.float64)
    rel = coords - origin
    idx = np.arange(n, dtype=np.float64)
    w1 = np.cos(0.73 * idx + 0.17)
    w2 = np.sin(1.31 * idx + 0.41)
    u1 = np.sum(w1[:, None] * rel, axis=0)
    u2 = np.sum(w2[:, None] * rel, axis=0)

    n1 = float(np.linalg.norm(u1))
    if n1 < 1e-8:
        return np.eye(3, dtype=np.float64)
    e1 = u1 / n1
    u2_orth = u2 - float(np.dot(u2, e1)) * e1
    n2 = float(np.linalg.norm(u2_orth))
    if n2 < 1e-8:
        # Rare degenerate weight collision: scan atom vectors for non-collinear direction
        for r_vec in rel:
            cand = r_vec - float(np.dot(r_vec, e1)) * e1
            nc = float(np.linalg.norm(cand))
            if nc > 1e-6:
                e2 = cand / nc
                e3 = np.cross(e1, e2)
                return np.column_stack((e1, e2, e3))
        return np.eye(3, dtype=np.float64)
    e2 = u2_orth / n2
    e3 = np.cross(e1, e2)
    return np.column_stack((e1, e2, e3))


def write_sanitized_pdb(
    structure: StructureData,
    output_path: str | Path,
    b_factor_by_residue: Optional[Dict[Tuple[str, int, str], float]] = None,
    remarks: Optional[List[str]] = None,
) -> Path:
    """Write a clean PDB file preserving chain IDs and insertion codes."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = list(remarks or [])

    serial = 1
    prev_chain: Optional[str] = None
    for res in structure.residues:
        if prev_chain is not None and res.chain_id != prev_chain:
            lines.append(f"TER   {serial:5d}      {res.resname:>3s} {prev_chain:1s}")
            serial += 1
        prev_chain = res.chain_id

        override_b = b_factor_by_residue.get(res.key) if b_factor_by_residue is not None else None
        for atom in res.atoms:
            b_val = override_b if override_b is not None else atom.b_factor
            b_val = max(-99.99, min(999.99, float(b_val)))
            name_fmt = f" {atom.name:<3s}" if len(atom.name) < 4 else f"{atom.name:<4s}"
            x, y, z = atom.coord
            lines.append(
                f"ATOM  {serial:5d} {name_fmt:4s} {res.resname:>3s} {res.chain_id:1s}"
                f"{res.auth_seq_id:4d}{res.ins_code or ' ':1s}   "
                f"{x:8.3f}{y:8.3f}{z:8.3f}{atom.occupancy:6.2f}{b_val:6.2f}          {atom.element:>2s}"
            )
            serial += 1

    if prev_chain is not None:
        lines.append(f"TER   {serial:5d}")
    lines.append("END")

    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def write_risk_pdb(
    structure: StructureData,
    residue_risk_scores: Dict[Tuple[str, int, str], float],
    output_path: str | Path,
) -> Path:
    """Write a PDB file whose B-factor column holds signed per-residue biophysical_risk_score [-99.99, +99.99]."""
    remarks = [
        f"REMARK   1 GENERATED BY BIOPHYS-GUARD FROM {structure.input_path.name}",
        "REMARK   2 B-FACTOR ENCODES SIGNED BIOPHYSICAL RISK SCORE [-99.99..+99.99]",
        "REMARK   3 -99.99 = STABLE CORE (BLUE), 0 = NEUTRAL, +99.99 = AGGREGATION HOTSPOT (RED)",
    ]
    return write_sanitized_pdb(
        structure=structure,
        output_path=output_path,
        b_factor_by_residue=residue_risk_scores,
        remarks=remarks,
    )
