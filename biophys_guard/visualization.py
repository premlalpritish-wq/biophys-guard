"""3D PyMOL, NGL Viewer HTML, risk-encoded PDB, and Mol* sidecar overlay generators."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Dict, List, Literal, Tuple

import numpy as np

from .io_parser import StructureData, write_risk_pdb
from .sap import SAPResult
from .schema import BiophysGuardReport, ResidueRiskProfile, ViewerOverlaySpec
from .stability import StabilityResult


COLOR_HEX: Dict[str, str] = {
    "aggregation_hotspot": "#dc2626",         # Red
    "destabilized": "#f97316",                # Orange
    "neutral_surface": "#e2e8f0",             # Light slate / neutral
    "stable_core": "#2563eb",                 # Blue
    "not_applicable_backbone_only": "#94a3b8",  # Muted gray
}


def _group_contiguous_spans(
    residues: List[ResidueRiskProfile],
) -> List[Dict[str, object]]:
    """Collapse sorted ResidueRiskProfile items into contiguous auth_seq_id spans per chain."""
    if not residues:
        return []
    ordered = sorted(residues, key=lambda r: (r.chain_id, r.auth_seq_id, r.ins_code))
    spans: List[Dict[str, object]] = []
    cur_chain = ordered[0].chain_id
    start_id = ordered[0].auth_seq_id
    end_id = ordered[0].auth_seq_id
    cur_ins = ordered[0].ins_code

    for r in ordered[1:]:
        if r.chain_id == cur_chain and not cur_ins and not r.ins_code and r.auth_seq_id == end_id + 1:
            end_id = r.auth_seq_id
        else:
            spans.append(
                {
                    "auth_asym_id": cur_chain,
                    "start_auth_seq_id": start_id,
                    "end_auth_seq_id": end_id,
                }
            )
            cur_chain = r.chain_id
            start_id = r.auth_seq_id
            end_id = r.auth_seq_id
            cur_ins = r.ins_code
    spans.append(
        {
            "auth_asym_id": cur_chain,
            "start_auth_seq_id": start_id,
            "end_auth_seq_id": end_id,
        }
    )
    return spans


def classify_residues_and_scores(
    structure: StructureData,
    sap_res: SAPResult,
    stab_res: StabilityResult,
    res_charges: np.ndarray,
    res_effective_pka: List[float | None],
    res_scm: np.ndarray,
) -> Tuple[List[ResidueRiskProfile], Dict[Tuple[str, int, str], float]]:
    """Classify every residue and assign signed [-99.99, +99.99] risk scores."""
    hotspot_set = set(sap_res.hotspot_residue_indices)
    core_set = set(stab_res.stable_core_indices)
    destab_set = set(stab_res.destabilized_indices)

    profiles: List[ResidueRiskProfile] = []
    risk_map: Dict[Tuple[str, int, str], float] = {}

    for i, res in enumerate(structure.residues):
        rsasa = float(sap_res.res_rsasa[i])
        plddt_val = round(res.mean_b_factor, 2) if structure.plddt_detected else None
        low_conf_warn = bool(plddt_val is not None and plddt_val < 70.0 and rsasa >= 0.15)

        if structure.is_backbone_only:
            risk_score = 0.0
            category: Literal[
                "aggregation_hotspot",
                "destabilized",
                "neutral_surface",
                "stable_core",
                "not_applicable_backbone_only",
            ] = "not_applicable_backbone_only"
            hex_color = COLOR_HEX[category]
            risk_map[res.key] = risk_score
            profiles.append(
                ResidueRiskProfile(
                    chain_id=res.chain_id,
                    auth_seq_id=res.auth_seq_id,
                    ins_code=res.ins_code,
                    auth_comp_id=res.resname,
                    aa1=res.aa1,
                    secondary_structure=stab_res.secondary_structures[i],
                    plddt=plddt_val,
                    low_confidence_rotamer_warning=low_conf_warn,
                    sasa_total=round(float(sap_res.res_sasa_total[i]), 2),
                    sasa_sidechain=0.0,
                    rsasa=0.0,
                    sap_r5=None,
                    sap_r10=None,
                    effective_pka=None,
                    charge_ph=0.0,
                    scm_r10=None,
                    stability_score_residue=None,
                    ala_scan_score=None,
                    buns_atom_count=0,
                    clash_atom_count=0,
                    biophysical_risk_score=risk_score,
                    risk_category=category,
                    hex_color=hex_color,
                )
            )
            continue

        sap_r5 = float(sap_res.res_sap_r5[i])
        sap_r10 = float(sap_res.res_sap_r10[i])
        stab_i = float(stab_res.res_stability_score[i])
        ala_i = float(stab_res.res_ala_score[i])

        if i in hotspot_set:
            category = "aggregation_hotspot"
            risk_score = float(np.clip(45.0 + 180.0 * max(0.0, sap_r5 - 0.08), 45.0, 99.99))
        elif i in destab_set:
            category = "destabilized"
            clash_boost = 20.0 * float(stab_res.res_clashes[i])
            buns_boost = 22.0 * float(stab_res.res_buns[i])
            risk_score = float(np.clip(25.0 + 18.0 * max(0.0, stab_i) + clash_boost + buns_boost, 25.0, 85.0))
        elif i in core_set:
            category = "stable_core"
            burial_strength = max(0.0, 1.0 - rsasa)
            stabilization = max(0.0, -stab_i) + 0.45 * max(0.0, ala_i)
            risk_score = float(np.clip(-(40.0 + 25.0 * burial_strength + 18.0 * stabilization), -99.99, -40.0))
        else:
            category = "neutral_surface"
            risk_score = float(np.clip(60.0 * sap_r5 + 8.0 * stab_i, -30.0, 30.0))

        hex_color = COLOR_HEX[category]
        risk_map[res.key] = round(risk_score, 2)

        profiles.append(
            ResidueRiskProfile(
                chain_id=res.chain_id,
                auth_seq_id=res.auth_seq_id,
                ins_code=res.ins_code,
                auth_comp_id=res.resname,
                aa1=res.aa1,
                secondary_structure=stab_res.secondary_structures[i],
                plddt=plddt_val,
                low_confidence_rotamer_warning=low_conf_warn,
                sasa_total=round(float(sap_res.res_sasa_total[i]), 2),
                sasa_sidechain=round(float(sap_res.res_sasa_sidechain[i]), 2),
                rsasa=round(rsasa, 4),
                sap_r5=round(sap_r5, 4),
                sap_r10=round(sap_r10, 4),
                effective_pka=res_effective_pka[i],
                charge_ph=round(float(res_charges[i]), 3),
                scm_r10=round(float(res_scm[i]), 3),
                stability_score_residue=round(stab_i, 3),
                ala_scan_score=round(ala_i, 3),
                buns_atom_count=int(stab_res.res_buns[i]),
                clash_atom_count=int(stab_res.res_clashes[i]),
                biophysical_risk_score=round(risk_score, 2),
                risk_category=category,
                hex_color=hex_color,
            )
        )

    return profiles, risk_map


def _pymol_selection_expr(profiles: List[ResidueRiskProfile]) -> str:
    """Build a PyMOL selection expression preserving chain and insertion codes."""
    if not profiles:
        return "none"
    by_chain: Dict[str, List[str]] = {}
    for p in profiles:
        tag = f"{p.auth_seq_id}{p.ins_code}".strip()
        by_chain.setdefault(p.chain_id, []).append(tag)
    clauses = []
    for chain_id, tags in sorted(by_chain.items()):
        unique_tags = list(dict.fromkeys(tags))
        id_str = "+".join(unique_tags)
        clauses.append(f"(chain {chain_id} and resi {id_str})")
    return " or ".join(clauses)


def write_pymol_overlay_script(
    risk_pdb_path: Path,
    profiles: List[ResidueRiskProfile],
    output_pml_path: Path,
) -> Path:
    """Generate a PyMOL (.pml) script coloring residues by biophysical risk."""
    hotspots = [p for p in profiles if p.risk_category == "aggregation_hotspot"]
    cores = [p for p in profiles if p.risk_category == "stable_core"]
    destab = [p for p in profiles if p.risk_category == "destabilized"]

    hotspot_sel = _pymol_selection_expr(hotspots)
    core_sel = _pymol_selection_expr(cores)
    destab_sel = _pymol_selection_expr(destab)

    lines = [
        "# PyMOL Biophysical Risk Overlay generated by biophys-guard",
        "reinitialize",
        f"load {risk_pdb_path.name}, design_risk",
        "bg_color white",
        "hide everything, design_risk",
        "show cartoon, design_risk",
        "set cartoon_fancy_helices, 1",
        "set cartoon_smooth_loops, 1",
        "",
        "# Continuous Blue (-99.99 stable core) -> White (0 neutral) -> Red (+99.99 hotspot) spectrum",
        "spectrum b, blue_white_red, design_risk, minimum=-99.99, maximum=99.99",
        "",
        "# Named biophysical risk selections",
        f"select aggregation_hotspots, design_risk and ({hotspot_sel})",
        f"select stable_core, design_risk and ({core_sel})",
        f"select destabilized_sites, design_risk and ({destab_sel})",
        "",
        "color 0xdc2626, aggregation_hotspots",
        "show sticks, aggregation_hotspots and not name N+CA+C+O",
        "create hotspot_surface, aggregation_hotspots",
        "show surface, hotspot_surface",
        "color 0xdc2626, hotspot_surface",
        "set transparency, 0.45, hotspot_surface",
        "",
        "color 0x2563eb, stable_core",
        "show sticks, stable_core and not name N+CA+C+O",
        "",
        "color 0xf97316, destabilized_sites",
        "show sticks, destabilized_sites and not name N+CA+C+O",
        "",
        "deselect",
        "orient design_risk",
        "zoom design_risk, 4",
    ]
    output_pml_path.parent.mkdir(parents=True, exist_ok=True)
    output_pml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_pml_path


def _safe_embed_text(raw: str) -> str:
    """Prevent embedded JSON/PDB strings from closing a <script> tag."""
    return raw.replace("</", "<\\/")


def write_ngl_viewer_html(
    report: BiophysGuardReport,
    risk_pdb_path: Path,
    output_html_path: Path,
) -> Path:
    """Generate a self-contained NGL.js 3D biophysical risk viewer with offline fallback."""
    pdb_text = _safe_embed_text(risk_pdb_path.read_text(encoding="utf-8"))
    report_json = _safe_embed_text(report.model_dump_json())

    def ngl_sele(items: List[ResidueRiskProfile]) -> List[Dict[str, object]]:
        return [{"chain": r.chain_id, "resi": f"{r.auth_seq_id}^{r.ins_code}" if r.ins_code else str(r.auth_seq_id)} for r in items]

    hotspot_res_js = _safe_embed_text(json.dumps(ngl_sele([r for r in report.residues if r.risk_category == "aggregation_hotspot"])))
    core_res_js = _safe_embed_text(json.dumps(ngl_sele([r for r in report.residues if r.risk_category == "stable_core"])))
    destab_res_js = _safe_embed_text(json.dumps(ngl_sele([r for r in report.residues if r.risk_category == "destabilized"])))
    residue_color_map = _safe_embed_text(
        json.dumps({f"{r.chain_id}:{r.auth_seq_id}{r.ins_code}": r.hex_color for r in report.residues})
    )

    badge_colors = {
        "GREEN": "#16a34a",
        "AMBER": "#d97706",
        "RED": "#dc2626",
        "INVALID_BACKBONE_ONLY": "#64748b",
    }
    verdict_badge_color = badge_colors[report.verdict.status]
    title = html.escape(Path(report.input_file).name)

    sap_score_disp = f"{report.sap_metrics.sap_score:.2f}" if report.sap_metrics.sap_score is not None else "N/A"
    max_sap_disp = f"{report.sap_metrics.max_sap:.3f}" if report.sap_metrics.max_sap is not None else "N/A"
    stab_disp = (
        f"{report.stability_metrics.score_value:+.1f} ({report.stability_metrics.score_unit})"
        if report.stability_metrics.score_value is not None
        else "N/A"
    )

    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>biophys-guard 3D Report — {title}</title>
  <script src="https://unpkg.com/ngl@2.3.1/dist/ngl.js"></script>
  <style>
    :root {{
      --bg: #0f172a;
      --panel: #1e293b;
      --border: #334155;
      --text: #f1f5f9;
      --muted: #94a3b8;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: var(--bg);
      color: var(--text);
      display: grid;
      grid-template-columns: 1.45fr 1fr;
      grid-template-rows: auto 1fr;
      height: 100vh;
      overflow: hidden;
    }}
    header {{
      grid-column: 1 / -1;
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 12px 20px;
      background: #020617;
      border-bottom: 1px solid var(--border);
    }}
    .brand {{ display: flex; align-items: center; gap: 12px; }}
    .badge {{
      padding: 4px 10px;
      border-radius: 999px;
      font-weight: 700;
      font-size: 12px;
      background: {verdict_badge_color};
      color: white;
      letter-spacing: 0.04em;
    }}
    .toolbar {{ display: flex; gap: 8px; flex-wrap: wrap; }}
    button {{
      background: var(--panel);
      color: var(--text);
      border: 1px solid var(--border);
      padding: 6px 12px;
      border-radius: 6px;
      cursor: pointer;
      font-size: 12px;
      font-weight: 600;
    }}
    button:hover {{ background: #334155; }}
    button.active {{ border-color: #60a5fa; background: #1d4ed8; }}
    #viewport-wrap {{ position: relative; height: 100%; background: #090d16; }}
    #viewport {{ width: 100%; height: 100%; }}
    #offline-msg {{
      display: none;
      position: absolute;
      inset: 20px;
      align-items: center;
      justify-content: center;
      text-align: center;
      color: var(--muted);
      border: 1px dashed var(--border);
      border-radius: 8px;
      padding: 24px;
    }}
    .legend {{
      position: absolute;
      left: 14px;
      bottom: 14px;
      background: rgba(15, 23, 42, 0.88);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 10px 14px;
      font-size: 12px;
      display: flex;
      flex-direction: column;
      gap: 6px;
    }}
    .leg-item {{ display: flex; align-items: center; gap: 8px; }}
    .swatch {{ width: 12px; height: 12px; border-radius: 3px; display: inline-block; }}
    aside {{
      overflow-y: auto;
      padding: 16px 18px;
      background: var(--bg);
      border-left: 1px solid var(--border);
      display: flex;
      flex-direction: column;
      gap: 14px;
    }}
    .kpis {{
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 10px;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 10px 12px;
    }}
    .kpi-label {{ font-size: 11px; color: var(--muted); text-transform: uppercase; }}
    .kpi-val {{ font-size: 16px; font-weight: 700; margin-top: 4px; }}
    h3 {{ margin: 0 0 8px 0; font-size: 13px; color: #cbd5e1; text-transform: uppercase; letter-spacing: 0.05em; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
    th, td {{ text-align: left; padding: 6px 8px; border-bottom: 1px solid var(--border); }}
    th {{ color: var(--muted); font-weight: 600; }}
    tr.clickable:hover {{ background: #334155; cursor: pointer; }}
    .pill {{
      display: inline-block;
      padding: 2px 7px;
      border-radius: 999px;
      font-size: 11px;
      font-weight: 600;
    }}
  </style>
</head>
<body>
  <header>
    <div class="brand">
      <strong>biophys-guard</strong>
      <span style="color: var(--muted); font-size: 13px;">{title} ({html.escape(report.detected_model_source)})</span>
      <span class="badge">{report.verdict.status} · G:{report.verdict.green_count} A:{report.verdict.amber_count} R:{report.verdict.red_count}</span>
    </div>
    <div class="toolbar">
      <button id="btn-cat" class="active" onclick="setColorMode('categorical')">Risk Categories</button>
      <button id="btn-bfac" onclick="setColorMode('spectrum')">B-Factor Spectrum</button>
      <button id="btn-surf" class="active" onclick="toggleHotspotSurface()">Hotspot Surface</button>
      <button id="btn-spin" onclick="toggleSpin()">Spin</button>
      <button onclick="stage && stage.autoView(600)">Reset View</button>
    </div>
  </header>

  <div id="viewport-wrap">
    <div id="viewport"></div>
    <div id="offline-msg">NGL.js WebGL library could not load from CDN (offline mode). Use the generated PyMOL script (_overlay.pml) or Antigravity Mol* Biomolecular Structure Viewer sidecar, and review the quantitative tables on the right.</div>
    <div class="legend">
      <div class="leg-item"><span class="swatch" style="background:#dc2626"></span>Aggregation Hot-spot (SAP &ge; {report.config.sap_hotspot_threshold})</div>
      <div class="leg-item"><span class="swatch" style="background:#f97316"></span>Destabilized / Clash / Atomic BUNS</div>
      <div class="leg-item"><span class="swatch" style="background:#e2e8f0"></span>Neutral Surface</div>
      <div class="leg-item"><span class="swatch" style="background:#2563eb"></span>Stable Buried Core</div>
    </div>
  </div>

  <aside>
    <div class="kpis">
      <div class="card"><div class="kpi-label">Coventry SAP Score</div><div class="kpi-val">{sap_score_disp}</div></div>
      <div class="card"><div class="kpi-label">Max SAP (R=5&Aring;)</div><div class="kpi-val">{max_sap_disp}</div></div>
      <div class="card"><div class="kpi-label">Atomic BUNS / Clash</div><div class="kpi-val">{report.stability_metrics.buns_count} / {report.stability_metrics.steric_clash_count}</div></div>
      <div class="card"><div class="kpi-label">Net Q (pH {report.charge_metrics.ph}) / pI</div><div class="kpi-val">{report.charge_metrics.net_charge:+.1f} / {report.charge_metrics.isoelectric_point:.1f}</div></div>
      <div class="card"><div class="kpi-label">Dipole (CoM)</div><div class="kpi-val">{report.charge_metrics.dipole_moment_debye:.1f} D</div></div>
      <div class="card"><div class="kpi-label">Stability</div><div class="kpi-val">{stab_disp}</div></div>
    </div>

    <div class="card">
      <h3>Traffic-Light Gate Checks</h3>
      <div id="gates-box"></div>
    </div>

    <div class="card">
      <h3>Hydrophobic Patches ({len(report.sap_metrics.hydrophobic_patches)})</h3>
      <div id="patches-box"></div>
    </div>

    <div class="card">
      <h3>Variant Predictions ({html.escape(report.stability_metrics.backend_used)})</h3>
      <div id="variants-box"></div>
    </div>

    <div class="card">
      <h3>Per-Residue Risk Profile (click row to focus)</h3>
      <div style="max-height: 240px; overflow-y: auto;">
        <table>
          <thead><tr><th>Residue</th><th>SAP</th><th>rSASA</th><th>BUNS/Clash</th><th>Category</th></tr></thead>
          <tbody id="res-tbody"></tbody>
        </table>
      </div>
    </div>
  </aside>

  <script id="pdb-data" type="text/plain">{pdb_text}</script>
  <script id="report-data" type="application/json">{report_json}</script>
  <script>
    const pdbString = document.getElementById("pdb-data").textContent;
    const report = JSON.parse(document.getElementById("report-data").textContent);
    const resColorMap = {residue_color_map};
    const hotspotList = {hotspot_res_js};
    const coreList = {core_res_js};
    const destabList = {destab_res_js};

    const toSele = (arr) => arr.length
      ? arr.map(r => `(${{r.resi}}:${{r.chain}})`).join(" or ")
      : "none";

    const hotspotSele = toSele(hotspotList);
    const coreSele = toSele(coreList);
    const destabSele = toSele(destabList);

    let stage = null;
    let compRef = null;
    let cartoonRepr = null;
    let hotspotSurfRepr = null;
    let spinning = false;
    let showSurf = true;

    if (typeof NGL === "undefined") {{
      document.getElementById("offline-msg").style.display = "flex";
    }} else {{
      const riskSchemeId = NGL.ColormakerRegistry.addScheme(function () {{
        this.atomColor = function (atom) {{
          const ins = atom.inscode ? atom.inscode.trim() : "";
          const key = `${{atom.chainname}}:${{atom.resno}}${{ins}}`;
          const hex = resColorMap[key] || "#e2e8f0";
          return parseInt(hex.slice(1), 16);
        }};
      }}, "biophys_risk");

      stage = new NGL.Stage("viewport", {{ backgroundColor: "#090d16" }});
      const blob = new Blob([pdbString], {{ type: "text/plain" }});
      stage.loadFile(blob, {{ ext: "pdb" }}).then(function (comp) {{
        compRef = comp;
        cartoonRepr = comp.addRepresentation("cartoon", {{ color: riskSchemeId, quality: "high" }});
        if (hotspotList.length) {{
          comp.addRepresentation("licorice", {{ sele: `(${{hotspotSele}}) and not backbone`, color: "#dc2626", radiusScale: 1.2 }});
          hotspotSurfRepr = comp.addRepresentation("surface", {{ sele: hotspotSele, color: "#dc2626", opacity: 0.42, surfaceType: "av" }});
        }}
        if (coreList.length) {{
          comp.addRepresentation("licorice", {{ sele: `(${{coreSele}}) and not backbone`, color: "#2563eb", radiusScale: 1.0 }});
        }}
        if (destabList.length) {{
          comp.addRepresentation("licorice", {{ sele: `(${{destabSele}}) and not backbone`, color: "#f97316", radiusScale: 1.1 }});
        }}
        comp.autoView();
      }});
      window.addEventListener("resize", () => stage && stage.handleResize(), false);
    }}

    function setColorMode(mode) {{
      if (!cartoonRepr) return;
      document.getElementById("btn-cat").classList.toggle("active", mode === "categorical");
      document.getElementById("btn-bfac").classList.toggle("active", mode === "spectrum");
      if (mode === "categorical") {{
        cartoonRepr.setParameters({{ color: "biophys_risk" }});
      }} else {{
        cartoonRepr.setParameters({{ colorScheme: "bfactor", colorScale: ["#2563eb", "#f8fafc", "#dc2626"], colorDomain: [-99.99, 99.99] }});
      }}
    }}

    function toggleHotspotSurface() {{
      showSurf = !showSurf;
      document.getElementById("btn-surf").classList.toggle("active", showSurf);
      if (hotspotSurfRepr) hotspotSurfRepr.setVisibility(showSurf);
    }}

    function toggleSpin() {{
      if (!stage) return;
      spinning = !spinning;
      document.getElementById("btn-spin").classList.toggle("active", spinning);
      stage.setSpin(spinning);
    }}

    const gatesBox = document.getElementById("gates-box");
    const flagCols = {{ GREEN: "#16a34a", AMBER: "#d97706", RED: "#dc2626", INVALID_BACKBONE_ONLY: "#64748b" }};
    gatesBox.innerHTML = report.verdict.gate_checks.map(g =>
      `<div style="font-size:12px;margin-bottom:6px;display:flex;justify-content:space-between;gap:8px;">
        <span><strong>${{g.name}}</strong>: ${{g.detail}}</span>
        <span class="pill" style="background:${{flagCols[g.flag]}};color:#fff;white-space:nowrap;">${{g.flag}}</span>
      </div>`
    ).join("");

    const patchesBox = document.getElementById("patches-box");
    if (report.sap_metrics.hydrophobic_patches.length === 0) {{
      patchesBox.innerHTML = "<span style='color:#94a3b8;font-size:12px;'>No contiguous hydrophobic patches detected.</span>";
    }} else {{
      patchesBox.innerHTML = report.sap_metrics.hydrophobic_patches.map(p =>
        `<div style="font-size:12px;margin-bottom:6px;">
          <strong style="color:#f87171;">${{p.patch_id}}</strong> (${{p.hydrophobic_sidechain_area_a2}} &Aring;&sup2; SC, max SAP ${{p.max_sap}}) &mdash; ${{p.residue_labels.join(", ")}}
        </div>`
      ).join("");
    }}

    const varBox = document.getElementById("variants-box");
    const vars = report.stability_metrics.variant_scores;
    if (!vars || vars.length === 0) {{
      varBox.innerHTML = "<span style='color:#94a3b8;font-size:12px;'>No explicit point variants requested.</span>";
    }} else {{
      varBox.innerHTML = `<table><thead><tr><th>Mut</th><th>Score</th><th>Unit</th><th>Effect</th></tr></thead><tbody>` +
        vars.map(v => `<tr><td>${{v.mutation}}</td><td>${{v.score_value >= 0 ? "+" : ""}}${{v.score_value.toFixed(3)}}</td><td>${{v.score_unit}}</td><td>${{v.classification}}</td></tr>`).join("") +
        `</tbody></table>`;
    }}

    const tbody = document.getElementById("res-tbody");
    tbody.innerHTML = report.residues.map(r => {{
      const sapStr = r.sap_r5 !== null ? r.sap_r5.toFixed(3) : "N/A";
      const seleStr = r.ins_code ? `${{r.auth_seq_id}}^${{r.ins_code}}:${{r.chain_id}}` : `${{r.auth_seq_id}}:${{r.chain_id}}`;
      return `<tr class="clickable" onclick="compRef && compRef.autoView('${{seleStr}}', 500)">
        <td>${{r.chain_id}}:${{r.aa1}}${{r.auth_seq_id}}${{r.ins_code}}</td>
        <td>${{sapStr}}</td>
        <td>${{r.rsasa.toFixed(2)}}</td>
        <td>${{r.buns_atom_count}} / ${{r.clash_atom_count}}</td>
        <td><span class="pill" style="background:${{r.hex_color}};color:${{r.risk_category === 'neutral_surface' ? '#0f172a' : '#ffffff'}}">${{r.risk_category}}</span></td>
      </tr>`;
    }}).join("");
  </script>
</body>
</html>
"""
    output_html_path.parent.mkdir(parents=True, exist_ok=True)
    output_html_path.write_text(html_doc, encoding="utf-8")
    return output_html_path


def write_molstar_sidecar_spec(
    report: BiophysGuardReport,
    artifact_structure_name: str,
    output_json_path: Path,
) -> Path:
    """Write structured MCP call specifications for science_biomolecular_structure_viewer_mcp."""
    hotspots = [r for r in report.residues if r.risk_category == "aggregation_hotspot"]
    cores = [r for r in report.residues if r.risk_category == "stable_core"]
    destab = [r for r in report.residues if r.risk_category == "destabilized"]

    annotations: List[Dict[str, object]] = []
    if hotspots:
        hs_labels = ", ".join(f"{r.aa1}{r.auth_seq_id}{r.ins_code}" for r in hotspots[:8])
        if len(hotspots) > 8:
            hs_labels += f" (+{len(hotspots) - 8} more)"
        annotations.append(
            {
                "tool": "annotate_region",
                "arguments": {
                    "residues": _group_contiguous_spans(hotspots),
                    "color": COLOR_HEX["aggregation_hotspot"],
                    "text": (
                        f"Aggregation Hot-Spots (Red): {hs_labels} | "
                        f"SapScore={report.sap_metrics.sap_score or 0.0:.2f}, "
                        f"Max SAP={report.sap_metrics.max_sap or 0.0:.3f}"
                    ),
                },
            }
        )
    if cores:
        core_labels = ", ".join(f"{r.aa1}{r.auth_seq_id}{r.ins_code}" for r in cores[:8])
        if len(cores) > 8:
            core_labels += f" (+{len(cores) - 8} more)"
        annotations.append(
            {
                "tool": "annotate_region",
                "arguments": {
                    "residues": _group_contiguous_spans(cores),
                    "color": COLOR_HEX["stable_core"],
                    "text": (
                        f"Stable Buried Core (Blue): {core_labels} | "
                        f"Packing={report.stability_metrics.packing_score:.2f}, "
                        f"BUNS={report.stability_metrics.buns_count}"
                    ),
                },
            }
        )
    if destab:
        dest_labels = ", ".join(f"{r.aa1}{r.auth_seq_id}{r.ins_code}" for r in destab[:8])
        annotations.append(
            {
                "tool": "annotate_region",
                "arguments": {
                    "residues": _group_contiguous_spans(destab),
                    "color": COLOR_HEX["destabilized"],
                    "text": f"Destabilized / Clash / BUNS Sites (Orange): {dest_labels}",
                },
            }
        )

    payload = {
        "load_structure": {"id": artifact_structure_name, "source": "artifact"},
        "set_overlays": {"confidence": bool(report.verdict.ml_confidence_flag is not None and report.verdict.ml_confidence_flag != "GREEN"), "annotations": True},
        "annotations": annotations,
    }
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return output_json_path


def generate_all_overlays(
    structure: StructureData,
    report: BiophysGuardReport,
    risk_map: Dict[Tuple[str, int, str], float],
    output_dir: Path,
    prefix: str,
) -> ViewerOverlaySpec:
    """Create risk PDB, PyMOL script, NGL HTML viewer, and Mol* sidecar spec."""
    output_dir.mkdir(parents=True, exist_ok=True)

    risk_pdb_path = write_risk_pdb(
        structure=structure,
        residue_risk_scores=risk_map,
        output_path=output_dir / f"{prefix}_biophys_risk.pdb",
    )
    pml_path = write_pymol_overlay_script(
        risk_pdb_path=risk_pdb_path,
        profiles=report.residues,
        output_pml_path=output_dir / f"{prefix}_overlay.pml",
    )
    ngl_path = write_ngl_viewer_html(
        report=report,
        risk_pdb_path=risk_pdb_path,
        output_html_path=output_dir / f"{prefix}_ngl_viewer.html",
    )
    sidecar_path = write_molstar_sidecar_spec(
        report=report,
        artifact_structure_name=structure.input_path.name,
        output_json_path=output_dir / f"{prefix}_molstar_sidecar.json",
    )

    return ViewerOverlaySpec(
        pymol_script_path=str(pml_path),
        risk_pdb_path=str(risk_pdb_path),
        ngl_viewer_path=str(ngl_path),
        molstar_sidecar_path=str(sidecar_path),
        red_hotspot_residues=[
            f"{r.chain_id}:{r.aa1}{r.auth_seq_id}{r.ins_code}"
            for r in report.residues
            if r.risk_category == "aggregation_hotspot"
        ],
        blue_core_residues=[
            f"{r.chain_id}:{r.aa1}{r.auth_seq_id}{r.ins_code}"
            for r in report.residues
            if r.risk_category == "stable_core"
        ],
        destabilized_residues=[
            f"{r.chain_id}:{r.aa1}{r.auth_seq_id}{r.ins_code}"
            for r in report.residues
            if r.risk_category == "destabilized"
        ],
    )
