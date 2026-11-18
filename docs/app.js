/* Interactive logic & 3D WebGL / Canvas Workbench for biophys-guard Distill.pub Monograph.
 * Reads verified JSON reports and risk-encoded PDBs exclusively from window.BIOPHYS_MONOGRAPH_DATA.
 */
(function () {
  "use strict";

  const DATA = window.BIOPHYS_MONOGRAPH_DATA;
  if (!DATA) {
    console.error("BIOPHYS_MONOGRAPH_DATA missing.");
    return;
  }

  const state = {
    benchmarkId: DATA.meta.benchmark_order[0],
    colorMode: "risk",
    showSurface: true,
    showSidechains: true,
    spinning: false,
    focusedResidueKey: null,
    resSearch: "",
    resFilter: "all",
    forensicId: DATA.audit_trajectory.forensic_case_study[0].id,
    pillarNum: 1,
    paratopeFilter: "all",
    variantMode: "variants",
  };

  let nglStage = null;
  let nglComp = null;
  let nglCartoon = null;
  let nglSurface = null;
  let nglSidechains = null;
  let nglFocusRepr = null;
  let useFallback3D = false;

  const fallbackCam = {
    rotX: 0.35,
    rotY: -0.45,
    zoom: 1.0,
    dragging: false,
    lastX: 0,
    lastY: 0,
    atoms: [],
    center: [0, 0, 0],
    scale: 1,
    rafId: null,
  };

  function esc(val) {
    return String(val ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function resKey(r) {
    return `${r.chain_id}:${r.auth_seq_id}${r.ins_code || ""}`;
  }

  function resLabel(r) {
    return `${r.chain_id}:${r.aa1}${r.auth_seq_id}${r.ins_code || ""}`;
  }

  function nglSele(r) {
    const ins = r.ins_code ? `^${r.ins_code}` : "";
    return `${r.auth_seq_id}${ins}:${r.chain_id}`;
  }

  function flagClass(flag) {
    if (flag === "GREEN") return "pill-green";
    if (flag === "AMBER") return "pill-amber";
    if (flag === "RED") return "pill-red";
    return "pill-invalid";
  }

  function fmtNum(val, digits = 2, fallback = "N/A") {
    if (val === null || val === undefined || Number.isNaN(Number(val))) return fallback;
    return Number(val).toFixed(digits);
  }

  function lerpHex(hexA, hexB, t) {
    const clamped = Math.max(0, Math.min(1, t));
    const a = parseInt(hexA.slice(1), 16);
    const b = parseInt(hexB.slice(1), 16);
    const ar = (a >> 16) & 255, ag = (a >> 8) & 255, ab = a & 255;
    const br = (b >> 16) & 255, bg = (b >> 8) & 255, bb = b & 255;
    const rr = Math.round(ar + (br - ar) * clamped);
    const rg = Math.round(ag + (bg - ag) * clamped);
    const rb = Math.round(ab + (bb - ab) * clamped);
    return `#${((1 << 24) + (rr << 16) + (rg << 8) + rb).toString(16).slice(1)}`;
  }

  function divergingColor(val, minVal, midVal, maxVal, lowHex, midHex, highHex) {
    if (val === null || val === undefined) return "#94a3b8";
    if (val <= midVal) {
      const denom = midVal - minVal || 1;
      return lerpHex(lowHex, midHex, (val - minVal) / denom);
    }
    const denom = maxVal - midVal || 1;
    return lerpHex(midHex, highHex, (val - midVal) / denom);
  }

  function colorForResidue(r, mode, isBackboneOnly) {
    if (isBackboneOnly) return "#94a3b8";
    if (mode === "risk") return r.hex_color || "#e2e8f0";
    if (mode === "sap") return divergingColor(r.sap_r5, -0.35, 0.0, 0.35, "#2563eb", "#f8fafc", "#dc2626");
    if (mode === "scm") return divergingColor(r.scm_r10, -4.0, 0.0, 4.0, "#dc2626", "#f8fafc", "#2563eb");
    if (mode === "confidence") {
      if (r.plddt !== null && r.plddt !== undefined) {
        if (r.plddt >= 90) return "#0053d6";
        if (r.plddt >= 70) return "#65cbf3";
        if (r.plddt >= 50) return "#ffdb13";
        return "#ff7d45";
      }
      if (r.rsasa < 0.18) return "#2563eb";
      if (r.low_confidence_rotamer_warning) return "#f97316";
      return "#0f766e";
    }
    return "#e2e8f0";
  }

  /* ================= THEME, NAV & HERO ================= */
  function initThemeAndNav() {
    const themeBtn = document.getElementById("theme-toggle-btn");
    themeBtn.addEventListener("click", () => {
      const next = document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark";
      document.documentElement.setAttribute("data-theme", next);
      themeBtn.textContent = next === "dark" ? "Theme: Dark" : "Theme: Light";
      if (useFallback3D) drawFallback3D();
    });

    const navToggle = document.getElementById("mobile-nav-toggle");
    const nav = document.getElementById("primary-nav");
    navToggle.addEventListener("click", () => {
      const open = nav.classList.toggle("open");
      navToggle.setAttribute("aria-expanded", String(open));
    });
    nav.querySelectorAll("a").forEach((a) => {
      a.addEventListener("click", () => {
        nav.classList.remove("open");
        navToggle.setAttribute("aria-expanded", "false");
      });
    });

    const tocLinks = Array.from(document.querySelectorAll(".toc-link"));
    const sections = tocLinks.map((l) => document.querySelector(l.getAttribute("href"))).filter(Boolean);
    if ("IntersectionObserver" in window && sections.length) {
      const obs = new IntersectionObserver(
        (entries) => {
          const vis = entries.filter((e) => e.isIntersecting).sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top);
          if (!vis.length) return;
          const activeId = `#${vis[0].target.id}`;
          tocLinks.forEach((l) => l.classList.toggle("active", l.getAttribute("href") === activeId));
        },
        { rootMargin: "-20% 0px -65% 0px" }
      );
      sections.forEach((s) => obs.observe(s));
    }
  }

  function renderHero() {
    const traj = DATA.audit_trajectory;
    const mask = DATA.paratope_masking_comparison;
    const se3 = DATA.se3_invariance;
    const pk = DATA.propka_salt_bridge_demo;

    document.getElementById("hero-schema-tag").textContent = `${DATA.meta.schema_title} v${DATA.meta.schema_version}`;
    document.getElementById("badge-audit-score").textContent = `Final Audit: ${traj.score_final.toFixed(2)} / ${traj.score_max.toFixed(2)} (${traj.verdict_final})`;
    document.getElementById("badge-se3-tests").textContent = DATA.meta.test_suite_summary;
    document.getElementById("badge-benchmark-count").textContent = `${DATA.meta.benchmark_order.length} RCSB PDB / AlphaFold Benchmarks`;
    document.getElementById("hero-test-status").textContent = DATA.meta.test_suite_summary;

    document.getElementById("hero-final-score").textContent = `1.90 -> 9.00 -> ${traj.score_final.toFixed(2)} / 10`;
    document.getElementById("hero-masking-delta").textContent = `${mask.monomer_chain_a_sap_score.toFixed(2)} -> ${mask.complex_chain_a_sap_score.toFixed(2)} (-${mask.pct_reduction}%)`;
    document.getElementById("hero-se3-status").textContent = `${se3.dipole_vector_so3_covariance_error_debye} D covar err`;
    document.getElementById("hero-propka-shift").textContent = `${pk.isolated_buried_asp_pka.toFixed(2)} -> ${pk.salt_bridged_asp_pka.toFixed(2)} (${pk.pka_shift_by_salt_bridge})`;
  }

  /* ================= CHAPTER 1: FORENSICS ================= */
  function renderChapter1() {
    const cases = DATA.audit_trajectory.forensic_case_study;
    const btnHost = document.getElementById("forensic-case-buttons");
    btnHost.innerHTML = cases
      .map(
        (c) =>
          `<button type="button" class="seg-btn ${c.id === state.forensicId ? "active" : ""}" data-fid="${esc(c.id)}">${esc(c.id.replace(/_/g, " "))}</button>`
      )
      .join("");

    btnHost.querySelectorAll("button").forEach((b) => {
      b.addEventListener("click", () => {
        state.forensicId = b.dataset.fid;
        renderChapter1();
      });
    });

    const cur = cases.find((c) => c.id === state.forensicId) || cases[0];
    document.getElementById("forensic-detail-card").innerHTML = `
      <div class="diff-card flaw">
        <div class="diff-tag">Iteration 0 Defect &middot; ${esc(cur.artifact)}</div>
        <p style="margin:0;">${esc(cur.defect)}</p>
      </div>
      <div class="diff-card note">
        <div class="diff-tag">Scientific Failure Consequence</div>
        <p style="margin:0;">${esc(cur.impact)}</p>
      </div>
    `;
  }

  /* ================= CHAPTER 2: AUDIT PROGRESSION ================= */
  function renderAuditSvg() {
    const pillars = DATA.audit_trajectory.pillars;
    const W = 880, H = 240, padL = 48, padR = 20, padT = 26, padB = 46;
    const plotW = W - padL - padR;
    const plotH = H - padT - padB;
    const groupW = plotW / pillars.length;

    let bars = "";
    for (let g = 0; g <= 4; g++) {
      const val = (g * 0.5).toFixed(1);
      const y = padT + plotH - (g * 0.5 / 2.0) * plotH;
      bars += `<line x1="${padL}" y1="${y}" x2="${W - padR}" y2="${y}" stroke="currentColor" stroke-opacity="0.14" stroke-dasharray="3 3" />`;
      bars += `<text x="${padL - 8}" y="${y + 4}" text-anchor="end" font-size="10.5" fill="currentColor" opacity="0.7" font-family="IBM Plex Mono">${val}</text>`;
    }

    pillars.forEach((p, idx) => {
      const x0 = padL + idx * groupW + 18;
      const bw = Math.min(34, (groupW - 44) / 3);
      const vals = [
        { v: p.iter0, col: "#b91c1c", lbl: "Iter 0" },
        { v: p.iter1, col: "#b45309", lbl: "Iter 1" },
        { v: p.iter2, col: "#0f766e", lbl: "Iter 2" },
      ];
      vals.forEach((item, k) => {
        const bh = (item.v / 2.0) * plotH;
        const bx = x0 + k * (bw + 6);
        const by = padT + plotH - bh;
        bars += `<rect x="${bx}" y="${by}" width="${bw}" height="${bh}" rx="2" fill="${item.col}" opacity="${k === 2 ? "0.95" : "0.82"}" />`;
        bars += `<text x="${bx + bw / 2}" y="${by - 5}" text-anchor="middle" font-size="10" font-weight="600" fill="currentColor" font-family="IBM Plex Mono">${item.v.toFixed(2)}</text>`;
      });
      bars += `<text x="${padL + idx * groupW + groupW / 2}" y="${H - 18}" text-anchor="middle" font-size="11" font-weight="600" fill="currentColor" font-family="IBM Plex Sans">Pillar ${p.num}</text>`;
    });

    const legend = `
      <rect x="${W - 310}" y="6" width="10" height="10" fill="#b91c1c" /><text x="${W - 295}" y="15" font-size="11" fill="currentColor">Iter 0 (1.90)</text>
      <rect x="${W - 210}" y="6" width="10" height="10" fill="#b45309" /><text x="${W - 195}" y="15" font-size="11" fill="currentColor">Iter 1 (9.00)</text>
      <rect x="${W - 110}" y="6" width="10" height="10" fill="#0f766e" /><text x="${W - 95}" y="15" font-size="11" fill="currentColor">Iter 2 (9.55)</text>
    `;

    document.getElementById("audit-progression-svg").innerHTML = `
      <svg viewBox="0 0 ${W} ${H}" width="100%" role="img" aria-label="Grouped bar chart of audit rubric scores across Iterations 0, 1, and 2">
        ${legend}
        ${bars}
      </svg>
    `;
  }

  function renderChapter2() {
    renderAuditSvg();
    const pillars = DATA.audit_trajectory.pillars;
    const btnHost = document.getElementById("pillar-selector-buttons");
    btnHost.innerHTML = pillars
      .map(
        (p) =>
          `<button type="button" class="seg-btn ${p.num === state.pillarNum ? "active" : ""}" data-pnum="${p.num}">Pillar ${p.num}: ${esc(p.name.split("&")[0].trim())}</button>`
      )
      .join("");
    btnHost.querySelectorAll("button").forEach((b) => {
      b.addEventListener("click", () => {
        state.pillarNum = Number(b.dataset.pnum);
        renderChapter2();
      });
    });

    const p = pillars.find((x) => x.num === state.pillarNum) || pillars[0];
    document.getElementById("pillar-diff-panel").innerHTML = `
      <div class="diff-card flaw">
        <div class="diff-tag">Iteration 0 Failure (${p.iter0.toFixed(2)} / ${p.max.toFixed(2)})</div>
        <p style="margin:0;">${esc(p.iter0_flaw)}</p>
      </div>
      <div class="diff-card resolved">
        <div class="diff-tag">Iteration 1 &rarr; 2 Remediation (${p.iter1.toFixed(2)} &rarr; ${p.iter2.toFixed(2)} / ${p.max.toFixed(2)})</div>
        <p style="margin:0 0 6px;">${esc(p.remediation)}</p>
        <p style="margin:0;font-size:12px;opacity:0.88;"><strong>Final Verification:</strong> ${esc(p.iter2_note)}</p>
      </div>
    `;

    const rows = pillars
      .map(
        (row) => `
      <tr>
        <td class="num">${row.num}</td>
        <td><strong>${esc(row.name)}</strong></td>
        <td class="num" style="color:var(--trace-alert-ink);font-weight:600;">${row.iter0.toFixed(2)}</td>
        <td class="num">${row.iter1.toFixed(2)}</td>
        <td class="num" style="color:var(--trace-gt-ink);font-weight:600;">${row.iter2.toFixed(2)}</td>
        <td class="num">${row.max.toFixed(2)}</td>
        <td>${esc(row.iter2_note)}</td>
      </tr>`
      )
      .join("");
    const tot0 = pillars.reduce((s, r) => s + r.iter0, 0).toFixed(2);
    const tot1 = pillars.reduce((s, r) => s + r.iter1, 0).toFixed(2);
    const tot2 = pillars.reduce((s, r) => s + r.iter2, 0).toFixed(2);
    const totM = pillars.reduce((s, r) => s + r.max, 0).toFixed(2);
    document.getElementById("audit-rubric-tbody").innerHTML =
      rows +
      `<tr style="background:var(--bg-panel);font-weight:700;">
        <td></td><td>TOTAL SCORE</td>
        <td class="num" style="color:var(--trace-alert-ink);">${tot0} (NO-GO)</td>
        <td class="num">${tot1} (GO)</td>
        <td class="num" style="color:var(--trace-gt-ink);">${tot2} (GO)</td>
        <td class="num">${totM}</td>
        <td>Final clearance granted by @deane_coventry_twin</td>
      </tr>`;
  }

  /* ================= CHAPTER 3: SAP & PARATOPE MASKING ================= */
  function renderChapter3() {
    const all = DATA.meta.benchmark_order.map((id) => ({ id, ...DATA.benchmarks[id] }));
    const refused = all.filter((b) => b.report.sidechain_completeness < 0.8);
    const truncated = all.filter((b) => b.report.sidechain_completeness >= 0.8 && b.report.sidechain_completeness < 0.95);
    const cleared = all.filter((b) => b.report.sidechain_completeness >= 0.95);

    const fmtList = (arr) =>
      arr
        .map(
          (b) =>
            `<li><code>${esc(b.id)}</code> &mdash; <strong>${(b.report.sidechain_completeness * 100).toFixed(1)}%</strong> (${esc(b.report.verdict.status)})</li>`
        )
        .join("");

    document.getElementById("completeness-tier-cards").innerHTML = `
      <div class="tier-card" style="border-left:3px solid var(--trace-alert-ink);">
        <div class="diff-tag">Tier 1 &middot; &lt; 80% Completeness (Refused)</div>
        <p style="margin:0 0 6px;">Refused as <code>INVALID_BACKBONE_ONLY</code> with <code>calculable: false</code>; zero virtual-C<sub>&beta;</sub> metric spoofing.</p>
        <ul style="margin:0;padding-left:18px;font-size:12.5px;">${fmtList(refused)}</ul>
      </div>
      <div class="tier-card" style="border-left:3px solid var(--trace-hyb-ink);">
        <div class="diff-tag">Tier 2 &middot; 80&ndash;95% Completeness (AMBER)</div>
        <p style="margin:0 0 6px;">Triggers <code>PARTIAL_SIDECHAIN_TRUNCATION_AMBER</code> alerting surface rotamer uncertainty.</p>
        <ul style="margin:0;padding-left:18px;font-size:12.5px;">${fmtList(truncated)}</ul>
      </div>
      <div class="tier-card" style="border-left:3px solid var(--trace-gt-ink);">
        <div class="diff-tag">Tier 3 &middot; &ge; 95% Completeness (Cleared)</div>
        <p style="margin:0 0 6px;">Passes completeness gating into empirical SAP, SCM, BUNS, and ML pLDDT checks.</p>
        <ul style="margin:0;padding-left:18px;font-size:12.5px;">${fmtList(cleared)}</ul>
      </div>
    `;

    const fg = document.getElementById("paratope-filter-group");
    fg.querySelectorAll("button").forEach((b) => {
      b.classList.toggle("active", b.dataset.pfilter === state.paratopeFilter);
      b.onclick = () => {
        state.paratopeFilter = b.dataset.pfilter;
        renderChapter3();
      };
    });

    const mask = DATA.paratope_masking_comparison;
    document.getElementById("paratope-masking-kpis").textContent =
      `Chain A SapScore: ${mask.monomer_chain_a_sap_score.toFixed(3)} -> ${mask.complex_chain_a_sap_score.toFixed(3)} (${mask.delta_sap_score.toFixed(3)}, -${mask.pct_reduction}%)`;

    const resRows =
      state.paratopeFilter === "all"
        ? mask.residues
        : mask.residues.filter((r) => r.delta_sap < -0.04 || r.is_hotspot_mono);

    const W = 880, H = 240, padL = 50, padR = 20, padT = 26, padB = 42;
    const plotW = W - padL - padR;
    const plotH = H - padT - padB;
    const yMin = -0.30, yMax = 0.30;
    const yZero = padT + ((yMax - 0) / (yMax - yMin)) * plotH;
    const yScale = (v) => padT + ((yMax - v) / (yMax - yMin)) * plotH;

    let svgBody = `<line x1="${padL}" y1="${yZero}" x2="${W - padR}" y2="${yZero}" stroke="currentColor" stroke-opacity="0.45" />`;
    [-0.3, -0.15, 0, 0.15, 0.3].forEach((tick) => {
      const y = yScale(tick);
      svgBody += `<line x1="${padL}" y1="${y}" x2="${W - padR}" y2="${y}" stroke="currentColor" stroke-opacity="0.12" stroke-dasharray="3 3" />`;
      svgBody += `<text x="${padL - 8}" y="${y + 4}" text-anchor="end" font-size="10.5" fill="currentColor" opacity="0.7" font-family="IBM Plex Mono">${tick.toFixed(2)}</text>`;
    });

    const step = plotW / Math.max(1, resRows.length);
    const bw = Math.max(2, Math.min(12, step * 0.38));

    resRows.forEach((r, i) => {
      const cx = padL + i * step + step * 0.5;
      const ym = yScale(r.sap_mono);
      const yc = yScale(r.sap_complex);
      const colM = r.is_hotspot_mono ? "#dc2626" : "#b45309";
      const colC = "#0f766e";
      svgBody += `<rect class="pbar" data-idx="${i}" x="${cx - bw - 1}" y="${Math.min(ym, yZero)}" width="${bw}" height="${Math.max(1, Math.abs(ym - yZero))}" fill="${colM}" opacity="0.88" />`;
      svgBody += `<rect class="pbar" data-idx="${i}" x="${cx + 1}" y="${Math.min(yc, yZero)}" width="${bw}" height="${Math.max(1, Math.abs(yc - yZero))}" fill="${colC}" opacity="0.88" />`;
      if (resRows.length <= 26 || i % 5 === 0) {
        svgBody += `<text x="${cx}" y="${H - 12}" text-anchor="middle" font-size="10" fill="currentColor" opacity="0.8" font-family="IBM Plex Mono">${esc(r.label)}</text>`;
      }
    });

    const legend = `
      <rect x="${W - 355}" y="6" width="10" height="10" fill="#dc2626" /><text x="${W - 340}" y="15" font-size="11" fill="currentColor">Monomer Hot-Spot / SAP</text>
      <rect x="${W - 180}" y="6" width="10" height="10" fill="#0f766e" /><text x="${W - 165}" y="15" font-size="11" fill="currentColor">RBD-Complex Masked SAP</text>
    `;

    const svgHost = document.getElementById("paratope-masking-svg");
    svgHost.innerHTML = `<svg viewBox="0 0 ${W} ${H}" width="100%" role="img" aria-label="Per-residue SAP change between unbound LCB1 monomer and RBD-bound complex">${legend}${svgBody}</svg>`;

    svgHost.querySelectorAll(".pbar").forEach((el) => {
      el.style.cursor = "pointer";
      const showInfo = () => {
        const r = resRows[Number(el.dataset.idx)];
        document.getElementById("paratope-hover-readout").textContent =
          `A:${r.label} | Monomer SAP=${r.sap_mono.toFixed(4)} (rSASA=${r.rsasa_mono.toFixed(2)}) -> Complex SAP=${r.sap_complex.toFixed(4)} (rSASA=${r.rsasa_complex.toFixed(2)}) | Delta=${r.delta_sap.toFixed(4)}`;
      };
      el.addEventListener("mouseenter", showInfo);
      el.addEventListener("click", showInfo);
    });
  }

  /* ================= CHAPTER 4: ELECTROSTATICS & SE(3) ================= */
  function renderChapter4() {
    const pk = DATA.propka_salt_bridge_demo;
    const ch4Iso = document.getElementById("ch4-iso-pka");
    const ch4Sb = document.getElementById("ch4-sb-pka");
    const ch4Shift = document.getElementById("ch4-shift-pka");
    if (ch4Iso) ch4Iso.textContent = pk.isolated_buried_asp_pka.toFixed(2);
    if (ch4Sb) ch4Sb.textContent = `pKa = ${pk.salt_bridged_asp_pka.toFixed(2)}`;
    if (ch4Shift) ch4Shift.textContent = `delta = ${pk.pka_shift_by_salt_bridge.toFixed(2)}`;

    document.getElementById("propka-coupling-panel").innerHTML = `
      <div class="diff-card note">
        <div class="diff-tag">Isolated Buried ASP 100 (Ala102-Capped Hydrophobic Shell)</div>
        <p style="margin:0;">Hydrophobic shell desolvation shifts effective \\(\\text{p}K_a\\) from model <strong>${pk.model_asp_pka.toFixed(2)}</strong> upward to <strong>${pk.isolated_buried_asp_pka.toFixed(2)}</strong>, shifting equilibrium partially toward the neutral acid (sidechain <code>q=${pk.isolated_buried_asp_q_ph74.toFixed(3)}</code> vs <code>-1.000</code> at pH 7.4).</p>
      </div>
      <div class="diff-card resolved">
        <div class="diff-tag">Coupled Buried ASP 100 &ndash; LYS 101 Salt Bridge (${pk.salt_bridge_count} pair)</div>
        <p style="margin:0;">Coulombic + H-bond coupling shifts effective \\(\\text{p}K_a\\) back down to <strong>${pk.salt_bridged_asp_pka.toFixed(2)}</strong> (<strong>${pk.pka_shift_by_salt_bridge.toFixed(2)} pKa units</strong>), restoring full carboxylate ionization (<code>q=${pk.salt_bridged_asp_q_ph74.toFixed(3)}</code> at pH 7.4).</p>
      </div>
    `;

    const se3 = DATA.se3_invariance;
    document.getElementById("se3-invariance-tbody").innerHTML = se3.metrics
      .map(
        (m) => `
      <tr>
        <td>${esc(m.metric)}</td>
        <td><code>${esc(m.unit)}</code></td>
        <td class="num">${m.original}</td>
        <td class="num">${m.transformed}</td>
        <td class="num" style="color:var(--trace-gt-ink);font-weight:600;">${m.abs_delta}</td>
        <td><code>${esc(m.tolerance)}</code></td>
      </tr>`
      )
      .join("");

    document.getElementById("se3-dipole-covar").textContent =
      `Vector Covariance Check ||mu(rot) - R * mu(orig)|| = ${se3.dipole_vector_so3_covariance_error_debye} D (exact float64 rigid-body invariance).`;

    const calcIds = DATA.meta.benchmark_order.filter((id) => DATA.benchmarks[id].report.charge_metrics.calculable);
    document.getElementById("electrostatics-benchmark-tbody").innerHTML = calcIds
      .map((id) => {
        const c = DATA.benchmarks[id].report.charge_metrics;
        let interchain = "Monomer (N/A)";
        if (c.interchain_sfvcsp !== null && c.interchain_sfvcsp !== undefined) {
          interchain = id === "1N8Z_Trastuzumab_Fab" ? `OPIG SFvCSP = ${c.interchain_sfvcsp.toFixed(2)} (GREEN)` : `Binder-Target qA*qB = ${c.interchain_sfvcsp.toFixed(2)}`;
        }
        return `
        <tr>
          <td><code>${esc(id)}</code></td>
          <td class="num">${c.net_charge >= 0 ? "+" : ""}${c.net_charge.toFixed(2)}</td>
          <td class="num">${c.isoelectric_point.toFixed(2)}</td>
          <td class="num">${c.dipole_moment_debye.toFixed(1)}</td>
          <td class="num">${c.charge_symmetry_score.toFixed(3)}</td>
          <td class="num">${c.max_scm_positive.toFixed(2)} e</td>
          <td class="num">${c.min_scm_negative.toFixed(2)} e</td>
          <td>${esc(interchain)}</td>
        </tr>`;
      })
      .join("");
  }

  /* ================= CHAPTER 5: UNITS, VARIANTS & pLDDT ================= */
  function renderChapter5() {
    document.getElementById("unit-honesty-matrix-tbody").innerHTML = `
      <tr>
        <td><strong>FoldX Force Field</strong></td>
        <td><code>kcal/mol</code></td>
        <td><code>true</code></td>
        <td>Full 3D side-chain rotamer repacking &amp; empirical van der Waals/solvation energies; required before claiming physical \\(\\Delta\\Delta G\\).</td>
      </tr>
      <tr>
        <td><strong>ESM-1v Masked Marginal</strong></td>
        <td><code>log_likelihood_ratio</code></td>
        <td><code>false</code></td>
        <td>Dimensionless sequence evolutionary likelihood \\(\\log P(x_{\\text{wt}} \\mid x_{-i}) - \\log P(x_{\\text{mut}} \\mid x_{-i})\\) via <code>transformers</code> or <code>fair-esm</code> (positive = deleterious, matching \\(\\Delta\\Delta G\\) polarity).</td>
      </tr>
      <tr>
        <td><strong>Built-In Static Heuristic</strong></td>
        <td><code>heuristic_au</code></td>
        <td><code>false</code></td>
        <td>Fast buried transfer + cavity + steric clash + Proline helix/sheet proxy (<code>delta_g_fold_kcal_mol = null</code>).</td>
      </tr>
    `;

    const seg = document.getElementById("variant-explorer-seg");
    seg.querySelectorAll("button").forEach((b) => {
      b.classList.toggle("active", b.dataset.vmode === state.variantMode);
      b.onclick = () => {
        state.variantMode = b.dataset.vmode;
        renderChapter5();
      };
    });

    const st = DATA.benchmarks["7JZU_LCB1_monomer"].report.stability_metrics;
    const rows = state.variantMode === "variants" ? st.variant_scores : st.alanine_scan;

    const W = 880, H = 200, padL = 48, padR = 16, padT = 20, padB = 34;
    const plotW = W - padL - padR, plotH = H - padT - padB;
    const vals = rows.map((r) => r.score_value);
    const yMin = Math.min(-1.5, Math.min(...vals));
    const yMax = Math.max(3.5, Math.max(...vals));
    const yScale = (v) => padT + ((yMax - v) / (yMax - yMin || 1)) * plotH;
    const yZero = yScale(0);

    let svg = `<line x1="${padL}" y1="${yZero}" x2="${W - padR}" y2="${yZero}" stroke="currentColor" stroke-opacity="0.45" />`;
    const step = plotW / Math.max(1, rows.length);
    const bw = Math.max(4, Math.min(48, step * 0.65));
    rows.forEach((r, i) => {
      const cx = padL + i * step + step * 0.5;
      const yv = yScale(r.score_value);
      const col = r.score_value > 1.5 ? "#dc2626" : r.score_value < -0.5 ? "#0f766e" : "#64748b";
      svg += `<rect x="${cx - bw / 2}" y="${Math.min(yv, yZero)}" width="${bw}" height="${Math.max(2, Math.abs(yv - yZero))}" fill="${col}" rx="1.5" />`;
      if (rows.length <= 10 || i % 5 === 0) {
        svg += `<text x="${cx}" y="${H - 10}" text-anchor="middle" font-size="10" fill="currentColor" font-family="IBM Plex Mono">${esc(r.mutation.replace("A:", ""))}</text>`;
      }
    });
    document.getElementById("variant-scan-svg").innerHTML = `<svg viewBox="0 0 ${W} ${H}" width="100%" role="img" aria-label="Variant and alanine scan heuristic score profile">${svg}</svg>`;

    document.getElementById("variant-scan-tbody").innerHTML = rows
      .map(
        (v) => `
      <tr>
        <td><code>${esc(v.mutation)}</code></td>
        <td class="num" style="font-weight:600;color:${v.score_value > 1.5 ? "var(--trace-alert-ink)" : v.score_value < -0.5 ? "var(--trace-gt-ink)" : "inherit"};">${v.score_value >= 0 ? "+" : ""}${v.score_value.toFixed(3)}</td>
        <td><code>${esc(v.score_unit)}</code></td>
        <td class="num">${v.sasa_burial.toFixed(3)}</td>
        <td>${esc(v.classification)}</td>
        <td>${esc(v.rationale)}</td>
      </tr>`
      )
      .join("");

    // p53 pLDDT track
    const p53 = DATA.benchmarks["AF_P04637_p53"].report;
    const pRes = p53.residues;
    const pW = 880, pH = 190, ppL = 44, ppR = 16, ppT = 18, ppB = 28;
    const pPlotW = pW - ppL - ppR, pPlotH = pH - ppT - ppB;
    const pScaleY = (v) => ppT + ((100 - v) / 100) * pPlotH;
    const y70 = pScaleY(70);
    const pts = pRes.map((r, i) => `${(ppL + (i / (pRes.length - 1)) * pPlotW).toFixed(1)},${pScaleY(r.plddt).toFixed(1)}`).join(" ");
    let bars = "";
    pRes.forEach((r, i) => {
      if (r.plddt < 70) {
        const x = ppL + (i / pRes.length) * pPlotW;
        bars += `<rect x="${x.toFixed(1)}" y="${ppT}" width="${(pPlotW / pRes.length + 0.5).toFixed(1)}" height="${pPlotH}" fill="#dc2626" opacity="0.14" />`;
      }
    });
    document.getElementById("plddt-track-svg").innerHTML = `
      <svg viewBox="0 0 ${pW} ${pH}" width="100%" role="img" aria-label="AF_P04637_p53 pLDDT confidence track">
        ${bars}
        <line x1="${ppL}" y1="${y70}" x2="${pW - ppR}" y2="${y70}" stroke="#dc2626" stroke-dasharray="4 3" />
        <text x="${ppL - 6}" y="${y70 + 4}" text-anchor="end" font-size="10.5" fill="#dc2626" font-family="IBM Plex Mono">70</text>
        <polyline fill="none" stroke="#1d4ed8" stroke-width="1.8" points="${pts}" />
        <text x="${ppL}" y="${pH - 8}" font-size="10.5" fill="currentColor" font-family="IBM Plex Mono">Residue 1 (N-TAD)</text>
        <text x="${pW / 2}" y="${pH - 8}" text-anchor="middle" font-size="10.5" fill="currentColor" font-family="IBM Plex Mono">DBD Core (~100&ndash;290, high pLDDT)</text>
        <text x="${pW - ppR}" y="${pH - 8}" text-anchor="end" font-size="10.5" fill="currentColor" font-family="IBM Plex Mono">Residue ${pRes.length} (C-reg)</text>
      </svg>
    `;
    const pctLow = ((p53.verdict.low_confidence_residue_count / p53.residue_count) * 100).toFixed(1);
    document.getElementById("plddt-summary-note").textContent =
      `AF_P04637_p53: mean_pLDDT=${p53.stability_metrics.mean_plddt.toFixed(1)}, shaded RED zones = ${p53.verdict.low_confidence_residue_count} residues (${pctLow}%) < 70 pLDDT flagged for rotamer uncertainty.`;
  }

  /* ================= CHAPTER 6: 3D WORKBENCH & FALLBACK ================= */
  function parsePdbForFallback(pdbText, resList, mode, isBackboneOnly) {
    const resMap = {};
    resList.forEach((r) => { resMap[resKey(r)] = r; });

    const atoms = [];
    const lines = pdbText.split(/\r?\n/);
    for (const line of lines) {
      if (!line.startsWith("ATOM  ") && !line.startsWith("HETATM")) continue;
      const name = line.slice(12, 16).trim();
      if (name !== "CA") continue;
      const chain = (line.slice(21, 22).trim() || "A");
      const seq = parseInt(line.slice(22, 26).trim(), 10);
      const ins = line.slice(26, 27).trim();
      const x = parseFloat(line.slice(30, 38));
      const y = parseFloat(line.slice(38, 46));
      const z = parseFloat(line.slice(46, 54));
      if (Number.isNaN(x) || Number.isNaN(y) || Number.isNaN(z)) continue;
      const k = `${chain}:${seq}${ins}`;
      const r = resMap[k];
      atoms.push({
        key: k,
        label: r ? resLabel(r) : k,
        x, y, z,
        color: r ? colorForResidue(r, mode, isBackboneOnly) : "#94a3b8",
        isHotspot: r ? r.risk_category === "aggregation_hotspot" : false,
      });
    }

    if (!atoms.length) return;
    let cx = 0, cy = 0, cz = 0;
    atoms.forEach((a) => { cx += a.x; cy += a.y; cz += a.z; });
    cx /= atoms.length; cy /= atoms.length; cz /= atoms.length;
    let maxR = 1;
    atoms.forEach((a) => {
      a.dx = a.x - cx; a.dy = a.y - cy; a.dz = a.z - cz;
      const d = Math.hypot(a.dx, a.dy, a.dz);
      if (d > maxR) maxR = d;
    });
    fallbackCam.atoms = atoms;
    fallbackCam.scale = 1 / maxR;
  }

  function drawFallback3D() {
    const canvas = document.getElementById("fallback-3d-canvas");
    if (!canvas || !useFallback3D) return;
    const rect = canvas.parentElement.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.max(300, Math.round(rect.width * dpr));
    canvas.height = Math.max(240, Math.round(rect.height * dpr));
    const ctx = canvas.getContext("2d");
    const W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);

    const cosX = Math.cos(fallbackCam.rotX), sinX = Math.sin(fallbackCam.rotX);
    const cosY = Math.cos(fallbackCam.rotY), sinY = Math.sin(fallbackCam.rotY);
    const radius = Math.min(W, H) * 0.40 * fallbackCam.zoom;

    const proj = fallbackCam.atoms.map((a, idx) => {
      const x1 = a.dx * cosY + a.dz * sinY;
      const z1 = -a.dx * sinY + a.dz * cosY;
      const y2 = a.dy * cosX - z1 * sinX;
      const z2 = a.dy * sinX + z1 * cosX;
      const depth = 1 / (2.6 - z2 * fallbackCam.scale);
      return {
        ...a,
        idx,
        sx: W * 0.5 + x1 * fallbackCam.scale * radius * depth * 2.1,
        sy: H * 0.5 - y2 * fallbackCam.scale * radius * depth * 2.1,
        z: z2,
        focused: a.key === state.focusedResidueKey,
      };
    });

    // Backbone trace segments
    ctx.lineWidth = 2.4 * dpr;
    for (let i = 1; i < proj.length; i++) {
      const a = proj[i - 1], b = proj[i];
      if (Math.hypot(a.dx - b.dx, a.dy - b.dy, a.dz - b.dz) > 6.5) continue;
      ctx.strokeStyle = b.color;
      ctx.beginPath();
      ctx.moveTo(a.sx, a.sy);
      ctx.lineTo(b.sx, b.sy);
      ctx.stroke();
    }

    // Painter's depth-sorted spheres for hotspots and focused residues
    proj.sort((a, b) => a.z - b.z).forEach((p) => {
      if (!p.isHotspot && !p.focused && proj.length > 140 && p.idx % 2 !== 0) return;
      const r = (p.focused ? 7.5 : p.isHotspot ? 5.2 : 3.2) * dpr * fallbackCam.zoom;
      ctx.fillStyle = p.color;
      ctx.beginPath();
      ctx.arc(p.sx, p.sy, Math.max(2, r), 0, Math.PI * 2);
      ctx.fill();
      if (p.focused) {
        ctx.strokeStyle = "#facc15";
        ctx.lineWidth = 2.5 * dpr;
        ctx.stroke();
      }
    });
  }

  function initFallbackInteraction() {
    const canvas = document.getElementById("fallback-3d-canvas");
    canvas.addEventListener("pointerdown", (e) => {
      fallbackCam.dragging = true;
      fallbackCam.lastX = e.clientX;
      fallbackCam.lastY = e.clientY;
    });
    window.addEventListener("pointermove", (e) => {
      if (!fallbackCam.dragging) return;
      fallbackCam.rotY += (e.clientX - fallbackCam.lastX) * 0.008;
      fallbackCam.rotX += (e.clientY - fallbackCam.lastY) * 0.008;
      fallbackCam.lastX = e.clientX;
      fallbackCam.lastY = e.clientY;
      drawFallback3D();
    });
    window.addEventListener("pointerup", () => { fallbackCam.dragging = false; });
    canvas.addEventListener(
      "wheel",
      (e) => {
        if (!useFallback3D) return;
        e.preventDefault();
        fallbackCam.zoom = Math.max(0.5, Math.min(2.6, fallbackCam.zoom * (e.deltaY > 0 ? 0.92 : 1.08)));
        drawFallback3D();
      },
      { passive: false }
    );
  }

  function updateLegend() {
    const legendEl = document.getElementById("wb-stage-legend");
    const bm = DATA.benchmarks[state.benchmarkId];
    const items = [];
    if (bm.report.is_backbone_only) {
      items.push(["#94a3b8", "Refused Backbone / CB-Stub Only (< 80%)"]);
    } else if (state.colorMode === "risk") {
      items.push(["#dc2626", "Aggregation Hot-Spot (SAP >= 0.15)"]);
      items.push(["#2563eb", "Stable Buried Core"]);
      items.push(["#f97316", "Destabilized / Clash / BUNS"]);
      items.push(["#e2e8f0", "Neutral Surface"]);
    } else if (state.colorMode === "sap") {
      items.push(["#dc2626", "High Hydrophobic SAP (+0.35)"]);
      items.push(["#f8fafc", "Neutral SAP (0.00)"]);
      items.push(["#2563eb", "Hydrophilic Patch (-0.35)"]);
    } else if (state.colorMode === "scm") {
      items.push(["#2563eb", "Exposed Positive SCM (+4.0 e)"]);
      items.push(["#f8fafc", "Neutral SCM (0.0 e)"]);
      items.push(["#dc2626", "Exposed Negative SCM (-4.0 e)"]);
    } else {
      const hasPlddt = bm.report.residues.some((r) => r.plddt !== null);
      if (hasPlddt) {
        items.push(["#0053d6", "Very high (pLDDT >= 90)"]);
        items.push(["#65cbf3", "Confident (70 - 90)"]);
        items.push(["#ffdb13", "Low confidence (50 - 70)"]);
        items.push(["#ff7d45", "Disordered (< 50)"]);
      } else {
        items.push(["#2563eb", "Buried Core (rSASA < 0.18)"]);
        items.push(["#0f766e", "Solvent Exposed Crystal"]);
        items.push(["#f97316", "Flagged Rotamer Warning"]);
      }
    }
    legendEl.innerHTML = items
      .map(([col, txt]) => `<div class="legend-row"><span class="swatch-dot" style="background:${col};"></span><span>${esc(txt)}</span></div>`)
      .join("");
  }

  function loadWorkbenchStructure() {
    const bm = DATA.benchmarks[state.benchmarkId];
    const rep = bm.report;
    const isBB = Boolean(rep.is_backbone_only);

    const viewportEl = document.getElementById("ngl-viewport");
    const fallbackEl = document.getElementById("fallback-3d-canvas");

    parsePdbForFallback(bm.pdb_text, rep.residues, state.colorMode, isBB);

    if (typeof window.NGL === "undefined") {
      useFallback3D = true;
    } else if (!nglStage) {
      try {
        nglStage = new window.NGL.Stage("ngl-viewport", { backgroundColor: "#101622" });
        if (!nglStage || !nglStage.viewer || !nglStage.viewer.renderer) {
          useFallback3D = true;
        } else {
          nglStage.signals.hovered.add((pickingProxy) => {
            if (!pickingProxy || !pickingProxy.atom) return;
            const a = pickingProxy.atom;
            const ins = a.inscode ? a.inscode.trim() : "";
            const key = `${a.chainname}:${a.resno}${ins}`;
            const cur = DATA.benchmarks[state.benchmarkId].report;
            const r = cur.residues.find((x) => resKey(x) === key);
            if (!r) return;
            document.getElementById("wb-hover-atom").textContent =
              `${resLabel(r)} (${a.atomname}) | ${r.risk_category} | SAP(5A)=${fmtNum(r.sap_r5, 3)} | rSASA=${fmtNum(r.rsasa, 2)} | SCM=${fmtNum(r.scm_r10, 2)}`;
          });
        }
      } catch (_err) {
        useFallback3D = true;
      }
    }

    if (useFallback3D) {
      viewportEl.style.display = "none";
      fallbackEl.style.display = "block";
      drawFallback3D();
      updateLegend();
      return;
    }

    viewportEl.style.display = "block";
    fallbackEl.style.display = "none";
    nglStage.removeAllComponents();
    nglComp = null; nglCartoon = null; nglSurface = null; nglSidechains = null; nglFocusRepr = null;

    const blob = new Blob([bm.pdb_text], { type: "text/plain" });
    nglStage.loadFile(blob, { ext: "pdb" }).then((comp) => {
      nglComp = comp;
      applyNglRepresentations();
      comp.autoView();
      nglStage.setSpin(state.spinning);
    }).catch(() => {
      useFallback3D = true;
      viewportEl.style.display = "none";
      fallbackEl.style.display = "block";
      drawFallback3D();
    });
    updateLegend();
  }

  function applyNglRepresentations() {
    if (!nglComp || useFallback3D) {
      if (useFallback3D) {
        const bm = DATA.benchmarks[state.benchmarkId];
        parsePdbForFallback(bm.pdb_text, bm.report.residues, state.colorMode, bm.report.is_backbone_only);
        drawFallback3D();
      }
      updateLegend();
      return;
    }

    const bm = DATA.benchmarks[state.benchmarkId];
    const rep = bm.report;
    const isBB = Boolean(rep.is_backbone_only);
    const colorMap = {};
    rep.residues.forEach((r) => {
      colorMap[resKey(r)] = colorForResidue(r, state.colorMode, isBB);
    });

    const schemeId = window.NGL.ColormakerRegistry.addScheme(function () {
      this.atomColor = function (atom) {
        const ins = atom.inscode ? atom.inscode.trim() : "";
        const k = `${atom.chainname}:${atom.resno}${ins}`;
        const hex = colorMap[k] || "#e2e8f0";
        return parseInt(hex.slice(1), 16);
      };
    }, `bg_${state.benchmarkId}_${state.colorMode}`);

    nglComp.removeAllRepresentations();
    nglCartoon = nglComp.addRepresentation("cartoon", { color: schemeId, quality: "high" });

    const hotspots = rep.residues.filter((r) => r.risk_category === "aggregation_hotspot");
    const core = rep.residues.filter((r) => r.risk_category === "stable_core");
    const destab = rep.residues.filter((r) => r.risk_category === "destabilized");

    if (!isBB && state.showSidechains) {
      if (hotspots.length) {
        nglComp.addRepresentation("licorice", { sele: `(${hotspots.map(nglSele).join(" or ")}) and not backbone`, color: "#dc2626", radiusScale: 1.25 });
      }
      if (core.length && core.length <= 80) {
        nglComp.addRepresentation("licorice", { sele: `(${core.map(nglSele).join(" or ")}) and not backbone`, color: "#2563eb", radiusScale: 0.95 });
      }
      if (destab.length) {
        nglComp.addRepresentation("licorice", { sele: `(${destab.map(nglSele).join(" or ")}) and not backbone`, color: "#f97316", radiusScale: 1.15 });
      }
    }
    if (!isBB && state.showSurface && hotspots.length) {
      const maxSurf = rep.residue_count > 120 ? 12 : 24;
      const hs = hotspots.slice(0, maxSurf).map(nglSele).join(" or ");
      nglSurface = nglComp.addRepresentation("surface", {
        sele: hs,
        color: "#dc2626",
        opacity: 0.38,
        surfaceType: rep.residue_count > 120 ? "sas" : "av",
      });
    }
    if (state.focusedResidueKey) {
      const target = rep.residues.find((r) => resKey(r) === state.focusedResidueKey);
      if (target) {
        nglFocusRepr = nglComp.addRepresentation("ball+stick", { sele: nglSele(target), color: "#facc15", radiusScale: 1.4 });
      }
    }
    updateLegend();
  }

  function focusResidue(r) {
    state.focusedResidueKey = resKey(r);
    document.getElementById("wb-hover-atom").textContent =
      `Focused ${resLabel(r)} | Category=${r.risk_category} | SAP(5A)=${fmtNum(r.sap_r5, 3)} | rSASA=${fmtNum(r.rsasa, 2)} | SCM(10A)=${fmtNum(r.scm_r10, 2)} | BUNS=${r.buns_atom_count} | Clashes=${r.clash_atom_count}`;
    if (!useFallback3D && nglComp) {
      applyNglRepresentations();
      nglComp.autoView(nglSele(r), 500);
    } else {
      drawFallback3D();
    }
    renderWorkbenchResidueTable();
    renderSequenceStrip();
  }

  function renderSequenceStrip() {
    const bm = DATA.benchmarks[state.benchmarkId];
    const rep = bm.report;
    const isBB = Boolean(rep.is_backbone_only);
    const strip = document.getElementById("wb-seq-strip");
    strip.innerHTML = rep.residues
      .map((r) => {
        const bg = colorForResidue(r, state.colorMode, isBB);
        const darkText = ["#e2e8f0", "#f8fafc", "#ffdb13"].includes(bg.toLowerCase());
        const k = resKey(r);
        const sel = state.focusedResidueKey === k ? " selected" : "";
        return `<button type="button" class="seq-cell${sel}" data-rkey="${esc(k)}" title="${esc(resLabel(r))} | SAP=${fmtNum(r.sap_r5, 3)} | ${esc(r.risk_category)}" style="background:${bg};color:${darkText ? "#111518" : "#ffffff"};">${esc(r.aa1)}</button>`;
      })
      .join("");

    strip.querySelectorAll("button").forEach((btn) => {
      btn.addEventListener("click", () => {
        const found = rep.residues.find((r) => resKey(r) === btn.dataset.rkey);
        if (found) focusResidue(found);
      });
    });
  }

  function renderWorkbenchResidueTable() {
    const rep = DATA.benchmarks[state.benchmarkId].report;
    const q = state.resSearch.trim().toLowerCase();
    const filtered = rep.residues.filter((r) => {
      if (state.resFilter !== "all" && r.risk_category !== state.resFilter) return false;
      if (!q) return true;
      const hay = `${resLabel(r)} ${r.auth_comp_id} ${r.risk_category}`.toLowerCase();
      return hay.includes(q);
    });

    const shownCount = Math.min(filtered.length, 250);
    document.getElementById("wb-res-count").textContent =
      filtered.length > 250
        ? `Showing first 250 of ${filtered.length} matching (${rep.residue_count} total)`
        : `Showing ${shownCount} / ${rep.residue_count} residues`;
    document.getElementById("wb-res-tbody").innerHTML = filtered
      .slice(0, 250)
      .map((r) => {
        const k = resKey(r);
        const focused = state.focusedResidueKey === k ? " focused-row" : "";
        const darkText = r.risk_category === "neutral_surface";
        return `<tr class="clickable${focused}" data-rkey="${esc(k)}">
          <td><code>${esc(resLabel(r))}</code></td>
          <td class="num">${fmtNum(r.sap_r5, 3)}</td>
          <td class="num">${fmtNum(r.rsasa, 2)}</td>
          <td class="num">${fmtNum(r.scm_r10, 2)}</td>
          <td class="num">${r.buns_atom_count} / ${r.clash_atom_count}</td>
          <td><span class="status-pill" style="background:${r.hex_color};color:${darkText ? "#0f172a" : "#ffffff"};border-color:transparent;">${esc(r.risk_category)}</span></td>
        </tr>`;
      })
      .join("");

    document.getElementById("wb-res-tbody").querySelectorAll("tr").forEach((tr) => {
      tr.addEventListener("click", () => {
        const found = rep.residues.find((r) => resKey(r) === tr.dataset.rkey);
        if (found) focusResidue(found);
      });
    });
  }

  function renderWorkbenchTelemetry() {
    const bm = DATA.benchmarks[state.benchmarkId];
    const rep = bm.report;
    const v = rep.verdict;
    const sap = rep.sap_metrics;
    const chg = rep.charge_metrics;
    const st = rep.stability_metrics;

    document.getElementById("wb-header-bar").innerHTML = `
      <div style="display:flex;flex-wrap:wrap;justify-content:space-between;align-items:center;gap:10px;">
        <div>
          <strong>${esc(bm.title)}</strong>
          <div style="font-size:12px;color:var(--ink-secondary);margin-top:2px;">${esc(bm.provenance)} &middot; Source: <code>${esc(rep.detected_model_source)}</code> (${esc(rep.structure_format.toUpperCase())}) &middot; ${rep.residue_count} aa / ${rep.atom_count} atoms</div>
        </div>
        <span class="status-pill ${flagClass(v.status)}">${esc(v.status)} &middot; G:${v.green_count} A:${v.amber_count} R:${v.red_count}</span>
      </div>
      <div style="margin-top:6px;font-size:12.5px;color:var(--ink-secondary);">${esc(bm.highlight)}</div>
    `;

    const refBanner = document.getElementById("wb-refusal-banner");
    if (rep.is_backbone_only) {
      refBanner.style.display = "block";
      refBanner.innerHTML = `<strong>STRICT SCIENTIFIC REFUSAL (${esc(v.status)}):</strong> ${esc(sap.status_reason || "Sidechain completeness < 80%; SAP, BUNS, and SCM cannot be fabricated.")}`;
    } else {
      refBanner.style.display = "none";
    }

    document.getElementById("wb-download-json").href = bm.report_file;
    document.getElementById("wb-download-json").download = `${bm.id}_report.json`;
    document.getElementById("wb-download-pdb").href = bm.pdb_file;
    document.getElementById("wb-download-pdb").download = `${bm.id}_biophys_risk.pdb`;

    document.getElementById("wb-kpis").innerHTML = [
      ["Verdict", v.status, `G:${v.green_count} A:${v.amber_count} R:${v.red_count}`],
      ["Sidechain Comp.", `${(v.sidechain_completeness * 100).toFixed(1)}%`, `${rep.chain_ids.join(", ")} chain(s)`],
      ["Coventry SapScore", sap.calculable ? fmtNum(sap.sap_score, 2) : "Refused", sap.calculable ? `Max SAP=${fmtNum(sap.max_sap, 3)}` : "Backbone gate"],
      ["Hot-Spots / Patches", sap.calculable ? `${sap.aggregation_hotspot_count} / ${sap.hydrophobic_patches.length}` : "N/A", sap.calculable ? `PSH=${fmtNum(sap.psh_surface_hydrophobicity, 2)}` : "N/A"],
      ["Net Q / pI", chg.calculable ? `${chg.net_charge >= 0 ? "+" : ""}${chg.net_charge.toFixed(2)} e` : "Refused", chg.calculable ? `pI=${chg.isoelectric_point.toFixed(2)}` : "N/A"],
      ["Dipole / CSS", chg.calculable ? `${chg.dipole_moment_debye.toFixed(1)} D` : "Refused", chg.calculable ? `CSS=${chg.charge_symmetry_score.toFixed(3)}` : "N/A"],
      ["Atomic BUNS", st.calculable ? String(st.buns_count) : "Refused", st.calculable ? `HB sat=${(st.hbond_satisfaction_rate * 100).toFixed(1)}%` : "N/A"],
      ["Clashes / Units", st.calculable ? `${fmtNum(st.molprobity_heavy_clashscore, 2)}` : "Refused", st.calculable ? `${st.score_value} ${st.score_unit}` : "not_applicable"],
    ]
      .map(
        ([k, val, sub]) => `
      <div class="readout-cell">
        <div class="r-head">${esc(k)}</div>
        <div class="r-val">${esc(val)}</div>
        <div class="r-sub">${esc(sub)}</div>
      </div>`
      )
      .join("");

    document.getElementById("wb-verdict-summary").textContent = `${v.gate_checks.length} checks`;
    document.getElementById("wb-gates-tbody").innerHTML = v.gate_checks
      .map(
        (g) => `
      <tr>
        <td><strong>${esc(g.name)}</strong><div style="font-size:11px;color:var(--ink-secondary);">${esc(g.reference_source)}</div></td>
        <td style="font-size:12px;">${esc(g.detail)}</td>
        <td><span class="status-pill ${flagClass(g.flag)}">${esc(g.flag)}</span></td>
      </tr>`
      )
      .join("");

    document.getElementById("wb-recommendations-box").innerHTML =
      v.recommendations && v.recommendations.length
        ? `<strong>Actionable Guidance:</strong> ${v.recommendations.map(esc).join(" ")}`
        : "";

    const nPatches = sap.hydrophobic_patches.length;
    document.getElementById("wb-patch-count").textContent = sap.calculable
      ? nPatches > 6 ? `${nPatches} patches (top 6 shown)` : `${nPatches} patches`
      : "N/A";
    document.getElementById("wb-patches-box").innerHTML =
      !sap.calculable || !sap.hydrophobic_patches.length
        ? `<span style="color:var(--ink-secondary);">${sap.calculable ? "No contiguous hydrophobic patches detected." : "Patches not computed on backbone-only / CB-stubbed model."}</span>`
        : sap.hydrophobic_patches
            .slice(0, 6)
            .map(
              (p) =>
                `<div style="margin-bottom:6px;"><strong style="color:var(--trace-alert-ink);">${esc(p.patch_id)}</strong> (${p.hydrophobic_sidechain_area_a2} &Aring;&sup2; SC, max SAP ${p.max_sap}) &mdash; ${esc(p.residue_labels.join(", "))}</div>`
            )
            .join("");

    // Callout chips
    const hotspots = rep.residues.filter((r) => r.risk_category === "aggregation_hotspot").slice(0, 12);
    const core = rep.residues.filter((r) => r.risk_category === "stable_core").slice(0, 10);
    const destab = rep.residues.filter((r) => r.risk_category === "destabilized").slice(0, 8);
    const makeChip = (r, cls) => `<button type="button" class="res-chip ${cls}" data-rkey="${esc(resKey(r))}">${esc(resLabel(r))}</button>`;

    const chipHost = document.getElementById("wb-callout-chips");
    if (rep.is_backbone_only) {
      chipHost.innerHTML = `<span style="font-size:12.5px;color:var(--ink-secondary);">No hotspot/core callouts on refused backbone-only structure.</span>`;
    } else {
      chipHost.innerHTML = `
        <div class="chip-row">
          ${hotspots.map((r) => makeChip(r, "hotspot")).join("")}
          ${core.map((r) => makeChip(r, "core")).join("")}
          ${destab.map((r) => makeChip(r, "destab")).join("")}
        </div>
      `;
      chipHost.querySelectorAll("button").forEach((b) => {
        b.addEventListener("click", () => {
          const found = rep.residues.find((r) => resKey(r) === b.dataset.rkey);
          if (found) focusResidue(found);
        });
      });
    }

    renderSequenceStrip();
    renderWorkbenchResidueTable();
  }

  function renderBenchmarkTabs() {
    const host = document.getElementById("wb-benchmark-tabs");
    host.innerHTML = DATA.meta.benchmark_order
      .map((id) => {
        const bm = DATA.benchmarks[id];
        const status = bm.report.verdict.status;
        const active = id === state.benchmarkId ? " active" : "";
        return `<button type="button" class="btn btn-sm${active}" data-bid="${esc(id)}"><span>${esc(bm.short_label)}</span><span class="status-pill ${flagClass(status)}" style="padding:1px 6px;font-size:10px;">${status === "INVALID_BACKBONE_ONLY" ? "REFUSED" : status}</span></button>`;
      })
      .join("");

    host.querySelectorAll("button").forEach((b) => {
      b.addEventListener("click", () => {
        state.benchmarkId = b.dataset.bid;
        state.focusedResidueKey = null;
        document.getElementById("wb-hover-atom").textContent = "Drag to rotate, scroll to zoom, or click a residue chip/bar to focus.";
        renderBenchmarkTabs();
        renderWorkbenchTelemetry();
        loadWorkbenchStructure();
      });
    });
  }

  function initWorkbenchControls() {
    initFallbackInteraction();

    document.getElementById("wb-color-modes").querySelectorAll("button").forEach((b) => {
      b.addEventListener("click", () => {
        state.colorMode = b.dataset.cmode;
        document.getElementById("wb-color-modes").querySelectorAll("button").forEach((x) => x.classList.toggle("active", x === b));
        applyNglRepresentations();
        renderSequenceStrip();
      });
    });

    const surfBtn = document.getElementById("wb-btn-surface");
    surfBtn.addEventListener("click", () => {
      state.showSurface = !state.showSurface;
      surfBtn.classList.toggle("active", state.showSurface);
      applyNglRepresentations();
    });

    const scBtn = document.getElementById("wb-btn-sidechains");
    scBtn.addEventListener("click", () => {
      state.showSidechains = !state.showSidechains;
      scBtn.classList.toggle("active", state.showSidechains);
      applyNglRepresentations();
    });

    const spinBtn = document.getElementById("wb-btn-spin");
    spinBtn.addEventListener("click", () => {
      state.spinning = !state.spinning;
      spinBtn.classList.toggle("active", state.spinning);
      if (nglStage) nglStage.setSpin(state.spinning);
    });

    const resetView = () => {
      state.focusedResidueKey = null;
      document.getElementById("wb-hover-atom").textContent = "Showing entire structure.";
      if (!useFallback3D && nglComp) {
        applyNglRepresentations();
        nglComp.autoView(500);
      } else {
        fallbackCam.rotX = 0.35; fallbackCam.rotY = -0.45; fallbackCam.zoom = 1.0;
        drawFallback3D();
      }
      renderWorkbenchResidueTable();
      renderSequenceStrip();
    };
    document.getElementById("wb-btn-reset").addEventListener("click", resetView);
    document.getElementById("wb-btn-show-all").addEventListener("click", resetView);

    document.getElementById("wb-res-search").addEventListener("input", (e) => {
      state.resSearch = e.target.value;
      renderWorkbenchResidueTable();
    });

    document.getElementById("wb-res-filter").querySelectorAll("button").forEach((b) => {
      b.addEventListener("click", () => {
        state.resFilter = b.dataset.rfilter;
        document.getElementById("wb-res-filter").querySelectorAll("button").forEach((x) => x.classList.toggle("active", x === b));
        renderWorkbenchResidueTable();
      });
    });

    window.addEventListener("resize", () => {
      if (nglStage && !useFallback3D) nglStage.handleResize();
      if (useFallback3D) drawFallback3D();
    });
  }

  function renderMath() {
    if (typeof window.renderMathInElement === "function") {
      window.renderMathInElement(document.body, {
        delimiters: [
          { left: "$$", right: "$$", display: true },
          { left: "\\[", right: "\\]", display: true },
          { left: "\\(", right: "\\)", display: false },
        ],
        throwOnError: false,
      });
    }
  }

  function boot() {
    initThemeAndNav();
    renderHero();
    renderChapter1();
    renderChapter2();
    renderChapter3();
    renderChapter4();
    renderChapter5();
    renderBenchmarkTabs();
    initWorkbenchControls();
    renderWorkbenchTelemetry();
    loadWorkbenchStructure();
    renderMath();
    window.addEventListener("load", renderMath);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
