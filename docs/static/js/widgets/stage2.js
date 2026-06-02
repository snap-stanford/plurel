/* ============================================================================
   Stage 2 - Foreign Key Generation via Bipartite Graphs (Hierarchical SBM).
   For a chosen (parent, child) table pair, every CHILD row is assigned one
   PARENT row. A hierarchical stochastic block model decides the assignment:
   nodes are placed in clusters at each level, and same-cluster pairs connect
   with high probability (0.9) vs a tiny background (~0.0015). More
   levels / clusters => smaller "pools" => tighter row-level locality.
   Faithful port of plurel/bipartite.py (assign_cluster_at_levels,
   get_probs_at_levels, sample_bipartite_assignments).
   ========================================================================== */
(function () {
  "use strict";
  const d3 = window.d3;
  const PV = window.PV;
  const W = 760, H = 196;
  const SIZE_A = 24, SIZE_B = 30;        // parent rows, child rows (full-width figure)
  const Y_A = 52, Y_B = 140, CELL = 14;
  // harmonious, Distill-flavoured cluster palette
  const PALETTE = ["#2563eb", "#0f766e", "#6d28d9", "#d97706", "#be185d",
                   "#0891b2", "#65a30d", "#475569", "#9333ea"];

  // ---- faithful HSBM ports --------------------------------------------------
  function assignClusters(numNodes, hierarchy) {
    const numBase = hierarchy.reduce((a, b) => a * b, 1);
    const nodesPerCluster = Math.ceil(numNodes / numBase);
    const offsets = [];
    for (let c = 0; c < numBase - 1; c++) offsets.push(nodesPerCluster * (c + 1));
    offsets.push(numNodes);
    const lvl = Array.from({ length: numNodes }, () => new Array(hierarchy.length).fill(0));
    const base = new Array(numNodes).fill(0);
    let start = 0;
    for (let c = 0; c < numBase; c++) {
      const end = Math.min(offsets[c], numNodes);
      for (let l = 0; l < hierarchy.length; l++) {
        let fac = 1;
        for (let q = l + 1; q < hierarchy.length; q++) fac *= hierarchy[q];
        for (let nidx = start; nidx < end; nidx++) lvl[nidx][l] = Math.floor(c / fac) % hierarchy[l];
      }
      for (let nidx = start; nidx < end; nidx++) base[nidx] = c;
      start = end;
    }
    return { lvl, base, numBase };
  }

  function probsAtLevels(ha, hb, rng) {
    const out = [];
    for (let l = 0; l < ha.length; l++) {
      const s0 = ha[l], s1 = hb[l];
      const P = Array.from({ length: s0 }, () => Array.from({ length: s1 }, () => rng.uniform(0.001, 0.002)));
      const mx = Math.max(s0, s1);
      for (let i = 0; i < mx; i++) P[i % s0][i % s1] = 0.9;
      out.push(P);
    }
    return out;
  }

  function sampleAssignments(sizeA, sizeB, hierarchy, rng) {
    const ca = assignClusters(sizeA, hierarchy);
    const cb = assignClusters(sizeB, hierarchy);
    const logP = probsAtLevels(hierarchy, hierarchy, rng).map((P) => P.map((row) => row.map(Math.log)));
    const parent = new Array(sizeB);
    for (let b = 0; b < sizeB; b++) {
      const lp = new Array(sizeA).fill(0);
      let mx = -Infinity;
      for (let a = 0; a < sizeA; a++) {
        let s = 0;
        for (let l = 0; l < hierarchy.length; l++) s += logP[l][ca.lvl[a][l]][cb.lvl[b][l]];
        lp[a] = s; if (s > mx) mx = s;
      }
      let sum = 0;
      for (let a = 0; a < sizeA; a++) { lp[a] = Math.exp(lp[a] - mx); sum += lp[a]; }
      const u = rng.next();
      let cdf = 0, pick = sizeA - 1;
      for (let a = 0; a < sizeA; a++) { cdf += lp[a] / sum; if (cdf >= u) { pick = a; break; } }
      parent[b] = pick;
    }
    return { parent, ca, cb };
  }

  PV.mountStage2 = function (root) {
    const state = { levels: 2, clusters: 2, seed: 11, hover: null };

    root.innerHTML = `
      <div class="pv-widget pv-stage2">
        <div class="pv-head">
          <span class="pv-badge">Stage 2 · foreign keys</span>
          <span class="pv-title">Wiring child rows to parent rows with an HSBM</span>
        </div>
        <div class="pv-controls">
          <div class="pv-control">
            <label>hsbm levels · <span class="pv-val" data-v="levels"></span></label>
            <input type="range" class="pv-range" data-c="levels" min="1" max="2" step="1">
          </div>
          <div class="pv-control">
            <label>clusters / level · <span class="pv-val" data-v="clusters"></span></label>
            <input type="range" class="pv-range" data-c="clusters" min="1" max="3" step="1">
          </div>
          <div class="pv-control pv-spacer">
            <label>&nbsp;</label>
            <button class="pv-btn" data-act="regen">↻ Resample</button>
          </div>
        </div>
        <div class="pv-bip-split">
          <svg class="pv-bip-svg" viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMidYMid meet" role="img" aria-label="Bipartite hierarchical stochastic block model: parent rows on top, child rows below; an arrow points from each parent row to the child rows that reference it"></svg>
          <div class="pv-bip-side">
            <div class="pv-stat"><span class="pv-stat-num" data-el="nclusters"></span><span class="pv-stat-lbl"># clusters (cᴸ)</span></div>
            <div class="pv-stat"><span class="pv-stat-num" data-el="pool"></span><span class="pv-stat-lbl">parent pool / child</span></div>
            <div class="pv-stat"><span class="pv-stat-num" data-el="within"></span><span class="pv-stat-lbl">within-cluster links</span></div>
          </div>
        </div>
        <div class="pv-readout" data-el="readout"></div>
        <p class="pv-note">The arrow runs from each <b>parent</b> row (top, a primary key) to the <b>child</b> rows that reference it (bottom, foreign keys). Same-cluster pairs (matching colour) connect with probability <b>0.9</b>, so more levels and clusters give each child a smaller, same-coloured pool of parents. Hover a row to trace its link.</p>
      </div>`;

    const svg = d3.select(root).select("svg");
    const readoutEl = root.querySelector('[data-el="readout"]');
    let model = null;

    function build() {
      const hierarchy = new Array(state.levels).fill(state.clusters);
      const rng = new PV.RNG(state.seed);
      const { parent, ca, cb } = sampleAssignments(SIZE_A, SIZE_B, hierarchy, rng);
      const numBase = ca.numBase;
      let within = 0;
      for (let b = 0; b < SIZE_B; b++) if (ca.base[parent[b]] === cb.base[b]) within++;
      model = { hierarchy, parent, ca, cb, numBase, within };
    }

    const xA = (i) => 40 + i * ((W - 80) / (SIZE_A - 1));
    const xB = (j) => 40 + j * ((W - 80) / (SIZE_B - 1));
    const color = (c) => PALETTE[c % PALETTE.length];

    function draw() {
      const { parent, ca, cb } = model;
      svg.selectAll("*").remove();
      PV.addArrowMarker(svg.append("defs"), "pv-bip-arrow", "context-stroke", 8);

      // group labels
      svg.append("text").attr("class", "pv-rowlabel").attr("x", 38).attr("y", 18)
        .attr("font-weight", 700).attr("fill", "var(--tok-input)").text("parent rows · table_A (entity, primary keys)");
      svg.append("text").attr("class", "pv-rowlabel").attr("x", 38).attr("y", H - 6)
        .attr("font-weight", 700).attr("fill", "var(--tok-output)").text("child rows · table_B (activity, foreign keys)");

      // soft cluster tints (filled rounded rect behind each contiguous cluster)
      // + a subtle id label, so membership is not conveyed by colour alone.
      function tints(size, base, xf, y, labelAbove) {
        const g = svg.append("g");
        if (model.numBase <= 1) return;
        let s = 0;
        for (let i = 1; i <= size; i++) {
          if (i === size || base[i] !== base[s]) {
            const x0 = xf(s) - CELL / 2 - 4, x1 = xf(i - 1) + CELL / 2 + 4, c = color(base[s]);
            g.append("rect").attr("x", x0).attr("y", y - CELL / 2 - 5).attr("width", x1 - x0).attr("height", CELL + 10).attr("rx", 6)
              .attr("fill", c).attr("fill-opacity", 0.1).attr("stroke", c).attr("stroke-opacity", 0.38).attr("stroke-width", 1);
            g.append("text").attr("x", (x0 + x1) / 2).attr("y", labelAbove ? y - CELL / 2 - 9 : y + CELL / 2 + 14)
              .attr("text-anchor", "middle").attr("font-size", "8px").attr("font-weight", 700).attr("fill", c).attr("fill-opacity", 0.9).text("c" + base[s]);
            s = i;
          }
        }
      }
      tints(SIZE_A, ca.base, xA, Y_A, true);
      tints(SIZE_B, cb.base, xB, Y_B, false);

      // curved, bundled child -> parent edges (vertical tangents => clean bundles)
      const edges = parent.map((a, b) => ({ a, b }));
      // parent (top) -> child (bottom): the primary key is referenced by the foreign key
      const bend = (Y_B - Y_A) * 0.42;
      const bipPath = (d) => {
        const x1 = xA(d.a), y1 = Y_A + CELL / 2, x2 = xB(d.b), y2 = Y_B - CELL / 2;
        return `M${x1},${y1} C${x1},${y1 + bend} ${x2},${y2 - bend} ${x2},${y2}`;
      };
      const edgeSel = svg.append("g").selectAll("path").data(edges).join("path")
        .attr("class", "pv-bip-edge").attr("stroke", (d) => color(cb.base[d.b])).attr("d", bipPath);

      // parent + child cells
      function cells(size, base, xf, y, kind) {
        return svg.append("g").selectAll("rect").data(d3.range(size)).join("rect")
          .attr("class", "pv-rowcell")
          .attr("x", (i) => xf(i) - CELL / 2).attr("y", y - CELL / 2).attr("width", CELL).attr("height", CELL).attr("rx", 3.5)
          .attr("fill", (i) => color(base[i]))
          .attr("tabindex", 0).attr("role", "button")
          .attr("aria-label", (i) => `${kind === "a" ? "parent row a" : "child row b"}${i}, cluster ${base[i]}`);
      }
      const aSel = cells(SIZE_A, ca.base, xA, Y_A, "a")
        .on("mouseenter", (e, i) => hoverParent(i)).on("mouseleave", clearHover)
        .on("focus", (e, i) => hoverParent(i)).on("blur", clearHover).on("click", (e, i) => hoverParent(i));
      const bSel = cells(SIZE_B, cb.base, xB, Y_B, "b")
        .on("mouseenter", (e, j) => hoverChild(j)).on("mouseleave", clearHover)
        .on("focus", (e, j) => hoverChild(j)).on("blur", clearHover).on("click", (e, j) => hoverChild(j));

      function hoverChild(j) {
        const a = parent[j];
        edgeSel.classed("hl", (d) => d.b === j).classed("dim", (d) => d.b !== j);
        aSel.classed("dim", (i) => i !== a);
        bSel.classed("dim", (k) => k !== j);
        const same = ca.base[a] === cb.base[j];
        readoutEl.innerHTML = `<b>child b${j}</b> (cluster ${cb.base[j]}) links to <b>parent a${a}</b> (cluster ${ca.base[a]}), ${same ? "<b>same</b> cluster (the 0.9 path)" : "a <b>cross-cluster</b> jump (rare background path)"}.`;
      }
      function hoverParent(i) {
        const kids = edges.filter((d) => d.a === i).map((d) => d.b);
        edgeSel.classed("hl", (d) => d.a === i).classed("dim", (d) => d.a !== i);
        aSel.classed("dim", (k) => k !== i);
        bSel.classed("dim", (k) => !kids.includes(k));
        readoutEl.innerHTML = `<b>parent a${i}</b> (cluster ${ca.base[i]}) is referenced by <b>${kids.length}</b> child row${kids.length === 1 ? "" : "s"}${kids.length ? " (" + kids.map((k) => "b" + k).join(", ") + ")" : ""}.`;
      }
      function clearHover() {
        edgeSel.classed("hl", false).classed("dim", false);
        aSel.classed("dim", false); bSel.classed("dim", false);
        renderReadout();
      }

      renderStats();
      renderReadout();
    }

    function renderStats() {
      const pool = (SIZE_A / model.numBase);
      root.querySelector('[data-el="nclusters"]').textContent = model.numBase;
      root.querySelector('[data-el="pool"]').textContent = (pool >= 1 ? pool.toFixed(pool < 10 ? 1 : 0) : "<1");
      root.querySelector('[data-el="within"]').textContent = Math.round((model.within / SIZE_B) * 100) + "%";
    }
    function renderReadout() {
      const locality = model.numBase === 1
        ? "every parent is equally likely, so rows depend on the <b>full</b> parent table (low locality)"
        : `each child draws from a pool of ≈ <b>${(SIZE_A / model.numBase).toFixed(1)}</b> same-cluster parents (high locality)`;
      readoutEl.innerHTML = `<b>${state.levels}</b> level${state.levels > 1 ? "s" : ""} × <b>${state.clusters}</b> cluster${state.clusters > 1 ? "s" : ""} → <b>${model.numBase}</b> block${model.numBase > 1 ? "s" : ""}: ${locality}.`;
    }

    function syncControls() {
      root.querySelectorAll("[data-v]").forEach((s) => (s.textContent = state[s.dataset.v]));
      root.querySelectorAll("[data-c]").forEach((inp) => (inp.value = state[inp.dataset.c]));
    }

    root.querySelectorAll("[data-c]").forEach((inp) =>
      inp.addEventListener("input", () => { state[inp.dataset.c] = +inp.value; syncControls(); build(); draw(); })
    );
    root.querySelector('[data-act="regen"]').addEventListener("click", () => { state.seed = PV.randomSeed(); build(); draw(); });

    syncControls();
    build();
    draw();
  };
})();
