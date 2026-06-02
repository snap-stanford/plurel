/* ============================================================================
   Stage 1 - Schema Generation via Directed Graphs.
   An interactive DAG of TABLES: nodes are tables, a parent->child edge is a
   foreign key in the child referencing the parent's primary key. Tables that
   are referenced (out-degree >= 1) are *entity* tables; leaves (out-degree 0)
   are *activity* tables (more rows, timestamped). Nodes are laid out by
   topological generation - the order in which tables are synthesized.
   Mirrors plurel/schema.py (RandomSchemaGraphBuilder) + plurel/dataset.py.
   ========================================================================== */
(function () {
  "use strict";
  const d3 = window.d3;
  const PV = window.PV;
  const W = 560, H = 248, RW = 52, RH = 30;

  PV.mountStage1 = function (root) {
    const state = { numTables: 7, layout: "BarabasiAlbert", seed: 7, hover: -1 };

    root.innerHTML = `
      <div class="pv-widget pv-stage1">
        <div class="pv-head">
          <span class="pv-badge">Stage 1 · schema</span>
          <span class="pv-title">A schema as a DAG of tables</span>
        </div>
        <div class="pv-controls">
          <div class="pv-control">
            <label>num_tables · <span class="pv-val" data-v="numTables"></span></label>
            <input type="range" class="pv-range" data-c="numTables" min="3" max="12" step="1">
          </div>
          <div class="pv-control">
            <label>table layout</label>
            <div class="pv-seg" data-seg="layout">
              <button data-val="BarabasiAlbert">Barabási–Albert</button>
              <button data-val="ReverseRandomTree">ReverseRandomTree</button>
              <button data-val="WattsStrogatz">Watts–Strogatz</button>
            </div>
          </div>
          <div class="pv-control pv-spacer">
            <label>&nbsp;</label>
            <button class="pv-btn" data-act="regen">↻ New schema</button>
          </div>
        </div>
        <div class="pv-legend">
          <span><span class="pv-sw entity"></span>entity table · referenced (out-degree ≥ 1)</span>
          <span><span class="pv-sw activity"></span>activity table · leaf, timestamped</span>
          <span><span class="pv-sw line"></span>edge: parent → child (FK lives in the child)</span>
        </div>
        <div class="pv-split">
          <svg class="pv-svg" viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMidYMid meet" role="img" aria-label="Schema as a directed acyclic graph of tables; arrows point from a parent table to a child table that holds a foreign key"></svg>
          <div data-el="panel"></div>
        </div>
        <div class="pv-readout" data-el="readout"></div>
        <p class="pv-note">Each table is a node; a <b>parent → child</b> edge is a foreign key in the child referencing the parent's <code>row_idx</code> primary key. Tables are generated left-to-right in <b>topological order</b>. Hover a table to inspect its columns; drag to rearrange.</p>
      </div>`;

    const svg = d3.select(root).select("svg");
    const panelEl = root.querySelector('[data-el="panel"]');
    const readoutEl = root.querySelector('[data-el="readout"]');

    let model = null;

    function build() {
      const rng = new PV.RNG(state.seed);
      const n = state.numTables;
      let edges = PV.genGraphEdges(n, state.layout, rng);
      const { generations, inDeg, outDeg } = PV.topoGenerations(n, edges);
      const inNbrs = Array.from({ length: n }, () => []);
      const outNbrs = Array.from({ length: n }, () => []);
      for (const [u, v] of edges) { inNbrs[v].push(u); outNbrs[u].push(v); }
      const genOf = new Array(n).fill(0);
      generations.forEach((g, gi) => g.forEach((id) => (genOf[id] = gi)));

      // per-table attributes (deterministic in seed so hover is stable)
      const tables = [];
      for (let i = 0; i < n; i++) {
        const tr = new PV.RNG(state.seed * 131 + i * 17 + 1);
        const isActivity = outDeg[i] === 0;
        const numFeat = PV.samplePowerLawInt(3, 40, tr); // num_cols_choices.sample_pl() over [3,40]
        const feats = [];
        for (let f = 0; f < numFeat; f++) {
          const cat = tr.next() < 0.5;
          feats.push(cat ? { name: `f${f}`, type: "categorical", ncat: 2 + tr.randint(9) } : { name: `f${f}`, type: "numeric" });
        }
        tables.push({
          id: i,
          name: `table_${i}`,
          isActivity,
          numRows: isActivity ? 2000 + tr.randint(3001) : 500 + tr.randint(501),
          gen: genOf[i],
          fkeys: inNbrs[i].slice().sort((a, b) => a - b),
          children: outNbrs[i].slice().sort((a, b) => a - b),
          feats,
        });
      }

      // layout: columns by generation, vertical spread within a generation
      const G = generations.length;
      const xFor = (g) => (G === 1 ? W / 2 : 46 + (g * (W - 92)) / (G - 1));
      const nodes = [];
      generations.forEach((g, gi) => {
        const k = g.length;
        g.forEach((id, j) => {
          const y = k === 1 ? H / 2 + 6 : 36 + (j * (H - 66)) / (k - 1);
          nodes.push({ id, x: xFor(gi), y });
        });
      });
      const pos = Object.fromEntries(nodes.map((p) => [p.id, p]));
      model = { n, edges, tables, pos, nodes, generations, inNbrs, outNbrs };
    }

    function draw() {
      const { edges, pos, nodes, tables } = model;
      svg.selectAll("*").remove();
      const defs = svg.append("defs");
      PV.addArrowMarker(defs, "pv-arrow");

      // generation guide labels
      const gxs = [...new Set(nodes.map((p) => Math.round(p.x)))].sort((a, b) => a - b);
      svg.append("g").selectAll("text").data(gxs).join("text")
        .attr("class", "pv-genlabel").attr("x", (d) => d).attr("y", 16).attr("text-anchor", "middle")
        .text((d, i) => (i === 0 ? "generated first →" : ""));

      const gE = svg.append("g");
      const gN = svg.append("g");

      const edgeSel = gE.selectAll("path").data(edges).join("path")
        .attr("class", "pv-edge")
        .attr("d", ([u, v]) => PV.edgePath(pos[u].x, pos[u].y, pos[v].x, pos[v].y, RW / 2 + 2, RW / 2 + 6, 0));

      const nodeSel = gN.selectAll("g").data(nodes, (d) => d.id).join("g")
        .attr("class", (d) => `pv-node ${tables[d.id].isActivity ? "activity" : "entity"}`)
        .attr("transform", (d) => `translate(${d.x},${d.y})`)
        .attr("tabindex", 0).attr("role", "button")
        .attr("aria-label", (d) => `${tables[d.id].name}, ${tables[d.id].isActivity ? "activity" : "entity"} table`);
      nodeSel.append("rect").attr("x", -RW / 2).attr("y", -RH / 2).attr("width", RW).attr("height", RH).attr("rx", 7);
      nodeSel.append("text").attr("text-anchor", "middle").attr("dy", "0.34em").attr("font-size", "11px")
        .text((d) => `T${d.id}`);

      function setHover(id) {
        state.hover = id;
        const t = id >= 0 ? tables[id] : null;
        const fkSet = new Set(id >= 0 ? model.inNbrs[id] : []);     // tables this references
        const childSet = new Set(id >= 0 ? model.outNbrs[id] : []); // tables referencing this
        edgeSel
          .classed("hl", ([u, v]) => id >= 0 && (v === id || u === id))
          .classed("dim", ([u, v]) => id >= 0 && !(v === id || u === id));
        nodeSel
          .classed("sel", (d) => d.id === id)
          .classed("parent", (d) => fkSet.has(d.id))
          .classed("dim", (d) => id >= 0 && d.id !== id && !fkSet.has(d.id) && !childSet.has(d.id));
        renderPanel(t);
      }

      nodeSel
        .on("mouseenter", (e, d) => setHover(d.id))
        .on("mouseleave", () => setHover(-1))
        .on("focus", (e, d) => setHover(d.id))
        .on("blur", () => setHover(-1))
        .on("click", (e, d) => setHover(d.id));

      nodeSel.style("cursor", "grab").call(
        d3.drag()
          .on("start", function () { d3.select(this).raise(); })
          .on("drag", function (e, d) {
            d.x = PV.clamp(e.x, RW / 2, W - RW / 2);
            d.y = PV.clamp(e.y, RH / 2, H - RH / 2);
            pos[d.id].x = d.x; pos[d.id].y = d.y;
            d3.select(this).attr("transform", `translate(${d.x},${d.y})`);
            edgeSel.attr("d", ([u, v]) => PV.edgePath(pos[u].x, pos[u].y, pos[v].x, pos[v].y, RW / 2 + 2, RW / 2 + 6, 0));
          })
      );

      setHover(-1);
      renderReadout();
    }

    function renderPanel(t) {
      if (!t) {
        const ent = model.tables.filter((x) => !x.isActivity).length;
        const act = model.tables.length - ent;
        const gens = model.generations.length;
        panelEl.innerHTML = `
          <div class="pv-tag">Schema summary</div>
          <div class="pv-summary">
            <div class="pv-srow"><span class="pv-snum">${model.tables.length}</span><span class="pv-slbl">tables</span></div>
            <div class="pv-srow"><span class="pv-snum">${model.edges.length}</span><span class="pv-slbl">foreign keys</span></div>
            <div class="pv-srow"><span class="pv-snum">${ent}</span><span class="pv-slbl"><i class="pv-dot entity"></i>entity tables</span></div>
            <div class="pv-srow"><span class="pv-snum">${act}</span><span class="pv-slbl"><i class="pv-dot activity"></i>activity tables</span></div>
            <div class="pv-srow"><span class="pv-snum">${gens}</span><span class="pv-slbl">generation${gens > 1 ? "s" : ""}</span></div>
          </div>
          <div class="pv-card-empty">Hover a table to inspect its columns.</div>`;
        return;
      }
      const cls = t.isActivity ? "activity" : "entity";
      const rows = [];
      rows.push(`<tr><td><span class="pv-colname">row_idx</span></td><td><span class="pv-keytag pk">PK</span></td></tr>`);
      t.fkeys.forEach((p, i) => {
        rows.push(`<tr><td><span class="pv-colname">foreign_row_${i}</span></td><td><span class="pv-keytag fk">FK → table_${p}</span></td></tr>`);
      });
      const maxFeat = 5;
      t.feats.slice(0, maxFeat).forEach((f) => {
        const tag = f.type === "numeric"
          ? `<span class="pv-keytag num">numeric</span>`
          : `<span class="pv-keytag cat">categorical · ${f.ncat}</span>`;
        rows.push(`<tr><td><span class="pv-colname">${f.name}</span></td><td>${tag}</td></tr>`);
      });
      const moreFeat = t.feats.length - maxFeat;
      const moreRow = moreFeat > 0 ? `<tr><td colspan="2" style="color:var(--d-faint);font-size:0.7rem">+ ${moreFeat} more feature column${moreFeat > 1 ? "s" : ""}</td></tr>` : "";
      panelEl.innerHTML = `
        <div class="pv-tag">Inspecting</div>
        <div class="pv-card">
          <div class="pv-card-hd ${cls}">
            <span>${t.name}</span>
            <span class="pv-rows">${t.isActivity ? "activity" : "entity"} · ${t.numRows.toLocaleString()} rows</span>
          </div>
          <div class="pv-card-body">
            <table class="pv-coltable">${rows.join("")}${moreRow}</table>
          </div>
        </div>`;
    }

    function renderReadout() {
      const id = state.hover;
      if (id < 0) {
        readoutEl.innerHTML = `<b>${state.layout}</b> schema with ${model.tables.length} tables. Activity (leaf) tables get a timestamp column and 2k–5k rows; entity tables get 500–1k rows.`;
        return;
      }
      const t = model.tables[id];
      const pill = t.isActivity ? `<span class="pv-pill activity">activity</span>` : `<span class="pv-pill entity">entity</span>`;
      const refs = t.fkeys.length
        ? `references <b>${t.fkeys.length}</b> parent table${t.fkeys.length > 1 ? "s" : ""} (${t.fkeys.map((p) => "table_" + p).join(", ")}) via foreign keys`
        : `has <b>no</b> foreign keys; it is a source table generated independently first`;
      const by = t.children.length
        ? `, and is referenced by <b>${t.children.length}</b> child table${t.children.length > 1 ? "s" : ""}`
        : `, and is referenced by none (a leaf)`;
      readoutEl.innerHTML = `${pill}<b>${t.name}</b> ${refs}${by}. Generated in topological generation <b>${t.gen + 1}</b>.`;
    }

    function syncControls() {
      root.querySelectorAll("[data-v]").forEach((s) => (s.textContent = state[s.dataset.v]));
      root.querySelectorAll("[data-c]").forEach((inp) => (inp.value = state[inp.dataset.c]));
      root.querySelectorAll('[data-seg="layout"] button').forEach((b) => {
        const on = b.dataset.val === state.layout;
        b.classList.toggle("active", on);
        b.setAttribute("aria-pressed", on);
      });
    }

    root.querySelectorAll("[data-c]").forEach((inp) =>
      inp.addEventListener("input", () => { state[inp.dataset.c] = +inp.value; syncControls(); build(); draw(); })
    );
    root.querySelectorAll('[data-seg="layout"] button').forEach((b) =>
      b.addEventListener("click", () => { state.layout = b.dataset.val; syncControls(); build(); draw(); })
    );
    root.querySelector('[data-act="regen"]').addEventListener("click", () => { state.seed = PV.randomSeed(); build(); draw(); });

    syncControls();
    build();
    draw();
  };
})();
