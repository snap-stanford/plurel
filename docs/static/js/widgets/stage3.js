/* ============================================================================
   Stage 3 - Feature Generation via Structural Causal Models.
   LEFT: a temporal-signal composer. Source columns are seeded by an exogenous
   input that sums a TREND (power-law), a CYCLE (sinusoidal) and a FLUCTUATION
   (AR noise) - a faithful port of plurel/ts.py (Trend, Cycle, TSDataGenerator).
   Entity tables are flat + noisy (static); activity tables are smooth + trended.
   RIGHT: each table is an SCM - a causal DAG over columns. Non-source columns
   are computed by a projection–reconstruction mechanism: encode parents (and
   parent-table foreign features), aggregate, add Beta noise, decode to the
   target type. Hover a node or press ▶ to watch values propagate.
   ========================================================================== */
(function () {
  "use strict";
  const d3 = window.d3;
  const PV = window.PV;
  const N = 120;                       // rows for the temporal signal
  const MINV = -1, MAXV = 1;
  const TW = 470, TH = 244;            // ts plot   - same aspect as the scm graph
  const GW = 470, GH = 244, R = 15;    // scm graph - so both figures render the same size
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

  // ---- ts.py ports ---------------------------------------------------------
  function trendVal(i, alpha, scale) { return Math.min(scale * Math.pow(i / N, alpha) + MINV, MAXV); }
  function cycleVal(i, freq, scale) { return PV.clamp(scale * Math.sin((i / freq) * Math.PI), MINV, MAXV); }

  function tsSeries({ alpha, freqPerc, noiseScale, rho, trendScale, cycleScale }, rng) {
    const freq = Math.max(1, freqPerc * N);
    const trend = [], cycle = [], noise = [], sum = [];
    let state = 0;
    for (let i = 0; i < N; i++) {
      const t = trendVal(i, alpha, trendScale);
      const c = cycleVal(i, freq, cycleScale);
      state = rho * state + rng.randn() * noiseScale;
      const ns = PV.clamp(state, MINV, MAXV);
      trend.push(t); cycle.push(c); noise.push(ns); sum.push((t + c + ns) / 3);
    }
    return { trend, cycle, noise, sum };
  }

  function aggregate(vals, agg) {
    if (!vals.length) return 0;
    if (agg === "sum") return vals.reduce((a, b) => a + b, 0);
    if (agg === "max") return vals.reduce((a, b) => Math.max(a, b), -Infinity);
    if (agg === "product") return vals.reduce((a, b) => a * b, 1);
    if (agg === "logexp") return Math.log(vals.reduce((a, b) => a + Math.exp(Math.min(b, 20)), 1e-9));
    return 0;
  }

  PV.mountStage3 = function (root) {
    const state = {
      preset: "activity", alpha: 1.2, freqPerc: 0.3, noiseScale: 0.05, rho: 0.3,
      trendScale: 1, cycleScale: 1, agg: "sum", scmSeed: 5, hover: -1,
    };

    root.innerHTML = `
      <div class="pv-widget pv-stage3">
        <div class="pv-head">
          <span class="pv-badge">Stage 3 · features</span>
          <span class="pv-title">Cell values from causal mechanisms + temporal signals</span>
        </div>
        <div class="pv-split-even">
          <div>
            <div class="pv-tag">Exogenous temporal input (seeds source columns)</div>
            <div class="pv-knobbox">
              <div class="pv-controls pv-controls-compact">
                <div class="pv-control">
                  <label>table type</label>
                  <div class="pv-seg" data-seg="preset">
                    <button data-val="activity">activity</button>
                    <button data-val="entity">entity</button>
                  </div>
                </div>
                <div class="pv-control">
                  <label>trend α · <span class="pv-val" data-v="alpha"></span></label>
                  <input type="range" class="pv-range" data-c="alpha" min="0" max="2" step="0.1">
                </div>
                <div class="pv-control">
                  <label>cycle freq · <span class="pv-val" data-v="freqPerc"></span></label>
                  <input type="range" class="pv-range" data-c="freqPerc" min="0.1" max="1" step="0.1">
                </div>
                <div class="pv-control">
                  <label>noise · <span class="pv-val" data-v="noiseScale"></span></label>
                  <input type="range" class="pv-range" data-c="noiseScale" min="0" max="1" step="0.05">
                </div>
              </div>
            </div>
            <svg class="pv-ts-svg" viewBox="0 0 ${TW} ${TH}" preserveAspectRatio="xMidYMid meet" role="img" aria-label="Exogenous temporal signal over rows: trend, cycle and fluctuation components and their mean"></svg>
            <div class="pv-ts-legend">
              <span class="k-sum"><span class="solid"></span>signal (mean)</span>
              <span class="k-trend"><span class="dash"></span>trend</span>
              <span class="k-cycle"><span class="dash"></span>cycle</span>
              <span class="k-noise"><span class="dash"></span>noise</span>
            </div>
          </div>
          <div>
            <div class="pv-tag">SCM: a causal DAG over this table's columns</div>
            <div class="pv-knobbox">
              <div class="pv-controls pv-controls-compact">
                <div class="pv-control">
                  <label>aggregation</label>
                  <div class="pv-seg" data-seg="agg">
                    <button data-val="sum">sum</button>
                    <button data-val="max">max</button>
                    <button data-val="product">product</button>
                    <button data-val="logexp">logexp</button>
                  </div>
                </div>
                <div class="pv-control pv-spacer">
                  <label>&nbsp;</label>
                  <button class="pv-btn pv-btn-accent" data-act="prop">▶ Propagate</button>
                </div>
                <div class="pv-control">
                  <label>&nbsp;</label>
                  <button class="pv-btn" data-act="regen">↻ New SCM</button>
                </div>
              </div>
            </div>
            <svg class="pv-scm-svg" viewBox="0 0 ${GW} ${GH}" preserveAspectRatio="xMidYMid meet" role="img" aria-label="Structural causal model: a directed acyclic graph over a table's columns, source columns on the left"></svg>
            <div class="pv-legend">
              <span><span class="pv-sw round" style="background:var(--tok-exo-fill)"></span>source</span>
              <span><span class="pv-sw round numeric"></span>numeric</span>
              <span><span class="pv-sw round categorical"></span>categorical</span>
              <span><span class="pv-sw round" style="background:#b7c0cc"></span>latent</span>
            </div>
          </div>
        </div>
        <div class="pv-mech" data-el="mech"></div>
        <p class="pv-note">A <b>source</b> column (no parents) is seeded by an exogenous generator. With the temporal one, its value is <span style="color:var(--tok-exo);font-weight:700">(trend + cycle + fluctuation) / 3</span> from the left. Every other column is a <b>projection–reconstruction</b> mechanism: parent values (and foreign features from referenced tables) are pushed through small frozen MLPs into a shared latent space, <b>aggregated</b>, perturbed with <b>Beta noise</b>, then decoded back to a numeric or categorical value. Thicker-ringed nodes are the columns that actually appear in the table; the rest are latent.</p>
      </div>`;

    const tsSvg = d3.select(root).select(".pv-ts-svg");
    const scmSvg = d3.select(root).select(".pv-scm-svg");
    const mechEl = root.querySelector('[data-el="mech"]');
    let ts = null, scm = null, runId = 0;

    // -------- build models --------
    function buildTS() {
      ts = tsSeries(state, new PV.RNG(909));
    }

    function buildSCM() {
      const n = 9;
      const rng = new PV.RNG(state.scmSeed * 101 + 3);
      const layout = ["RandomTree", "Layered", "ErdosRenyi", "ReverseRandomTree", "BarabasiAlbert"][state.scmSeed % 5];
      const edges = PV.genGraphEdges(n, layout, rng);
      const { generations, inDeg } = PV.topoGenerations(n, edges);
      const inNbrs = Array.from({ length: n }, () => []);
      for (const [u, v] of edges) inNbrs[v].push(u);
      const genOf = new Array(n).fill(0);
      generations.forEach((g, gi) => g.forEach((id) => (genOf[id] = gi)));

      // node roles: ~60% are column nodes; non-source cols carry a type
      const nodes = [];
      for (let i = 0; i < n; i++) {
        const nr = new PV.RNG(state.scmSeed * 13 + i * 31 + 7);
        const isSource = inDeg[i] === 0;
        const isCol = nr.next() < 0.62;
        const type = nr.next() < 0.5 ? "numeric" : "categorical";
        const ncat = 2 + nr.randint(9);
        nodes.push({ id: i, gen: genOf[i], isSource, isCol, type, ncat, parents: inNbrs[i].slice().sort((a, b) => a - b) });
      }
      // layout by generation
      const G = generations.length;
      const xFor = (g) => (G === 1 ? GW / 2 : 40 + (g * (GW - 80)) / (G - 1));
      generations.forEach((g, gi) => {
        const k = g.length;
        g.forEach((id, j) => {
          nodes[id].x = xFor(gi);
          nodes[id].y = k === 1 ? GH / 2 : 36 + (j * (GH - 72)) / (k - 1);
        });
      });
      scm = { n, edges, nodes, generations, layout };
      computeValues();
    }

    function computeValues() {
      const rng = new PV.RNG(state.scmSeed * 733 + 5);
      const w = {};
      scm.edges.forEach(([u, v]) => (w[u + ">" + v] = rng.gaussian(0, 1.1)));
      const srcNodes = scm.nodes.filter((nd) => nd.isSource);
      srcNodes.forEach((nd, k) => {
        const row = Math.floor(((k + 1) / (srcNodes.length + 1)) * (N - 1));
        nd._raw = ts.sum[row];
      });
      for (const gen of scm.generations) {
        for (const id of gen) {
          const nd = scm.nodes[id];
          if (nd.isSource) { nd.val = nd._raw; }
          else {
            const terms = nd.parents.map((p) => w[p + ">" + id] * scm.nodes[p].val);
            const noise = (rng.beta(2, 2) - 0.5) * 0.35;             // Beta-noise
            nd.val = Math.tanh((aggregate(terms, state.agg) / Math.max(1, nd.parents.length)) * 0.9 + noise);
          }
          nd.display = nd.isCol && nd.type === "categorical"
            ? "cat " + Math.min(nd.ncat - 1, Math.floor(((nd.val + 1) / 2) * nd.ncat))
            : nd.val.toFixed(2);
        }
      }
    }

    function nodeClass(nd) {
      let c = "pv-scm-node ";
      c += nd.isSource ? "source" : nd.isCol ? nd.type : "latent";
      if (nd.isCol) c += " col";
      return c;
    }

    // -------- drawing --------
    function drawTS() {
      tsSvg.selectAll("*").remove();
      const M = { l: 30, r: 12, t: 10, b: 34 };
      const x = d3.scaleLinear().domain([0, N - 1]).range([M.l, TW - M.r]);
      const y = d3.scaleLinear().domain([MINV, MAXV]).range([TH - M.b, M.t]);
      tsSvg.append("g").attr("class", "pv-ts-grid")
        .selectAll("line").data(y.ticks(5)).join("line")
        .attr("x1", M.l).attr("x2", TW - M.r).attr("y1", (d) => y(d)).attr("y2", (d) => y(d));
      tsSvg.append("g").attr("class", "pv-ts-axis").attr("transform", `translate(0,${TH - M.b})`).call(d3.axisBottom(x).ticks(6).tickSize(3));
      tsSvg.append("g").attr("class", "pv-ts-axis").attr("transform", `translate(${M.l},0)`).call(d3.axisLeft(y).ticks(5).tickSize(3));
      tsSvg.append("text").attr("class", "pv-ts-axis").attr("x", (M.l + TW - M.r) / 2).attr("y", TH - 3).attr("text-anchor", "middle").attr("font-weight", 600).text("row index");
      const line = d3.line().x((d, i) => x(i)).y((d) => y(d));
      // components: dashed + light; the composed signal: solid + bold (drawn last, on top)
      tsSvg.append("path").attr("class", "pv-ts-comp pv-ts-trend").attr("d", line(ts.trend))
        .style("opacity", state.trendScale === 0 ? 0.18 : null);
      tsSvg.append("path").attr("class", "pv-ts-comp pv-ts-cycle").attr("d", line(ts.cycle))
        .style("opacity", state.cycleScale === 0 ? 0.18 : null);
      tsSvg.append("path").attr("class", "pv-ts-comp pv-ts-noise").attr("d", line(ts.noise));
      tsSvg.append("path").attr("class", "pv-ts-sum").attr("d", line(ts.sum));
    }

    function drawSCM() {
      const { edges, nodes } = scm;
      scmSvg.selectAll("*").remove();
      const defs = scmSvg.append("defs");
      PV.addArrowMarker(defs, "pv-scm-arrow");
      scmSvg.append("text").attr("class", "pv-genlabel").attr("x", 40).attr("y", 14).text("sources →");

      const gE = scmSvg.append("g");
      const gN = scmSvg.append("g");
      const edgeSel = gE.selectAll("path").data(edges).join("path").attr("class", "pv-scm-edge")
        .attr("d", ([u, v]) => PV.edgePath(nodes[u].x, nodes[u].y, nodes[v].x, nodes[v].y, R + 1, R + 6, 0));

      const nodeSel = gN.selectAll("g").data(nodes, (d) => d.id).join("g")
        .attr("class", nodeClass).attr("transform", (d) => `translate(${d.x},${d.y})`)
        .attr("tabindex", 0).attr("role", "button")
        .attr("aria-label", (d) => `column y${d.id}, ${d.isSource ? "source" : d.isCol ? d.type + " column" : "latent node"}`);
      nodeSel.append("circle").attr("r", R);
      nodeSel.append("text").attr("text-anchor", "middle").attr("dy", "-0.05em").text((d) => "y" + d.id);
      nodeSel.append("text").attr("text-anchor", "middle").attr("dy", "0.95em").attr("font-size", "7.5px")
        .attr("class", "pv-scm-valtext").text((d) => d.display);

      function setHover(id) {
        state.hover = id;
        const parents = new Set(id >= 0 ? nodes[id].parents : []);
        edgeSel.classed("hl", ([u, v]) => id >= 0 && v === id).classed("dim", ([u, v]) => id >= 0 && v !== id);
        nodeSel.classed("active", (d) => d.id === id)
          .classed("dim", (d) => id >= 0 && d.id !== id && !parents.has(d.id));
        renderMech(id);
      }
      nodeSel.on("mouseenter", (e, d) => setHover(d.id)).on("mouseleave", () => setHover(-1))
        .on("focus", (e, d) => setHover(d.id)).on("blur", () => setHover(-1))
        .on("click", (e, d) => setHover(d.id));
      scm._nodeSel = nodeSel; scm._edgeSel = edgeSel; scm._setHover = setHover;
      setHover(-1);
    }

    function renderMech(id) {
      if (id < 0) {
        mechEl.innerHTML = `Hover a column node to see how its value is produced. <b>Source</b> nodes (amber) are seeded by the temporal signal; downstream nodes combine their parents through the chosen <b>${state.agg}</b> aggregation.`;
        return;
      }
      const nd = scm.nodes[id];
      if (nd.isSource) {
        mechEl.innerHTML = `<span class="pv-pill" style="background:var(--tok-exo-fill)">source</span><b>y${id}</b>${nd.isCol ? ` · ${nd.type} column` : " · latent"}, seeded by an exogenous generator. With the temporal generator its value is
          <span class="pv-formula"><span class="hl-exo">y${id} = ( trend + cycle + fluctuation ) / 3</span>  =  ${nd.val.toFixed(3)}</span>
          drawn from the signal on the left. Sampled value: <span class="pv-val-chip">${nd.display}</span>.`;
        return;
      }
      const pstr = nd.parents.map((p) => "y" + p).join(", ");
      mechEl.innerHTML = `<b>y${id}</b>${nd.isCol ? ` · <b>${nd.type}</b> column` : " · latent node"} is a projection–reconstruction mechanism over its parents (${pstr}):
        <span class="pv-formula">y${id} = decode( <b>${state.agg}</b>( <span class="hl-parent">Σ wₑ·encode(parent)</span> ⊕ encode(foreign feats) ) + <span class="hl-noise">Beta-noise</span> )</span>
        Aggregates ${nd.parents.length} parent${nd.parents.length === 1 ? "" : "s"}, perturbs with Beta noise, decodes to ${nd.type}. Current value: <span class="pv-val-chip">${nd.display}</span>.`;
    }

    async function propagate() {
      runId++; const myId = runId;
      const btn = root.querySelector('[data-act="prop"]');
      btn.disabled = true; btn.textContent = "▶ Propagating…";
      scm._nodeSel.classed("dim", true);
      scm._edgeSel.classed("dim", true);
      for (const gen of scm.generations) {
        if (myId !== runId) { btn.disabled = false; btn.textContent = "▶ Propagate"; return; }
        scm._nodeSel.filter((d) => gen.includes(d.id)).classed("dim", false).classed("active", true);
        scm._edgeSel.filter(([u, v]) => gen.includes(v)).classed("dim", false).classed("hl", true);
        await sleep(560);
        scm._nodeSel.filter((d) => gen.includes(d.id)).classed("active", false);
        scm._edgeSel.filter(([u, v]) => gen.includes(v)).classed("hl", false);
      }
      scm._nodeSel.classed("dim", false);
      scm._edgeSel.classed("dim", false);
      btn.disabled = false; btn.textContent = "▶ Propagate";
    }

    // -------- controls --------
    function syncControls() {
      root.querySelectorAll("[data-v]").forEach((s) => (s.textContent = (+state[s.dataset.v]).toFixed(s.dataset.v === "alpha" || s.dataset.v === "freqPerc" || s.dataset.v === "rho" || s.dataset.v === "noiseScale" ? (s.dataset.v === "noiseScale" || s.dataset.v === "rho" ? 2 : 1) : 0)));
      root.querySelectorAll("[data-c]").forEach((inp) => (inp.value = state[inp.dataset.c]));
      root.querySelectorAll('[data-seg="preset"] button').forEach((b) => {
        const on = b.dataset.val === state.preset;
        b.classList.toggle("active", on); b.setAttribute("aria-pressed", on);
      });
      root.querySelectorAll('[data-seg="agg"] button').forEach((b) => {
        const on = b.dataset.val === state.agg;
        b.classList.toggle("active", on); b.setAttribute("aria-pressed", on);
      });
    }

    root.querySelectorAll("[data-c]").forEach((inp) =>
      inp.addEventListener("input", () => {
        state[inp.dataset.c] = +inp.value; syncControls();
        buildTS(); computeValues(); drawTS();
        scm._nodeSel.select(".pv-scm-valtext").text((d) => d.display);
      })
    );
    root.querySelectorAll('[data-seg="preset"] button').forEach((b) =>
      b.addEventListener("click", () => {
        state.preset = b.dataset.val;
        if (state.preset === "activity") { state.trendScale = 1; state.cycleScale = 1; state.noiseScale = 0.05; }
        else { state.trendScale = 0; state.cycleScale = 0; state.noiseScale = 1; }
        syncControls(); buildTS(); computeValues(); drawTS();
        scm._nodeSel.select(".pv-scm-valtext").text((d) => d.display);
      })
    );
    root.querySelectorAll('[data-seg="agg"] button').forEach((b) =>
      b.addEventListener("click", () => {
        state.agg = b.dataset.val; syncControls(); computeValues();
        scm._nodeSel.select(".pv-scm-valtext").text((d) => d.display);
        if (state.hover >= 0) renderMech(state.hover);
      })
    );
    root.querySelector('[data-act="prop"]').addEventListener("click", propagate);
    root.querySelector('[data-act="regen"]').addEventListener("click", () => { runId++; state.scmSeed = PV.randomSeed(); buildSCM(); drawSCM(); });

    syncControls();
    buildTS();
    buildSCM();
    drawTS();
    drawSCM();
  };
})();
