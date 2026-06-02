/* ============================================================================
   PluRel interactive widgets - shared library (classic script, global `PV`).

   These widgets reproduce the *logic* of the PluRel generator (plurel/dag.py,
   plurel/bipartite.py, plurel/ts.py, plurel/scm.py), not numpy's exact bit
   stream. A small seeded PRNG (mulberry32) makes every "Regenerate" reproducible.
   ========================================================================== */
(function () {
  "use strict";

  // ---- seeded PRNG (mulberry32) + sampling helpers -------------------------
  class RNG {
    constructor(seed = 42) {
      this._state = (seed >>> 0) || 1;
      this._spare = null;
    }
    next() {
      this._state = (this._state + 0x6d2b79f5) >>> 0;
      let t = this._state;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    }
    randint(max) { return Math.floor(this.next() * max); }           // [0, max)
    uniform(lo, hi) { return lo + (hi - lo) * this.next(); }
    choice(arr) { return arr[this.randint(arr.length)]; }
    // standard normal via Box–Muller (cached spare)
    randn() {
      if (this._spare !== null) { const s = this._spare; this._spare = null; return s; }
      let u = 0, v = 0;
      while (u === 0) u = this.next();
      while (v === 0) v = this.next();
      const mag = Math.sqrt(-2 * Math.log(u));
      this._spare = mag * Math.sin(2 * Math.PI * v);
      return mag * Math.cos(2 * Math.PI * v);
    }
    gaussian(mean = 0, std = 1) { return mean + std * this.randn(); }
    // sample k distinct items (np.random.choice replace=False)
    sampleWithoutReplacement(arr, k) {
      const pool = arr.slice(), out = [];
      for (let i = 0; i < k && pool.length; i++) out.push(pool.splice(this.randint(pool.length), 1)[0]);
      return out;
    }
    shuffle(arr) {
      const a = arr.slice();
      for (let i = a.length - 1; i > 0; i--) { const j = this.randint(i + 1); [a[i], a[j]] = [a[j], a[i]]; }
      return a;
    }
    // Marsaglia–Tsang gamma(shape, 1), then Beta(a,b) = X/(X+Y)
    gamma(shape) {
      if (shape < 1) return this.gamma(shape + 1) * Math.pow(this.next() || 1e-12, 1 / shape);
      const d = shape - 1 / 3, c = 1 / Math.sqrt(9 * d);
      for (;;) {
        let x, v;
        do { x = this.randn(); v = 1 + c * x; } while (v <= 0);
        v = v * v * v;
        const u = this.next();
        if (u < 1 - 0.0331 * x * x * x * x) return d * v;
        if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v;
      }
    }
    beta(a, b) { const x = this.gamma(a), y = this.gamma(b); return x / (x + y); }
  }

  const clamp = (x, lo, hi) => Math.min(Math.max(x, lo), hi);

  // ---- topological layering (Kahn) ----------------------------------------
  // edges: [[u, v], ...] meaning u -> v (u parent/referenced, v child/has-FK).
  // Returns { generations: [[ids],...], inDeg, outDeg }.
  function topoGenerations(n, edges) {
    const inDeg = new Array(n).fill(0), outDeg = new Array(n).fill(0);
    const adj = Array.from({ length: n }, () => []);
    for (const [u, v] of edges) { adj[u].push(v); inDeg[v]++; outDeg[u]++; }
    const remaining = inDeg.slice();
    const seen = new Array(n).fill(false);
    const generations = [];
    let frontier = [];
    for (let i = 0; i < n; i++) if (remaining[i] === 0) frontier.push(i);
    while (frontier.length) {
      frontier.sort((a, b) => a - b);
      generations.push(frontier.slice());
      const next = [];
      for (const u of frontier) {
        seen[u] = true;
        for (const v of adj[u]) { if (--remaining[v] === 0 && !seen[v]) next.push(v); }
      }
      frontier = next;
    }
    // any leftover (shouldn't happen for a DAG) appended as a final layer
    const left = [];
    for (let i = 0; i < n; i++) if (!seen[i]) left.push(i);
    if (left.length) generations.push(left);
    return { generations, inDeg, outDeg };
  }

  // ---- random graph generators (one per layout) ---------------------------
  // All return a *DAG* edge list [[u, v], ...] with u -> v. The character of
  // each layout mirrors plurel/dag.py; the exact wiring is RNG-approximate.
  function ensureConnected(n, edgeSet, rng) {
    // union-find over undirected edges; stitch components low->high
    const parent = Array.from({ length: n }, (_, i) => i);
    const find = (x) => { while (parent[x] !== x) { parent[x] = parent[parent[x]]; x = parent[x]; } return x; };
    const union = (a, b) => { parent[find(a)] = find(b); };
    for (const k of edgeSet) { const [u, v] = k.split(",").map(Number); union(u, v); }
    let prevRoot = null;
    for (let i = 0; i < n; i++) {
      const r = find(i);
      if (prevRoot !== null && find(prevRoot) !== r) {
        const a = Math.min(prevRoot, i), b = Math.max(prevRoot, i);
        edgeSet.add(a + "," + b); union(prevRoot, i);
      }
      prevRoot = i;
    }
  }

  function barabasiAlbert(n, rng, m = 2, sinkDropout = 0.4) {
    const edges = new Set();
    m = Math.min(m, n - 1);
    const targets = [];
    for (let i = 0; i < m; i++) targets.push(i);
    const repeated = []; // preferential-attachment bag
    for (let v = m; v < n; v++) {
      for (const t of targets) { const a = Math.min(v, t), b = Math.max(v, t); edges.add(a + "," + b); repeated.push(t, v); }
      // choose next targets weighted by degree
      const chosen = new Set();
      let guard = 0;
      while (chosen.size < m && guard++ < 200) chosen.add(repeated.length ? rng.choice(repeated) : rng.randint(v));
      targets.length = 0; for (const c of chosen) targets.push(c);
    }
    ensureConnected(n, edges, rng);
    let E = [...edges].map((k) => k.split(",").map(Number));
    // sparsify_leaves (plurel/dag.py): a sink (out-degree 0) with >1 incoming
    // edge drops one random incoming edge with prob ba_sink_edge_dropout (0.4).
    const outDeg = new Array(n).fill(0), inIdx = Array.from({ length: n }, () => []);
    E.forEach((e, idx) => { outDeg[e[0]]++; inIdx[e[1]].push(idx); });
    const drop = new Set();
    for (let node = 0; node < n; node++)
      if (outDeg[node] === 0 && inIdx[node].length > 1 && rng.next() < sinkDropout)
        drop.add(inIdx[node][rng.randint(inIdx[node].length)]);
    return E.filter((_, idx) => !drop.has(idx));
  }

  function wattsStrogatz(n, rng, k = 2, p = 0.2) {
    k = Math.max(2, Math.min(k, n - 1));
    const edges = new Set();
    const key = (a, b) => Math.min(a, b) + "," + Math.max(a, b);
    const half = Math.floor(k / 2);
    for (let i = 0; i < n; i++) {
      for (let j = 1; j <= half; j++) {
        let v = (i + j) % n;
        // rewire one endpoint with prob p, preserving edge count (like
        // nx.watts_strogatz_graph: keep the original edge if no valid target).
        if (rng.next() < p) {
          let w = rng.randint(n), tries = 0;
          while ((w === i || edges.has(key(i, w))) && tries++ < 25) w = rng.randint(n);
          if (w !== i && !edges.has(key(i, w))) v = w;
        }
        edges.add(key(i, v));
      }
    }
    ensureConnected(n, edges, rng);
    return [...edges].map((s) => s.split(",").map(Number));
  }

  // Uniform random labeled tree via Prüfer-sequence decoding - the same
  // distribution as networkx.random_labeled_tree used by plurel/dag.py.
  function uniformLabeledTree(n, rng) {
    if (n < 2) return [];
    const seq = Array.from({ length: n - 2 }, () => rng.randint(n));
    const degree = new Array(n).fill(1);
    for (const x of seq) degree[x]++;
    const leaves = []; // kept sorted ascending
    for (let i = 0; i < n; i++) if (degree[i] === 1) leaves.push(i);
    const edges = [];
    for (const x of seq) {
      const leaf = leaves.shift();
      edges.push([leaf, x]);
      degree[leaf]--;
      if (--degree[x] === 1) {
        let k = 0; while (k < leaves.length && leaves[k] < x) k++;
        leaves.splice(k, 0, x);
      }
    }
    const rem = [];
    for (let i = 0; i < n; i++) if (degree[i] === 1) rem.push(i);
    edges.push([rem[0], rem[1]]);
    return edges;
  }

  // random tree, edges oriented *toward* a root (mirrors ReverseRandomTree)
  function reverseRandomTree(n, rng) {
    if (n < 2) return [];
    const adj = Array.from({ length: n }, () => []);
    for (const [a, b] of uniformLabeledTree(n, rng)) { adj[a].push(b); adj[b].push(a); }
    const root = rng.randint(n);
    const parent = new Array(n).fill(-1);
    const order = [root]; parent[root] = root;
    const seen = new Array(n).fill(false); seen[root] = true;
    for (let h = 0; h < order.length; h++) {
      const u = order[h];
      for (const w of adj[u]) if (!seen[w]) { seen[w] = true; parent[w] = u; order.push(w); }
    }
    // edge child -> rooted-parent : points toward root (root becomes a sink/activity)
    const edges = [];
    for (let i = 0; i < n; i++) if (parent[i] !== i) edges.push([i, parent[i]]);
    return edges;
  }

  // random tree, edges point *away* from root (mirrors RandomTree)
  function randomTree(n, rng) {
    return reverseRandomTree(n, rng).map(([u, v]) => [v, u]);
  }

  function erdosRenyi(n, rng, p = 0.45) {
    const edges = new Set();
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) if (rng.next() < p) edges.add(i + "," + j);
    ensureConnected(n, edges, rng);
    return [...edges].map((s) => s.split(",").map(Number));
  }

  function layered(n, rng, depth = null, pDrop = 0.15) {
    if (n < 2) return [];
    depth = depth || clamp(2 + rng.randint(Math.max(1, Math.floor(n / 2))), 2, n);
    const sizes = new Array(depth).fill(1);
    for (let r = 0; r < n - depth; r++) sizes[rng.randint(depth)]++;
    const layers = []; let cur = 0;
    for (const s of sizes) { layers.push(Array.from({ length: s }, (_, i) => cur + i)); cur += s; }
    const edges = [];
    for (let l = 0; l < layers.length - 1; l++) {
      for (const u of layers[l]) for (const v of layers[l + 1]) if (rng.next() >= pDrop) edges.push([u, v]);
      // guarantee each child has a parent and each parent a child
      for (const v of layers[l + 1]) if (!edges.some((e) => e[1] === v)) edges.push([rng.choice(layers[l]), v]);
      for (const u of layers[l]) if (!edges.some((e) => e[0] === u)) edges.push([u, rng.choice(layers[l + 1])]);
    }
    return edges;
  }

  function genGraphEdges(n, layout, rng) {
    switch (layout) {
      case "BarabasiAlbert": return barabasiAlbert(n, rng);
      case "WattsStrogatz": return wattsStrogatz(n, rng, 4, rng.uniform(0.25, 0.45));
      case "ReverseRandomTree": return reverseRandomTree(n, rng);
      case "RandomTree": return randomTree(n, rng);
      case "ErdosRenyi": return erdosRenyi(n, rng, rng.uniform(0.35, 0.6));
      case "Layered": return layered(n, rng);
      default: return barabasiAlbert(n, rng);
    }
  }

  // ---- tiny SVG helpers ----------------------------------------------------
  // `fill: context-stroke` makes the arrowhead inherit the edge's own stroke
  // colour (so highlighted/colored edges get matching heads). userSpaceOnUse
  // keeps the head a fixed pixel size regardless of stroke width.
  function addArrowMarker(defs, id, color = "context-stroke", size = 9) {
    defs.append("marker").attr("id", id).attr("viewBox", "0 0 10 10")
      .attr("refX", 8.5).attr("refY", 5).attr("markerWidth", size).attr("markerHeight", size)
      .attr("markerUnits", "userSpaceOnUse").attr("orient", "auto-start-reverse")
      .append("path").attr("d", "M1,1 L9,5 L1,9 z").attr("fill", color);
  }

  // curved edge path between two points (slight bow)
  function edgePath(x1, y1, x2, y2, r1 = 0, r2 = 0, bow = 0) {
    let dx = x2 - x1, dy = y2 - y1, L = Math.hypot(dx, dy) || 1;
    const ux = dx / L, uy = dy / L;
    const sx = x1 + ux * r1, sy = y1 + uy * r1;
    const ex = x2 - ux * r2, ey = y2 - uy * r2;
    const mx = (sx + ex) / 2 - uy * bow, my = (sy + ey) / 2 + ux * bow;
    return `M${sx},${sy} Q${mx},${my} ${ex},${ey}`;
  }

  function randomSeed() { return (Math.floor(Math.random() * 1e9) % 100000) + 1; }

  // power-law integer in [lo, hi] (probs ∝ 1/k^exp) - mirrors Choices.sample_pl,
  // which plurel/schema.py uses for a table's column count (most tables narrow).
  function samplePowerLawInt(lo, hi, rng, exp = 1) {
    let tot = 0; const w = [];
    for (let k = lo; k <= hi; k++) { const p = 1 / Math.pow(k, exp); w.push(p); tot += p; }
    let u = rng.next() * tot;
    for (let k = lo; k <= hi; k++) { u -= w[k - lo]; if (u <= 0) return k; }
    return hi;
  }

  window.PV = {
    RNG, clamp, topoGenerations, genGraphEdges, addArrowMarker, edgePath, randomSeed, samplePowerLawInt,
    layouts: { barabasiAlbert, wattsStrogatz, reverseRandomTree, randomTree, erdosRenyi, layered },
  };
})();
