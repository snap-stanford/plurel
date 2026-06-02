/* Mount the three PluRel stage widgets into their containers. Classic script;
   runs after common.js + stage{1,2,3}.js have defined PV.mountStage{1,2,3}. */
(function () {
  "use strict";
  const MOUNTS = {
    "pv-stage1": "mountStage1",
    "pv-stage2": "mountStage2",
    "pv-stage3": "mountStage3",
  };

  function mountAll() {
    const PV = window.PV || {};
    for (const [id, fn] of Object.entries(MOUNTS)) {
      const el = document.getElementById(id);
      if (!el) continue;
      try {
        if (typeof PV[fn] !== "function") throw new Error(`${fn} not defined`);
        PV[fn](el);
      } catch (err) {
        console.error(`PluRel widget #${id} failed to mount:`, err);
        el.innerHTML = `<div class="pv-widget"><p class="pv-note">This interactive demo failed to load. It needs JavaScript and a network connection (for D3). See the console for details.</p></div>`;
      }
    }
  }

  // Widow/runt control: bind the last few words of each text block with
  // non-breaking spaces so the final line can't be a 1- or 2-word runt.
  // (text-wrap: pretty handles 1-word widows; this also catches 2-word ones.)
  function deWidow(scope) {
    const sel = ".d-article p, .d-article li, .d-article figcaption, .pv-note, .pv-mech";
    scope.querySelectorAll(sel).forEach((el) => {
      const walker = document.createTreeWalker(el, NodeFilter.SHOW_TEXT);
      let last = null, n;
      while ((n = walker.nextNode())) if (n.nodeValue && n.nodeValue.trim()) last = n;
      if (!last) return;
      // keep the last three words together (bind the final two inter-word spaces)
      const bound = last.nodeValue.replace(/\s+$/, "").replace(/ (\S+) (\S+)$/, " $1 $2");
      if (bound !== last.nodeValue) last.nodeValue = bound;
    });
  }

  function init() {
    mountAll();
    deWidow(document);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
