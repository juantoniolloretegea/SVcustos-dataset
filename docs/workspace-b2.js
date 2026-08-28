"use strict";

(() => {
  const stylesheet = document.createElement("link");
  stylesheet.rel = "stylesheet";
  stylesheet.href = "workspace-b2.css";
  document.head.append(stylesheet);

  const referenceTargets = {
    "GRAMATICA_SUPERFICIAL_MINIMA_SV_v0_2.md": "https://lectura-sv.itvia.online/lenguaje/?file=GRAMATICA_SUPERFICIAL_MINIMA_SV_v0_2.md",
    "IR_CANONICA_BIENFORMACION_SV_v0_3.md": "https://lectura-sv.itvia.online/lenguaje/?file=IR_CANONICA_BIENFORMACION_SV_v0_3.md"
  };
  for (const link of document.querySelectorAll("a.rail-link")) {
    const fileName = link.querySelector("small code")?.textContent?.trim();
    if (fileName && referenceTargets[fileName]) link.href = referenceTargets[fileName];
  }

  const buttons = [...document.querySelectorAll("[data-workspace-target]")];
  const views = [...document.querySelectorAll("[data-workspace-view]")];

  function activate(name, updateHash = true) {
    if (!views.some(view => view.dataset.workspaceView === name)) return;
    for (const view of views) view.hidden = view.dataset.workspaceView !== name;
    for (const button of buttons) {
      const active = button.dataset.workspaceTarget === name;
      button.classList.toggle("is-active", active);
      button.setAttribute("aria-pressed", String(active));
    }
    if (updateHash) history.replaceState(null, "", `#${name}`);
    const target = views.find(view => view.dataset.workspaceView === name);
    target?.scrollIntoView({ block: "start", behavior: "instant" });
  }

  for (const button of buttons) {
    button.addEventListener("click", () => activate(button.dataset.workspaceTarget));
  }

  const requested = location.hash.replace(/^#/, "");
  activate(["editor", "assembly", "ide"].includes(requested) ? requested : "editor", false);
})();
