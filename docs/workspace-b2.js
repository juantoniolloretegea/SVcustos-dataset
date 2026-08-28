"use strict";

(() => {
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
