"use strict";

(() => {
  const COPY = {
    es: {
      statePrimary: "Beta B2 · experimental · no promovida",
      stateSecondary: "Gramática 0.2 · IR 0.3 · Proyección 0.1.0 · corte soberano",
      railReference: "REFERENCIA",
      railAssurance: "GARANTÍAS",
      railDevelopment: "DESARROLLO",
      railEditor: "Editor SVP",
      railAssembly: "Ensamblaje",
      grammar: "Gramática 0.2",
      reader: "abrir lector traducible",
      programmingMetrics: "Programación y métricas",
      futureBeta: "próxima Beta",
      editorHeading: "Editor SVP",
      editorIntro: "Compilación local con perfil fuente explícito. La interfaz y el lenguaje del código permanecen independientes.",
      sourceEs: "Español · SVP-ES",
      sourceEn: "English · SVP-EN",
      ideEyebrow: "PRÓXIMA BETA · PROPUESTA",
      ideTitle: "Entorno integrado de desarrollo",
      ideText: "Este espacio reserva la futura evolución del Playground hacia proyectos, varios archivos, pestañas, símbolos, diagnósticos estructurados, representación IR y trazas. Beta B2 no implementa esas capacidades ni introduce un segundo analizador.",
      ideProject: "Proyecto",
      ideEditor: "Editor / pestañas",
      ideDiagnostics: "Diagnósticos · IR · trazas",
      buildGrammar: "Gramática",
      navLabel: "Navegación del laboratorio",
      configLabel: "Configuración"
    },
    en: {
      statePrimary: "Beta B2 · experimental · not promoted",
      stateSecondary: "Grammar 0.2 · IR 0.3 · Projection 0.1.0 · sovereign cut",
      railReference: "REFERENCE",
      railAssurance: "ASSURANCE",
      railDevelopment: "DEVELOPMENT",
      railEditor: "SVP editor",
      railAssembly: "Assembly",
      grammar: "Grammar 0.2",
      reader: "open translatable reader",
      programmingMetrics: "Programming and metrics",
      futureBeta: "next Beta",
      editorHeading: "SVP editor",
      editorIntro: "Local compilation with an explicit source profile. Interface language and source-code language remain independent.",
      sourceEs: "Spanish · SVP-ES",
      sourceEn: "English · SVP-EN",
      ideEyebrow: "NEXT BETA · PROPOSAL",
      ideTitle: "Integrated development environment",
      ideText: "This area reserves the future evolution of the Playground toward projects, multiple files, tabs, symbols, structured diagnostics, IR representation and traces. Beta B2 does not implement those capabilities and does not introduce a second parser.",
      ideProject: "Project",
      ideEditor: "Editor / tabs",
      ideDiagnostics: "Diagnostics · IR · traces",
      buildGrammar: "Grammar",
      navLabel: "Laboratory navigation",
      configLabel: "Configuration"
    }
  };

  const q = selector => document.querySelector(selector);
  const qa = selector => [...document.querySelectorAll(selector)];

  function setText(selector, value) {
    const node = q(selector);
    if (node) node.textContent = value;
  }

  function setReference(fileName, title, readerText) {
    for (const link of qa("a.rail-link")) {
      if (link.querySelector("small code")?.textContent?.trim() !== fileName) continue;
      const strong = link.querySelector("strong");
      const span = link.querySelector("span");
      if (strong) strong.textContent = title;
      if (span) span.textContent = readerText;
    }
  }

  function setIdeButton(label) {
    const button = q('[data-workspace-target="ide"]');
    if (!button) return;
    button.replaceChildren(document.createTextNode("IDE "));
    const span = document.createElement("span");
    span.textContent = label;
    button.append(span);
  }

  function apply() {
    const lang = document.documentElement.lang === "en" ? "en" : "es";
    const c = COPY[lang];

    setText(".beta-state-line strong", c.statePrimary);
    const secondary = q(".beta-state-line span");
    if (secondary) {
      const code = secondary.querySelector("code")?.textContent ?? "f6b704e819e7…";
      secondary.replaceChildren(document.createTextNode(`${c.stateSecondary} `));
      const codeNode = document.createElement("code");
      codeNode.textContent = code;
      secondary.append(codeNode);
    }

    setText('.rail-group:nth-of-type(2) .rail-heading', c.railReference);
    setText('.rail-group:nth-of-type(3) .rail-heading', c.railAssurance);
    setText('.rail-group:nth-of-type(4) .rail-heading', c.railDevelopment);
    setText('[data-workspace-target="editor"]', c.railEditor);
    setText('[data-workspace-target="assembly"]', c.railAssembly);
    setIdeButton(c.futureBeta);

    setReference("GRAMATICA_SUPERFICIAL_MINIMA_SV_v0_2.md", c.grammar, c.reader);
    setReference("IR_CANONICA_BIENFORMACION_SV_v0_3.md", "IR 0.3", c.reader);
    setText('a.rail-link[href="programacion-metricas.html"]', c.programmingMetrics);

    setText("#view-editor .view-heading h2", c.editorHeading);
    setText("#view-editor .view-heading p:not(.eyebrow)", c.editorIntro);

    const esProfile = q('[data-profile-choice="es"]');
    const enProfile = q('[data-profile-choice="en"]');
    if (esProfile) esProfile.textContent = c.sourceEs;
    if (enProfile) enProfile.textContent = c.sourceEn;

    setText("#view-ide .eyebrow", c.ideEyebrow);
    setText("#view-ide h2", c.ideTitle);
    setText("#view-ide > .ide-shell-placeholder > p:not(.eyebrow)", c.ideText);
    const ideCells = qa("#view-ide .ide-placeholder-grid div");
    if (ideCells[0]) ideCells[0].textContent = c.ideProject;
    if (ideCells[1]) ideCells[1].textContent = c.ideEditor;
    if (ideCells[2]) ideCells[2].textContent = c.ideDiagnostics;

    const grammarDt = q("#build-grammar")?.closest("div")?.querySelector("dt");
    if (grammarDt) grammarDt.textContent = c.buildGrammar;

    const rail = q(".laboratory-rail");
    if (rail) rail.setAttribute("aria-label", c.navLabel);
    const controls = q(".control-panel");
    if (controls) controls.setAttribute("aria-label", c.configLabel);
  }

  const observer = new MutationObserver(mutations => {
    if (mutations.some(mutation => mutation.type === "attributes" && mutation.attributeName === "lang")) apply();
  });
  observer.observe(document.documentElement, { attributes: true, attributeFilter: ["lang"] });

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", apply, { once: true });
  else apply();
})();
