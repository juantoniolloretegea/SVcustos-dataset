"use strict";

const ERROR_FLAG = 1n << 63n;
const PTR_MASK = 0xffff_ffffn;
const LEN_MASK = 0x7fff_ffffn;
const encoder = new TextEncoder();
const decoder = new TextDecoder("utf-8", { fatal: true });

const I18N = {
  es: {
    laboratory: "LABORATORIO SV",
    title: "Lenguaje de Computación SV",
    subtitle: "Entorno experimental de programación y verificación en navegador.",
    stableEnvironment: "Entorno estable",
    canonicalGithub: "GitHub canónico",
    experimentalTitle: "Entorno experimental.",
    experimentalText: "Las funciones aquí expuestas están sometidas a validación y no constituyen por sí solas una versión normativa del Lenguaje SV.",
    configuration: "Configuración",
    interfaceLanguage: "Interfaz",
    sourceLanguage: "Código SV",
    activeProfile: "Perfil de código activo",
    profileExplanationEs: "El compilador interpretará el texto como SVP-ES. Cambiar el perfil no traduce ni modifica el contenido del editor.",
    sourceCode: "Código fuente",
    editor: "Editor SVP",
    fileName: "Archivo",
    download: "Descargar .svp",
    selectedProfile: "Perfil",
    run: "Ejecutar",
    clearOutput: "Limpiar resultado",
    execution: "Ejecución local",
    result: "Resultado",
    notExecuted: "Sin ejecutar",
    library: "Biblioteca",
    examples: "Ejemplos",
    examplesHelp: "Los ejemplos se cargan en el editor sólo cuando usted lo solicita. Cambiar el perfil no transforma el código existente.",
    architectureTitle: "Una sola identidad computacional",
    surfaceEs: "superficie española",
    surfaceEn: "superficie inglesa",
    canonicalIds: "identidades canónicas",
    singleCore: "analizador e IR únicos",
    sameCanonicalIds: "las mismas identidades canónicas",
    sameCore: "el mismo analizador e IR",
    scalabilityClosed: "La arquitectura permite estudiar perfiles lingüísticos adicionales sin duplicar el núcleo semántico. En este entorno sólo SVP-ES y SVP-EN están constituidos. Ningún otro perfil forma parte del Lenguaje SV por el mero hecho de ser técnicamente realizable.",
    reproducibility: "Identidad del experimento",
    sourceCut: "Corte fuente",
    wasmSize: "Tamaño WebAssembly",
    grammarVersion: "Gramática",
    footerText: "La interfaz web transporta bytes e invoca el módulo WebAssembly local. No interpreta ni modifica la semántica SV.",
    loadExample: "Cargar en el editor",
    requiredProfile: "Perfil requerido",
    loadedExample: "Ejemplo cargado",
    profileChanged: "Perfil de código cambiado. El contenido del editor no se ha traducido ni modificado.",
    uiChanged: "Idioma de la interfaz cambiado. El código y su perfil permanecen inalterados.",
    running: "Ejecutando",
    admitted: "Admitido",
    rejected: "No admitido",
    executedAs: "Ejecutado como",
    wasmLoading: "Cargando WebAssembly…",
    wasmReady: "WebAssembly preparado.",
    wasmError: "No se pudo inicializar WebAssembly",
    sourceHashMismatch: "Advertencia: el SHA-256 devuelto por la compilación no coincide con los bytes actuales del editor.",
    outputCleared: "Resultado limpiado.",
    downloadReady: "Fuente descargada sin traducción ni transformación.",
    descriptionCell: "Declaración mínima de codominio, semántica y especificación de celda.",
    descriptionGate: "Compuerta con tabla de admisibilidad y operación de evaluación.",
    descriptionResolve: "Resolución constituida y consulta de campos de la operación.",
    descriptionCompose: "Composición de estructuras bajo el régimen de la gramática vigente.",
    descriptionQuery: "Consultas y contextos tipados reconocidos por el Lenguaje.",
    descriptionTrajectory: "Trayectoria y alternancia válida en el corpus de conformidad."
  },
  en: {
    laboratory: "SV LABORATORY",
    title: "SV Computing Language",
    subtitle: "Experimental in-browser programming and verification environment.",
    stableEnvironment: "Stable environment",
    canonicalGithub: "Canonical GitHub",
    experimentalTitle: "Experimental environment.",
    experimentalText: "The functions exposed here are under validation and do not by themselves constitute a normative version of the SV Language.",
    configuration: "Configuration",
    interfaceLanguage: "Interface",
    sourceLanguage: "SV code",
    activeProfile: "Active source profile",
    profileExplanationEs: "The compiler will interpret the text under the selected SVP profile. Changing the profile does not translate or modify the editor contents.",
    sourceCode: "Source code",
    editor: "SVP editor",
    fileName: "File",
    download: "Download .svp",
    selectedProfile: "Profile",
    run: "Run",
    clearOutput: "Clear result",
    execution: "Local execution",
    result: "Result",
    notExecuted: "Not executed",
    library: "Library",
    examples: "Examples",
    examplesHelp: "Examples are loaded into the editor only when you request it. Changing the profile never transforms existing code.",
    architectureTitle: "One computational identity",
    surfaceEs: "Spanish surface",
    surfaceEn: "English surface",
    canonicalIds: "canonical identities",
    singleCore: "single parser and IR",
    sameCanonicalIds: "the same canonical identities",
    sameCore: "the same parser and IR",
    scalabilityClosed: "The architecture can support controlled study of additional linguistic profiles without duplicating the semantic core. Only SVP-ES and SVP-EN are constituted in this environment. No other profile becomes part of the SV Language merely because it is technically feasible.",
    reproducibility: "Experiment identity",
    sourceCut: "Source cut",
    wasmSize: "WebAssembly size",
    grammarVersion: "Grammar",
    footerText: "The web interface transports bytes and invokes the local WebAssembly module. It does not interpret or modify SV semantics.",
    loadExample: "Load into editor",
    requiredProfile: "Required profile",
    loadedExample: "Example loaded",
    profileChanged: "Source profile changed. The editor contents were not translated or modified.",
    uiChanged: "Interface language changed. The source code and its profile remain unchanged.",
    running: "Running",
    admitted: "Admitted",
    rejected: "Not admitted",
    executedAs: "Executed as",
    wasmLoading: "Loading WebAssembly…",
    wasmReady: "WebAssembly ready.",
    wasmError: "WebAssembly could not be initialized",
    sourceHashMismatch: "Warning: the SHA-256 returned by compilation does not match the current editor bytes.",
    outputCleared: "Result cleared.",
    downloadReady: "Source downloaded without translation or transformation.",
    descriptionCell: "Minimal codomain, output-semantics and cell-specification declaration.",
    descriptionGate: "Gate with admissibility table and evaluation operation.",
    descriptionResolve: "Constituted resolution and access to operation fields.",
    descriptionCompose: "Composition of structures under the current grammar regime.",
    descriptionQuery: "Typed queries and contexts recognized by the Language.",
    descriptionTrajectory: "Trajectory and valid alternation from the conformance corpus."
  }
};

const state = {
  ui: "es",
  profile: "es",
  examples: [],
  wasm: null,
  currentExample: null,
  hashSequence: 0
};

const el = {};

function t(key) {
  return I18N[state.ui][key] ?? I18N.es[key] ?? key;
}

function profileLong(profile = state.profile) {
  if (profile === "es") {
    return state.ui === "es" ? "ESPAÑOL · SVP-ES" : "SPANISH · SVP-ES";
  }
  return state.ui === "es" ? "INGLÉS · SVP-EN" : "ENGLISH · SVP-EN";
}

function profileShort(profile = state.profile) {
  return profile === "es" ? "SVP-ES" : "SVP-EN";
}

function setStatus(message) {
  el.status.textContent = message;
}

function setResultState(kind, label) {
  el.resultState.dataset.state = kind;
  el.resultState.textContent = label;
}

function applyTranslations() {
  document.documentElement.lang = state.ui;
  document.body.dataset.uiLang = state.ui;
  for (const node of document.querySelectorAll("[data-i18n]")) {
    node.textContent = t(node.dataset.i18n);
  }
  for (const button of document.querySelectorAll("[data-ui-choice]")) {
    button.setAttribute("aria-pressed", String(button.dataset.uiChoice === state.ui));
  }
  el.profileLabel.textContent = profileLong();
  el.editorProfileChip.textContent = profileShort();
  el.examplesProfileChip.textContent = profileShort();
  renderExamples();
}

function setUiLanguage(lang) {
  if (!Object.hasOwn(I18N, lang) || lang === state.ui) return;
  const sourceBefore = el.editor.value;
  const profileBefore = state.profile;
  state.ui = lang;
  applyTranslations();
  if (el.editor.value !== sourceBefore || state.profile !== profileBefore) {
    throw new Error("la interfaz no puede modificar fuente ni perfil");
  }
  setStatus(t("uiChanged"));
}

function setSourceProfile(profile) {
  if (profile !== "es" && profile !== "en") return;
  if (profile === state.profile) return;
  const sourceBefore = el.editor.value;
  state.profile = profile;
  document.body.dataset.sourceProfile = profile;
  for (const button of document.querySelectorAll("[data-profile-choice]")) {
    button.setAttribute("aria-pressed", String(button.dataset.profileChoice === profile));
  }
  el.profileLabel.textContent = profileLong();
  el.editorProfileChip.textContent = profileShort();
  el.examplesProfileChip.textContent = profileShort();
  el.sourceProfileCode.textContent = profile;
  renderExamples();
  if (el.editor.value !== sourceBefore) {
    throw new Error("cambiar el perfil no puede modificar la fuente");
  }
  setStatus(t("profileChanged"));
  setResultState("idle", t("notExecuted"));
}

function exampleDescription(example) {
  return t(example.description_key);
}

function renderExamples() {
  if (!el.examplesList) return;
  el.examplesList.replaceChildren();
  for (const example of state.examples) {
    const card = document.createElement("article");
    card.className = "example-card";
    const h3 = document.createElement("h3");
    h3.textContent = state.ui === "es" ? example.title_es : example.title_en;
    const p = document.createElement("p");
    p.textContent = `${exampleDescription(example)} · ${t("requiredProfile")}: ${profileShort()}`;
    const button = document.createElement("button");
    button.type = "button";
    button.className = "secondary";
    button.textContent = t("loadExample");
    button.addEventListener("click", () => loadExample(example));
    card.append(h3, p, button);
    el.examplesList.append(card);
  }
}

function loadExample(example) {
  const source = state.profile === "es" ? example.source_es : example.source_en;
  el.editor.value = source;
  el.fileName.value = example.file_name;
  state.currentExample = example.id;
  scheduleHash();
  setResultState("idle", t("notExecuted"));
  setStatus(`${t("loadedExample")}: ${state.ui === "es" ? example.title_es : example.title_en} · ${profileShort()}.`);
}

async function sha256(text) {
  if (!globalThis.crypto?.subtle) return "—";
  const digest = await crypto.subtle.digest("SHA-256", encoder.encode(text));
  return [...new Uint8Array(digest)].map(byte => byte.toString(16).padStart(2, "0")).join("");
}

function scheduleHash() {
  const sequence = ++state.hashSequence;
  const source = el.editor.value;
  setTimeout(async () => {
    const hash = await sha256(source);
    if (sequence === state.hashSequence) el.sourceSha.textContent = hash;
  }, 70);
}

function unpackResult(packed) {
  const value = BigInt(packed);
  return {
    error: (value & ERROR_FLAG) !== 0n,
    ptr: Number(value & PTR_MASK),
    len: Number((value >> 32n) & LEN_MASK)
  };
}

function writeInternalBuffer(exports, exportName, text) {
  const bytes = encoder.encode(text);
  const ptr = exports[exportName](bytes.length);
  new Uint8Array(exports.memory.buffer, ptr, bytes.length).set(bytes);
}

async function initializeWasm() {
  setStatus(t("wasmLoading"));
  const response = await fetch("sv_wasm.wasm", { cache: "no-store" });
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  const bytes = await response.arrayBuffer();
  const { instance } = await WebAssembly.instantiate(bytes, {});
  const exports = instance.exports;
  for (const name of ["memory", "sv_source_buffer", "sv_file_buffer", "sv_compile_svp_json_profile"]) {
    if (!(name in exports)) throw new Error(`export WebAssembly ausente: ${name}`);
  }
  state.wasm = exports;
  el.buildGrammar.textContent = `${exports.sv_grammar_version_major()}.${exports.sv_grammar_version_minor()}`;
  el.buildIr.textContent = `${exports.sv_ir_version_major()}.${exports.sv_ir_version_minor()}`;
  setStatus(t("wasmReady"));
}

async function runSource() {
  setResultState("running", t("running"));
  try {
    if (!state.wasm) await initializeWasm();
    const source = el.editor.value;
    const fileName = el.fileName.value.trim() || "programa.svp";
    const profileCode = state.profile === "es" ? 1 : 0;
    writeInternalBuffer(state.wasm, "sv_source_buffer", source);
    writeInternalBuffer(state.wasm, "sv_file_buffer", fileName);
    const result = unpackResult(state.wasm.sv_compile_svp_json_profile(profileCode));
    const bytes = new Uint8Array(state.wasm.memory.buffer, result.ptr, result.len);
    const raw = decoder.decode(bytes.slice());
    let formatted = raw;
    let returnedHash = null;
    try {
      const parsed = JSON.parse(raw);
      returnedHash = parsed.source_sha256 ?? null;
      formatted = JSON.stringify(parsed, null, 2);
    } catch (_) {
      // Los diagnósticos de error pueden no ser JSON; se presentan sin reinterpretación.
    }
    el.output.textContent = formatted;
    const localHash = await sha256(source);
    el.sourceSha.textContent = localHash;
    if (returnedHash && localHash !== "—" && returnedHash !== localHash) {
      setStatus(t("sourceHashMismatch"));
    } else {
      setStatus(`${t("executedAs")} ${profileLong()}.`);
    }
    setResultState(result.error ? "fail" : "pass", result.error ? t("rejected") : t("admitted"));
  } catch (error) {
    el.output.textContent = String(error && error.stack ? error.stack : error);
    setResultState("fail", t("rejected"));
    setStatus(`${t("wasmError")}: ${String(error)}`);
  }
}

function clearOutput() {
  el.output.textContent = "";
  setResultState("idle", t("notExecuted"));
  setStatus(t("outputCleared"));
}

function downloadSource() {
  const source = el.editor.value;
  const requested = el.fileName.value.trim() || "programa.svp";
  const fileName = requested.endsWith(".svp") ? requested : `${requested}.svp`;
  const blob = new Blob([source], { type: "text/plain;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = fileName;
  document.body.append(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
  setStatus(`${t("downloadReady")} ${profileShort()}.`);
}

async function loadBuildInfo() {
  const response = await fetch("build-info.json", { cache: "no-store" });
  if (!response.ok) return;
  const info = await response.json();
  el.buildSourceCommit.textContent = info.source_commit ?? "—";
  el.buildWasmSha.textContent = info.wasm_sha256 ?? "—";
  el.buildWasmBytes.textContent = info.wasm_bytes != null ? `${info.wasm_bytes} bytes` : "—";
}

async function loadExamples() {
  const response = await fetch("examples.json", { cache: "no-store" });
  if (!response.ok) throw new Error(`no se pudieron cargar ejemplos: HTTP ${response.status}`);
  state.examples = await response.json();
  renderExamples();
  if (state.examples.length > 0 && !el.editor.value) loadExample(state.examples[0]);
}

function bind() {
  Object.assign(el, {
    editor: document.getElementById("source-editor"),
    fileName: document.getElementById("file-name"),
    sourceSha: document.getElementById("source-sha"),
    sourceProfileCode: document.getElementById("source-profile-code"),
    status: document.getElementById("editor-status"),
    profileLabel: document.getElementById("profile-label"),
    editorProfileChip: document.getElementById("editor-profile-chip"),
    examplesProfileChip: document.getElementById("examples-profile-chip"),
    examplesList: document.getElementById("examples-list"),
    resultState: document.getElementById("result-state"),
    output: document.getElementById("result-output"),
    buildSourceCommit: document.getElementById("build-source-commit"),
    buildWasmSha: document.getElementById("build-wasm-sha"),
    buildWasmBytes: document.getElementById("build-wasm-bytes"),
    buildGrammar: document.getElementById("build-grammar"),
    buildIr: document.getElementById("build-ir")
  });

  for (const button of document.querySelectorAll("[data-ui-choice]")) {
    button.addEventListener("click", () => setUiLanguage(button.dataset.uiChoice));
  }
  for (const button of document.querySelectorAll("[data-profile-choice]")) {
    button.addEventListener("click", () => setSourceProfile(button.dataset.profileChoice));
  }
  document.getElementById("run-source").addEventListener("click", runSource);
  document.getElementById("clear-output").addEventListener("click", clearOutput);
  document.getElementById("download-source").addEventListener("click", downloadSource);
  el.editor.addEventListener("input", scheduleHash);
}

async function main() {
  bind();
  applyTranslations();
  await Promise.all([loadBuildInfo(), loadExamples(), initializeWasm()]);
  scheduleHash();
}

main().catch(error => {
  if (el.output) el.output.textContent = String(error && error.stack ? error.stack : error);
  if (el.resultState) setResultState("fail", t("rejected"));
  if (el.status) setStatus(`${t("wasmError")}: ${String(error)}`);
});
