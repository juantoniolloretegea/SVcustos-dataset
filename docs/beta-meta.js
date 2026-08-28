"use strict";

const BETA_LINK_LABELS = {
  es: {
    repository: "Repositorio canónico",
    features: "Características de esta versión",
    history: "Historial Beta"
  },
  en: {
    repository: "Canonical repository",
    features: "Version features",
    history: "Beta history"
  }
};

function applyBetaLinkLabels() {
  const lang = document.documentElement.lang === "en" ? "en" : "es";
  const labels = BETA_LINK_LABELS[lang];
  const repository = document.getElementById("canonical-repository-link");
  const features = document.getElementById("version-features-link");
  const history = document.getElementById("beta-history-link");
  if (repository) repository.textContent = labels.repository;
  if (features) features.textContent = labels.features;
  if (history) history.textContent = labels.history;
}

document.addEventListener("DOMContentLoaded", applyBetaLinkLabels);
new MutationObserver(applyBetaLinkLabels).observe(document.documentElement, {
  attributes: true,
  attributeFilter: ["lang"]
});
