"use strict";

(() => {
  const nativeFetch = window.fetch.bind(window);

  window.fetch = (input, init) => {
    const url = typeof input === "string" ? input : (input?.url ?? "");
    if (url.endsWith("sv_wasm_b2.wasm.gz.b64")) {
      return nativeFetch("sv_wasm_b2.wasm.gz.b64", { cache: "default" });
    }
    return nativeFetch(input, init);
  };
})();
