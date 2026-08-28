"use strict";

(() => {
  const nativeFetch = window.fetch.bind(window);
  const wasmChunks = [
    "wasm-b2/part-00.b64",
    "wasm-b2/part-01.b64",
    "wasm-b2/part-02.b64",
    "wasm-b2/part-03.b64",
    "wasm-b2/part-04.b64",
    "wasm-b2/part-05.b64",
    "wasm-b2/p06-0.b64",
    "wasm-b2/p06-1.b64",
    "wasm-b2/p06-2.b64",
    "wasm-b2/p06-3.b64",
    "wasm-b2/part-07.b64",
    "wasm-b2/tail-00.b64",
    "wasm-b2/tail-01.b64",
    "wasm-b2/tail-02.b64",
    "wasm-b2/t-00.b64",
    "wasm-b2/t-01.b64",
    "wasm-b2/t-02.b64",
    "wasm-b2/t-03.b64",
    "wasm-b2/t-04.b64",
    "wasm-b2/t-05.b64",
    "wasm-b2/t-06.b64",
    "wasm-b2/t-07.b64"
  ];

  window.fetch = async (input, init) => {
    const url = typeof input === "string" ? input : (input?.url ?? "");
    if (!url.endsWith("sv_wasm_b2.wasm.gz.b64")) {
      return nativeFetch(input, init);
    }

    const pieces = [];
    for (const path of wasmChunks) {
      const response = await nativeFetch(path, { cache: "no-store" });
      if (!response.ok) {
        return new Response("", { status: response.status, statusText: response.statusText });
      }
      pieces.push((await response.text()).trim());
    }

    return new Response(pieces.join(""), {
      status: 200,
      headers: {
        "Content-Type": "text/plain;charset=utf-8",
        "Cache-Control": "no-store"
      }
    });
  };
})();
