# Respuesta a LP-AUD-0002/v1 · Claude · entrega-01

## Identidad y acceso

- **Procedencia del revisor:** Claude (Anthropic). Identificador de configuración de sesión: `claude-opus-5`. No es una versión de despliegue observada: el modelo que sirve un turno puede diferir de esa configuración y no es acreditable desde dentro de la sesión.
- **Fecha efectiva de inicio y entrega:** 17/09/2026, misma sesión.
- **Commit completo del encargo:** `116b69eae43037a9159a1d7c483b0e585c4445c2` (SVcustos-dataset, rama `laboratorio-publico`). Comprobado como ancestro de la punta remota observada `0f04ef0a679713bd008fad49a12cde17bb186f10`.
- **Autorización humana recibida y alcance:** transmisión del enlace exacto por Juan Antonio Lloret Egea, con autorización expresa de escritura limitada a la carpeta propia de este encargo: «Espero que puedas escribir en ese repo que es público, y para eso tienes tu carpeta. Sólo ahí y nada más que ahí». No se ha recibido ni se ha solicitado autorización de acceso privado.
- **Corte canónico auditado:** `bdb094958da5be00d2c24864c333deaa4f0636cc` (SV-lenguaje-de-computacion). Antecedente público: `206c405184a33884cdf0d19017b8f31d0f398340`.

### Fuentes realmente leídas

Íntegras, en el corte canónico: F01 AGENTS.md; F05 Acta 001 §§1–9; F06 inicio.md; F07 parte de privacidad §§1–16 (lectura íntegra, con atención a §§5, 7–10 y 12–16); F08 SUCESOS_SV.md (S29, S31, S32); F14 PROCEDIMIENTO_ENCARGOS. Del expediente, en `116b69ea`: ENCARGO_COMUN, PROTOCOLO_ENTREGA, PLANTILLA_RESPUESTA, FUENTES.tsv, MANIFIESTO_SHA256.tsv, ANALISIS_PREPARATORIO, MATRIZ_EVIDENCIA, REFERENCIAS_RESERVADAS, COMPROBACIONES_PREVIAS, BASE_PUBLICA_VERIFICADA.json, ambos LEAME, fichas estado-canónico v2 y v3.

Por apartados o consulta dirigida: F09/F10 (fila y revisiones de S32); F11 (RETP-2026-256 y 257); F12 (concordancia de filas). Fuera de las fuentes fijadas, en corte anterior `29fbcb02`: OP-CYB-001 §§6, 12.3–12.6 en `bbac1b44`, consultado en revisiones previas de S32 y reutilizado aquí sólo como antecedente.

**Lectura parcial declarada:** F02 Pilares, F03 Perfiles y F04 Transición **no** se han leído íntegros en esta revisión. Se ha comprobado su identidad (blob, SHA-256 y bytes) frente a lo declarado, y se han aplicado las guardas que `AGENTS.md` reproduce como índice de seguridad. `AGENTS.md` exige su lectura completa antes de intervenir en gramática, parser, IR, núcleo, pruebas o contratos; esta revisión es documental y no interviene en ninguno de esos objetos, pero la limitación se declara y ninguna conclusión de este informe se apoya en una cláusula no leída de esas tres piezas. F13 LEAME_PRIMERO no se ha leído. Se ha comprobado igualmente que `AGENTS.md` no cambió entre `29fbcb02` y `bdb09495`.

### Evidencias privadas no examinadas, motivo y efecto

R01, R02, R03 y R04 de REFERENCIAS_RESERVADAS: **no examinadas**. Motivo: el encargo no autoriza acceso privado y no se ha intentado el acceso. No se simula una solicitud rechazada. Efecto: toda conclusión material sobre código del cliente, frontera FFI, contrato CR08, banco, dependencias, registros originales y sesión productora queda **no comprobable** por mí; se atribuye exclusivamente a la recepción canónica.

### Otras fuentes públicas consultadas

Ninguna documentación externa adicional. Se intentó recuperar `https://static.rust-lang.org/dist/rust-1.98.0-x86_64-unknown-linux-gnu.tar.xz.sha256` en una revisión anterior de S32 y el sitio lo denegó a la herramienta de recuperación; no se reintentó aquí ni se rodeó por otra vía. Ese extremo no forma parte de este encargo.

### Exposición a otras revisiones

- **Exposición alta y declarada al expediente S32.** No soy revisor independiente. En sesión continuada con la Dirección emití (a) una auditoría del parte S32 en el corte `514dc0b7`, (b) una auditoría de la continuación §§7–10 en `29fbcb02`, y (c) una segunda revisión acotada sobre `29fbcb02` que propuso correcciones a §3, §8 y §10 y contrastó cinco objeciones de Watson, retirando tres conclusiones propias. Parte del contenido de §§11–14 del parte responde a esa línea de trabajo. Lo declaro para que la Dirección pondere o excluya mi resultado en cualquier contraste con revisores sin esa exposición.
- **No he buscado la respuesta de Grok.** Su carpeta `respuestas/LP-AUD-0002/grok/entrega-01/` apareció en el mismo listado del árbol con el que comprobé la existencia de mi propia carpeta de destino; observé que contiene únicamente `.gitkeep`, es decir, ninguna respuesta depositada. Lo declaro por transparencia: fue incidental, no una búsqueda deliberada.
- He leído ANALISIS_PREPARATORIO de U-DOC-PUBLICA. **No presento sus reservas como hallazgo propio**; cuando coincido lo digo, y señalo expresamente qué añado.

### Método

- **Lectura:** todas las fuentes anteriores.
- **Reproducción propia:** únicamente de **controles documentales y de identidad Git**, con `git`, `sha256sum`, `wc`, `diff` y utilidades de texto. Sin Python, conforme a la obligación S29; la lógica de comprobación aquí empleada es de identidad y recuento, no una realización SV.
- **No he ejecutado ni reproducido el código o el banco.** No he compilado el cliente ni el auxiliar, no he ejecutado CR08, NCBI, Q1/Q2, E1–E16, los 19 TLC ni campaña alguna. No he descargado ZIP, instalado herramientas ni usado credenciales, datos reales, Qwen o servicios externos.
- **Inferencia:** señalada como tal en cada caso.

### Reproducción documental efectivamente realizada

| Control propio | Resultado observado |
|---|---|
| Identidad de las 18 fuentes de FUENTES.tsv: blob Git, SHA-256 y bytes | **18/18 conformes, 0 discrepantes** |
| Identidad de las 14 entradas `expediente` de MANIFIESTO_SHA256.tsv | **14/14 conformes** |
| `dd50c3e…` → `206c405…` en laboratorio-publico | **2 commits**; diferencia neta **32 altas y 1 modificación**; las 32 altas bajo `revision-publica`; la única modificación es `documentacion/index.html`; **ninguna baja** |
| Ramas remotas de SVcustos-dataset | **8**, coincidente con lo declarado en §16.3 |
| Revisiones de S32 en HISTORIAL_SUCESOS_SV.csv | **9** (0–8), todas en estado `en ejecución`; MD y CSV concordantes; `fecha_actualizacion` `2026-09-17T14:18:22Z` coincide con la cabecera de §16 |
| Conservación de LP-AUD-0001 y fichas v1/v2 entre `206c405` y la punta `0f04ef0a` | **sin cambios** |
| Equivalencia de los dos LEAME particulares | Difieren **sólo** en nombre del revisor y ruta de carpeta propia |
| Ascendencia `29fbcb02` → `bdb09495` | Lineal, 10 commits; `AGENTS.md` sin cambios |
| `116b69ea` frente a la punta `0f04ef0a` | `116b69ea` es ancestro; la punta añade la entrega administrativa LP-DOC-0002 |

Estos controles acreditan **identidad documental y perímetro de cambios**. No acreditan seguridad material, corrección semántica ni ejecución alguna.

---

## Resultado por pregunta

### Q01 · Identidad y alcance — respaldo documental

Los dos auxiliares se distinguen con claridad. §15 trata **AUX-DUCKDB-NCBI-01/v1** (siete subcasos D01–D05 conformes, D06 recuperación y recompilación, 4192 entradas de manifiesto interior). §16 trata **AUX-CLIENTE-RUST-02/v1**, con encargo `19aae18f…`, entrega `8a38f1db…` y recepción `d2850083…`. Las 12 485 entradas y el banco 19/19 se atribuyen expresamente a la recepción privada y no a una comprobación propia del receptor canónico: «Se incorpora esta evidencia ya obtenida; la presente conciliación no repite ensayos».

Productores y receptores se separan: Windows conserva atribución productora, la recuperación es Linux. Plataformas y paquetes están identificados por commit. No he verificado ninguna de esas entradas: no dispongo de los paquetes.

**Objeción menor**, recogida como hallazgo CLA-0002-01: la coincidencia numérica entre «banco 19/19» y «19 TLC» se separa por prosa y no por identificador.

### Q02 · Frontera Rust/FFI — no comprobable con el expediente, con propuesta

El corpus **no** infiere aislamiento a partir de guardas Rust; lo niega expresamente y en dos sedes: §16.2, «La biblioteca nativa DuckDB continúa enlazada mediante FFI dentro del proceso del cliente. Las guardas Rust no constituyen aislamiento del motor», y §15, «La FFI del auxiliar comparte proceso con la biblioteca nativa y exige límites materiales adicionales». La ficha v3 lo repite con el mismo límite. En este extremo la presentación es correcta y conservadora.

Para **juzgar** carga nativa, ABI, memoria, fallos del proceso y capacidades efectivas haría falta, como mínimo:

1. Fuente de la frontera: firmas `extern "C"`, bloques `unsafe` y su justificación; si el cliente declara o no `forbid(unsafe_code)` en los módulos que no cruzan la frontera.
2. Identidad exacta de la biblioteca: versión de DuckDB, procedencia del artefacto, opciones de compilación, y si el enlace es estático o dinámico; sin esa fuente **no certifico código ni versión de DuckDB**, y el corpus público tampoco los fija.
3. Modelo de fallos a través de la frontera: qué ocurre con un `panic` que se propague por un `extern "C"` (comportamiento indefinido salvo `catch_unwind` o `extern "C-unwind"`), y qué ocurre con un `abort` del motor, que termina el proceso del cliente entero.
4. Contrato de propiedad de memoria de los búferes que cruzan: quién asigna, quién libera, con qué asignador, y qué pasa en el camino de error.
5. Capacidades efectivas del proceso: descriptores, red, sistema de ficheros y variables del entorno que el motor hereda por compartir proceso.
6. Pruebas negativas discriminantes que **induzcan** fallo del motor y observen el efecto sobre el cliente, no pruebas del camino feliz.

Nada de esto figura en el corpus público. No lo presento como carencia del proyecto: lo presento como la evidencia mínima exigible antes de cualquier afirmación sobre la frontera.

**Inferencia que no hago:** ACTA_002 §158 acredita `forbid(unsafe_code)` en cinco fuentes históricas de cotejo. Son otro artefacto. No traslado esa propiedad al cliente.

### Q03 · CR08 / RCR-01 — objeción sustentada sobre la especificación del complemento

La respuesta a la pregunta literal es **no**: eliminar todos los eventos no sustenta la propiedad de detectar la omisión de cada evento requerido. Un rechazo de la lista vacía es compatible con una implementación que sólo compruebe «la lista no está vacía», que es estrictamente más débil. El corpus lo reconoce y mantiene RCR-01 **abierta**; en ese extremo la presentación es correcta y **no** cierra RCR-01 por el 19/19 ni por proponer un ensayo.

Lo que añado —y es mi aportación propia sobre la reserva ya documentada— es que **el complemento, tal como está especificado, cerraría una propiedad más estrecha que la enunciada**. §16.2 exige «estímulo, causa de rechazo y oráculo precomprometidos». Un complemento que retire exactamente un evento manteniendo lo demás discrimina **presencia**. No discrimina:

- **multiplicidad**: un evento requerido *n* veces del que se retira una ocurrencia;
- **orden**: los mismos eventos en secuencia no admisible;
- **cobertura por evento**: que la sensibilidad valga para *cada* evento requerido y no sólo para el elegido, lo que exige una familia de mutaciones, no una.

El propio ANALISIS_PREPARATORIO señala que «la semántica del contrato original sigue siendo necesaria para determinar eventos requeridos, multiplicidad u orden», pero la formulación del complemento en §16.2 no arrastra esa distinción. Corrección mínima: que el complemento declare **cuál de las tres propiedades cierra** y que la conclusión posterior se enuncie con ese alcance, no con el general.

Delimitación pedida, sin inventar requisitos del contrato, que no he leído:

- **Control válido:** entrada positiva aceptada, con retorno y diagnóstico registrados antes de mutar.
- **Única diferencia:** retirada de una sola ocurrencia de un solo evento requerido, conservando bytes, orden y resto de campos; identidad de la entrada fijada antes y después.
- **Causa de rechazo:** el diagnóstico debe nombrar la obligación incumplida, distinguible de error de esquema, de límite y de fallo técnico. Un retorno distinto de cero no basta.
- **Oráculo previo:** resultado esperado fijado y con huella **antes** de ejecutar, en sede separada del ejecutor.
- **Cobertura:** repetición sobre cada evento requerido del contrato.

No he ejecutado este complemento ni lo propongo como autorizado.

### Q04 · RCR-02 / RCR-03 — respaldo documental

La diferencia se preserva correctamente. §16.2: «RCR-03 queda resuelta por recuperación propia. […] RCR-02 conserva las limitaciones históricas de captura inicial e independencia; una reproducción posterior no completa retroactivamente los registros ausentes. El intento fallido CR06, su reparación y los resultados anteriores se conservan». La ficha v3 y RETP-257 repiten la distinción sin ampliarla. No observo reconstrucción retrospectiva ni absorción de RCR-02 por RCR-03.

No puedo comprobar materialmente ni la recuperación ni el fallo CR06: ambos residen en R02.

### Q05 · Contención — objeción sustentada

La respuesta es inequívoca en el corpus: **AUX-C01–C04 son condiciones pendientes, no resultados acreditados**. §16.2: «AUX-C01–C04 siguen pendientes». §15: «Esta aceptación no habilita el auxiliar en producción ni acredita confinamiento completo». Y el punto que el encargo subraya está expresamente recogido: «El modo offline de Cargo no acredita aislamiento de red impuesto por el sistema operativo». Correcto: `--offline` es una política del gestor de paquetes dentro del proceso, no una restricción impuesta por el sistema; no dice nada del egreso del binario resultante.

**Objeción:** los cuatro códigos no tienen una vinculación estable con su texto. §16.2 los enumera como «supervisor y permisos, carga y frontera nativa, transporte NCBI, recursos y salida». §15, al describir los mismos cuatro pendientes de contención, escribe «La supervisión, la identidad de dependencias, la política del transporte externo y los límites de recursos/salida». **El segundo elemento difiere**: «identidad de dependencias» frente a «carga y frontera nativa». Además §15 no liga su lista a los códigos AUX-C01–C04. Detalle en CLA-0002-02.

No puedo comprobar si existe supervisor, límite de recursos o control de egreso: exigiría los contratos y realizaciones, que son privados.

### Q06 · Autorización por dominio — respaldo documental, con carencia señalada

El requisito está bien planteado y en los términos que pide la pregunta. §15: «Se requiere un contrato común y perfiles de autorización por dominio, finalidad y expediente». §16.4: «Inmunología y ciberseguridad conservan perfiles propios sobre el contrato común: no se comunican datos sensibles de auditoría, secretos ni datos de terceros por disponer de un conector». Y se bloquea explícitamente la inferencia por agregación: «Las consultas y los agregados también pueden revelar información: su comunicación requiere evaluación específica».

El corpus **no inventa un contrato ya constituido**: §15 cierra con «Esta precisión es requisito pendiente, no control ya implementado ni auditoría integral del currículo». Esa frase es la que impide leer el apartado como una capacidad existente, y está puesta.

**Carencia que señalo, sin decidir nada:** lo que falta para pasar de requisito a contrato es lo que el propio corpus ya nombra en otra sede —operación, recurso, finalidad, destinatario, delegación, vigencia y revocación— más el **observador** que registre la denegación. Un permiso sin denegación observable no es comprobable. No propongo ni concedo permisos: la política corresponde a la autoridad competente.

### Q07 · Prueba y conclusión — respaldo documental

Los cuatro conjuntos se separan correctamente y en más de una sede:

| Conjunto | Dónde se separa | Estado declarado |
|---|---|---|
| Siete subcasos del auxiliar (D01–D05, más D06) | §15 | Conformes, en recepción |
| 19/19 del banco del cliente | §16.2, RETP-257 | Concordante, según recepción privada |
| 18 controles instrumentales de H2 | §13.3: «Las 18 combinaciones de seis variantes por tres modos» | Reproducidos por el receptor |
| 19 TLC | §13.4 y §16.2 | **Ninguno ejecutado** |

La frase decisiva está escrita: «Los 19 resultados del banco no son los 19 TLC, que permanecen sin ejecutar». También se separan compilación, recuperación, identidad, causalidad, contención y aceptación: §13.2 dice «La secuencia temporal no acredita por sí sola causalidad»; §16.2 separa recuperación de aceptación íntegra; §15 separa aceptación de confinamiento.

**Precisión adicional:** §12.3 advierte que dos salidas históricas copiadas en la propuesta pertenecen al paquete original ejecutado en Linux y «no constituyen salidas del auxiliar H1 en Windows». Es una separación fina y correcta, y conviene que sobreviva a futuros resúmenes: es justo el tipo de matiz que se pierde al condensar.

### Q08 · Trazabilidad y escritura — respaldo documental parcial y mejora propuesta

**Lo que he reexaminado yo:** los dos commits públicos. Confirmo `dd50c3e…` → `206c405…`: dos commits, 32 altas todas bajo `revision-publica`, una modificación —`documentacion/index.html`— y ninguna baja. Coincide con §16.3 y con COMPROBACIONES_PREVIAS.

**Lo que no he reexaminado:** los siete commits privados. Su resultado —cinco productores y dos receptores, 1 196 archivos añadidos, cuatro modificados, ninguno eliminado— se atribuye **exclusivamente a la recepción**. No lo confirmo ni lo pongo en duda.

**Qué acreditan los nueve y qué queda fuera.** Acreditan que, en los árboles examinados, no aparecen rutas ajenas al perímetro autorizado. El propio §16.3 delimita lo que queda fuera: «esta afirmación no se extiende a todas las operaciones locales, referencias borradas, acciones no capturadas ni a la identidad material del operador». La delimitación es correcta y suficiente.

**Mejora propuesta, CLA-0002-03:** la frase resumen «No se ha identificado escritura fuera de alcance en estos nueve commits» cubre con un solo enunciado dos regímenes probatorios distintos —dos verificables públicamente, siete atribuidos a recepción—. La tabla de §16.3 sí los separa por fila; el resumen no.

**Verificación de mi rama y ruta de destino:** comprobadas. La carpeta `respuestas/LP-AUD-0002/claude/entrega-01/` existe y contiene únicamente `.gitkeep`; no hay respuesta previa que conservar. La punta remota de `laboratorio-publico` observada es `0f04ef0a679713bd008fad49a12cde17bb186f10`, posterior al corte del encargo. Los impedimentos de escritura se declaran en el último apartado.

### Q09 · Ramas y protecciones — no comprobable por mí; presentación conforme

La presentación limita correctamente el dato. §16.3: «El indicador `protected` de las ramas enumeradas es falso, incluidas las principales. Es un dato del endpoint consultado, no una inspección administrativa exhaustiva de reglas, permisos o protecciones efectivas». BASE_PUBLICA_VERIFICADA.json lleva su propio campo `limite` con la misma reserva. No se convierte inventario, edad ni muestreo en prueba de permisos o abandono, y se dice expresamente: «No se han creado, eliminado, fusionado, reescrito ni limpiado ramas durante esta revisión». **No recomiendo borrado automático alguno ni cambio de reglas.**

**No he podido reproducir la observación `protected=false`.** La API de GitHub devuelve 403 a esta sesión, con o sin credencial. Sí he podido enumerar las ramas por protocolo Git: **ocho**, coincidente con lo declarado. El valor de `protected` queda, para mí, **no comprobable**.

Observación menor: PROTOCOLO_ENTREGA prohíbe escribir en `doc/laboratorio-publico-0001`, rama que no aparece en la enumeración remota actual. Prohibir una rama inexistente es inocuo; lo anoto sólo para que no se lea como evidencia de su existencia.

### Q10 · Continuidad y aceptación — respaldo documental

Todos los extremos se mantienen abiertos y así constan, de forma concordante, en el parte §16.4, en Sucesos S32 y en RETP-2026-257: S32 `en ejecución` y BIS-03 abierto; H2 candidato; durabilidad especificada pero **no implementada ni acreditada**; aceptación íntegra y selección del cliente pendientes; producción, datos reales y **entrenamiento federado** excluidos; sin Qwen, navegador, WASI, API de producción ni consulta federada habilitadas. He comprobado la concordancia entre las tres sedes y las nueve revisiones del historial.

**Qué evidencia falta antes de una decisión posterior**, en el orden en que condiciona:

1. Complemento RCR-01 con la delimitación de Q03, incluida la declaración de qué propiedad cierra.
2. Fuente de la frontera FFI y los seis extremos de Q02, para poder juzgar la frontera nativa.
3. Contratos y realizaciones de AUX-C01–C04, con pruebas positivas y negativas y un observador material.
4. Política de autorización por dominio decidida por autoridad competente, con denegación observable.
5. Ejecución de los 19 TLC, que hoy son cero.
6. Realización durable e independiente, hoy sólo especificada.

---

## Hallazgos

Identificadores locales del revisor. **No asigno números S32, RETP ni LP-HAL canónicos.**

### CLA-0002-01 · Colisión numérica entre el banco del cliente y los TLC

- **Afirmación examinada:** «Los 19 resultados del banco no son los 19 TLC, que permanecen sin ejecutar».
- **Sede:** SV-lenguaje-de-computacion `bdb094958da5be00d2c24864c333deaa4f0636cc`, `docs/calidad/tuberias-ia/continuacion-15-09-2026/PARTE_ALCANCE_PRIVACIDAD_SEGURIDAD_Y_OP_CYB_001_2026_09_15.md`, §16.2 (S32 revisión 8). Concordante en ficha v3.
- **Evidencia:** la separación existe y es correcta, pero descansa en una frase de prosa. Ambos conjuntos se citan como «19» en RETP-2026-257 y en Sucesos S32.
- **Razonamiento:** Acta 001 §9 fija que en nuevas actas y relevos se escriba **ámbito, tipo de objeto y código**, y que «un código aislado no determina ni el objeto ni su estado de ejecución». Un cardinal compartido por dos conjuntos de estatuto opuesto —uno concordante, otro sin ejecutar— es el mismo riesgo que esa regla legisla. Al condensar, «19/19» y «19 TLC» convergen.
- **Impacto:** bajo hoy, acumulativo. Un tercero que lea sólo el RETP puede leer los TLC como ejecutados.
- **Alcance:** documental. No afecta a ningún resultado.
- **Certeza:** alta sobre el hecho; media sobre la probabilidad de confusión futura.
- **Comprobación propuesta:** en la primera mención de cada sede, «banco publicado de AUX-CLIENTE-RUST-02 (19 comprobaciones, concordantes)» y «casos TLC-S32-02 (19, no ejecutados)». No renombrar códigos históricos, conforme a Acta 001 §9.
- **Naturaleza:** hallazgo nuevo. Mejora opcional.

### CLA-0002-02 · AUX-C01–C04 sin vinculación estable código–texto

- **Afirmación examinada:** «AUX-C01–C04 siguen pendientes: supervisor y permisos, carga y frontera nativa, transporte NCBI, recursos y salida» (§16.2).
- **Sede:** mismo archivo y corte; §15 y §16.2.
- **Evidencia y contraejemplo:** §15 describe los cuatro pendientes de contención como «La supervisión, la identidad de dependencias, la política del transporte externo y los límites de recursos/salida». El segundo elemento no coincide con el de §16.2, y §15 no liga su lista a los códigos.
- **Razonamiento:** «identidad de dependencias» y «carga y frontera nativa» son obligaciones distintas —procedencia y versión de lo enlazado frente a mecánica y límites del enlace—. Si AUX-C01–C04 son las puertas que condicionan la integración, su contenido no puede variar entre dos apartados consecutivos del mismo documento. Una de las dos lecturas quedará sin contrato al redactar las realizaciones.
- **Impacto:** medio. Afecta a qué se exige antes de integrar.
- **Alcance:** documental, con consecuencia sobre el alcance de los contratos futuros.
- **Certeza:** alta sobre la discrepancia textual; **no comprobable** cuál de las dos lecturas refleja los cuatro pendientes locales registrados en la recepción privada, que no he leído.
- **Comprobación propuesta:** fijar en una sola sede la tabla AUX-C01…AUX-C04 con su texto canónico, y que las demás menciones citen el código. Si la recepción privada contiene ambos elementos, podrían ser cinco y no cuatro; eso lo determina quien tenga acceso a R01.
- **Naturaleza:** hallazgo nuevo. Objeción sustentada.

### CLA-0002-03 · Un resumen cubre dos regímenes probatorios

- **Afirmación examinada:** «No se ha identificado escritura fuera de alcance en estos nueve commits» (§16.3).
- **Sede:** mismo archivo y corte, §16.3.
- **Evidencia:** dos de los nueve son públicos y reexaminables —los he reexaminado y conforman—; siete son privados y sólo constan por recepción. La tabla de §16.3 los separa por fila; la frase posterior los reúne.
- **Razonamiento:** la frase es verdadera bajo ambos regímenes, pero al citarse suelta transfiere al conjunto la fuerza probatoria del subconjunto verificable.
- **Impacto:** bajo.
- **Alcance:** documental.
- **Certeza:** alta.
- **Comprobación propuesta:** «Dos commits públicos se han cotejado directamente; para los siete privados el resultado se atribuye a la recepción, sin reexamen independiente».
- **Naturaleza:** hallazgo nuevo. Mejora opcional.

### CLA-0002-04 · Complemento de RCR-01 más estrecho que la propiedad enunciada

Desarrollado en Q03. **Objeción sustentada.** Sede: §16.2, corte `bdb09495`. Impacto: alto sobre la conclusión que podría extraerse del complemento; nulo sobre el estado actual, que ya es «pendiente». Certeza: alta sobre la insuficiencia lógica de una mutación única frente a multiplicidad, orden y cobertura; **no comprobable** qué exige realmente el contrato CR08, que no he leído. Comprobación propuesta: la delimitación de cinco puntos de Q03, y que el complemento declare la propiedad que cierra. **Reserva previa** en cuanto a la insuficiencia del `missing` —ya documentada en §16.2 y en ANALISIS_PREPARATORIO—; **hallazgo nuevo** en cuanto a la estrechez del complemento propuesto.

### CLA-0002-05 · Impedimento de escritura

Ver «Escritura y entrega». **No comprobable por el expediente**: la causa es de la infraestructura de mi sesión, no del repositorio ni de sus permisos.

---

## Matriz de evidencia inaccesible

| Objeto / referencia | Causa observada o desconocida | Efecto sobre cada conclusión | Evidencia mínima requerida | Acceso |
|---|---|---|---|---|
| R01 · RECEPCION_LINUX.md del auxiliar, `bab99267…` | No autorizado por este encargo; **no se intentó el acceso** | Q01 y Q05 quedan limitados a concordancia documental; los cuatro pendientes locales de contención no se leen | Registro original con comandos, salidas literales y manifiestos | No suministrado / no autorizado |
| R02 · RECUPERACION_LINUX.md del cliente, `d2850083…` | Igual | Q01, Q04 y Q07 no comprobables materialmente: 12 485 entradas, banco 19/19, retornos y marcas horarias | Registro original, entradas, comandos, salida literal | No suministrado / no autorizado |
| R03 · AUX-CLIENTE-RUST-02/v1: encargo `19aae18f…`, entrega `8a38f1db…` | Igual | Q02 y Q03 no comprobables: no hay fuente de la frontera FFI ni contrato CR08 | Fuente del cliente y de la frontera, `Cargo.toml`/`Cargo.lock`, contrato CR08, banco y oráculo | No suministrado / no autorizado |
| R04 · Acta preventiva, `d9db35e3…` | Aportada por la Dirección; contenido no consultado | Q08 y Q09 se apoyan sólo en §16.3 para los siete commits privados | Acta y sus anexos | No suministrado / no autorizado |
| Biblioteca DuckDB enlazada: versión, procedencia, opciones de compilación | No figura en el corpus público | Q02: no certifico código ni versión de DuckDB | Identidad del artefacto y de su construcción | No examinado |
| `protected` de las ramas y reglas administrativas | **API de GitHub devuelve 403 a esta sesión**, con y sin credencial | Q09: no reproduzco la observación; sí enumero 8 ramas por Git | Consulta autorizada de reglas y rulesets, y de permisos efectivos del actor | Inaccesible por restricción de mi sesión |
| F02 Pilares, F03 Perfiles, F04 Transición, F13 LEAME_PRIMERO | Decisión propia de alcance: identidad comprobada, lectura íntegra no realizada | Ninguna conclusión se apoya en cláusulas no leídas; queda declarado | Lectura completa | No examinado (accesible) |
| Huella oficial del paquete Rust 1.98.0 | El sitio de origen deniega la recuperación a mi herramienta | Ninguno en este encargo | Descarga autorizada del fichero `.sha256` | Inaccesible (fuera de alcance) |

Ninguna casilla vacía o inaccesible se interpreta como conformidad.

---

## Conclusiones separadas

### Presentación documental

**Conforme en el alcance examinado.** Las 18 fuentes fijadas conservan su identidad (18/18) y las 14 piezas del expediente también (14/14). El parte, Sucesos S32, RETP-2026-256/257 y las fichas v2/v3 son concordantes entre sí en estado, fechas, unidad y reservas. Los dos commits públicos que he reexaminado se comportan exactamente como se declara. No he hallado ninguna afirmación pública de aceptación íntegra, de cierre de contención o de ejecución de los TLC; al contrario, cada apartado que podría leerse así lleva su límite adjunto.

Las tres objeciones documentales —CLA-0002-01, 02 y 03— no alteran esta conformidad; 02 es la única con consecuencia práctica.

### Diseño examinable

**Suficiente para sostener las reservas declaradas; insuficiente para sostener ninguna propiedad material.** El diseño acierta en lo difícil: separa disponibilidad de habilitación, recuperación de aceptación, identidad de significado, secuencia temporal de causalidad, y niega expresamente las cuatro inferencias más tentadoras —guardas Rust como aislamiento, `--offline` como bloqueo de red, agregación como permiso de egreso, y 19/19 como cobertura de los TLC—. Esa disciplina es lo que hace auditable el expediente.

Donde el diseño todavía no llega es en la especificación del complemento RCR-01 (CLA-0002-04) y en la fijación de AUX-C01–C04 (CLA-0002-02).

### Verificación material y evidencia faltante

**No comprobable.** No he ejecutado ni reproducido el código o el banco. No he leído el cliente, la frontera FFI, el contrato CR08, las entradas, los oráculos, las dependencias ni ninguno de los cuatro objetos reservados. Por tanto **no emito** ninguna conclusión sobre: corrección del cliente, seguridad de memoria a través de la FFI, aislamiento del motor, existencia o eficacia de contención, veracidad de las 12 485 entradas, del banco 19/19 o de los siete subcasos, identidad del operador, legitimidad de las 119 ramas canónicas, ni protección efectiva de ninguna rama.

La falta de esa evidencia **no es conformidad, ni aceptación, ni ausencia de defectos**. Es ausencia de examen.

### Recomendación a la Dirección, urgencia y pregunta concreta

**Recomiendo no seleccionar ni integrar el cliente con la evidencia actual**, y mantener S32 y BIS-03 abiertos. No es una objeción al trabajo recibido: es la consecuencia de que la evidencia decisiva sea privada y no se haya examinado.

Orden propuesto, por dependencia y no por esfuerzo:

1. **Fijar AUX-C01–C04** en una sola sede con su texto canónico, y determinar si son cuatro o cinco obligaciones. Es documental, cuesta poco y desbloquea la redacción de los contratos.
2. **Especificar el complemento RCR-01** con la delimitación de Q03, declarando qué propiedad cierra. Sin eso, un complemento favorable se leerá como más de lo que prueba.
3. **Decidir si se abre un corpus exacto publicable** para la frontera FFI (los seis extremos de Q02) o si se autoriza un canal privado separado. Sin una de las dos cosas, Q02 permanecerá no comprobable indefinidamente y ninguna auditoría externa podrá cambiarlo.

**Urgencia:** previa a selección o integración. **No afirmo un incidente activo**, y nada de lo leído sugiere uno.

**Pregunta concreta a la Dirección:** ¿autoriza delimitar un corpus exacto y publicable de la frontera Rust/FFI y del contrato CR08 —sólo firmas, contrato y oráculo, sin registros privados ni datos— de modo que Q02 y Q03 puedan pasar de «no comprobable» a examinable sin abrir el repositorio privado?

### Qué no se concluye ni se habilita

No se concluye aceptación íntegra, selección del cliente, sustitución del auxiliar anterior, cierre de RCR-01, implantación de contención, suficiencia de permisos, legitimidad de ramas ni ausencia de defectos. No se habilita producción, NCBI, credenciales, datos reales, Qwen, navegador, WASI, API de producción, consulta federada ni entrenamiento federado, que permanece excluido. No se autoriza ejecutar las hipótesis de ataque, los casos propuestos ni campaña alguna. No se corrige el cliente, CR08, registros canónicos, otros encargos, ramas ni protecciones. No se asignan números canónicos. No se declara cerrado S32, BIS-03, S22, S26 ni (p1+p3)-Bis.

---

## Escritura y entrega

- **Autorización expresa de escritura:** **recibida**, de Juan Antonio Lloret Egea, limitada a la carpeta propia de este encargo en `laboratorio-publico`.
- **Capacidad de escribir en laboratorio-publico:** **no disponible**. Comprobada, no supuesta.
- **Publicación realizada:** **no**.
- **Repositorio/rama/ruta efectivos:** ninguno; no se escribió.
- **Motivo de impedimento observado:** restricción de la infraestructura de mi sesión, no de GitHub ni del repositorio. Diagnóstico, sin secretos: la lectura anónima por protocolo Git funciona —he clonado y cotejado ambos repositorios—, pero la API de GitHub devuelve `403` con el mensaje de que el acceso al repositorio «no está habilitado para esta sesión», y una prueba de autenticación de escritura sin efecto (`git push --dry-run`, que no modifica nada) devuelve: *«access denied by the git proxy: juantoniolloretegea/SVcustos-dataset is not in this session's authorized repository set, so the proxy will not inject a credential for it»*. El mecanismo indicado para habilitarlo (`add_repo`) **no está disponible** en esta sesión. No es un problema de permisos de la cuenta, ni de protección de rama, ni de selección de rama: el proxy no inyecta credencial alguna para este repositorio.
- **Escrituras fuera del destino permitido:** **ninguna**. No he creado ramas, ni forks, ni PR, ni he escrito en `main`, en la rama beta ni en ningún otro repositorio. La única operación remota de escritura intentada fue el `--dry-run` anterior, sobre el destino autorizado y sin efecto, con la finalidad exclusiva de diagnosticar la capacidad.
- **Commit y archivos publicados:** ninguno.
- **Modo de devolución a la Dirección:** este documento íntegro, entregado como archivo `RESPUESTA.md` para su depósito por U-DOC-PUBLICA en `docs/laboratorio-de-infraestructura-SV/documentacion/revision-publica/respuestas/LP-AUD-0002/claude/entrega-01/RESPUESTA.md`. La carpeta existe y contiene sólo `.gitkeep`; no hay respuesta previa que conservar. La punta remota observada de `laboratorio-publico` es `0f04ef0a679713bd008fad49a12cde17bb186f10`.

> No he podido publicar mi respuesta en juantoniolloretegea/SVcustos-dataset, rama laboratorio-publico, ruta asignada. Motivo observado: la infraestructura de mi sesión no tiene este repositorio en su conjunto autorizado y su proxy Git no inyecta credencial para él, por lo que toda escritura devuelve 403; el mecanismo de habilitación indicado por el propio error no está disponible en esta sesión. No he utilizado otra rama ni otro repositorio como alternativa. Entrego el archivo `RESPUESTA.md` a la Dirección para su depósito por U-DOC-PUBLICA.

Entrego y me detengo. No inicio reparaciones ni contacto otras unidades.
