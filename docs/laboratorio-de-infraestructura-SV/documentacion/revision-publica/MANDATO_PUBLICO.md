# Mandato público derivado · LP-DOC-0001/v1 + A01

**Vigencia:** aplicar [A01](ADENDA_A01.md) con prevalencia sobre las instrucciones de rama, integración y publicación del antecedente reproducido a continuación. Única rama remota de escritura: laboratorio-publico. No integrar en main ni en lab/lenguaje-sv-beta.

**Derivación declarada:** se omite exclusivamente la ruta local del equipo, sustituida por una indicación de custodia local. El original íntegro permanece localmente. SHA-256 del original: 44a18070c6660557892f86f3387e302294fce0a50ae34999a39befd493daf93f. Las referencias de v1 a otras ramas se conservan como antecedente superado por A01, no como autorización vigente.

---

# LP-DOC-0001 · Constitución del laboratorio público documental

**Revisión:** v1 · **Fecha:** 17/09/2026.  
**Unidad ejecutora:** U-DOC-PUBLICA.  
**Autoridad de dirección y autorización:** Dirección humana del proyecto.  
**Naturaleza:** encargo de arranque y mandato operativo documental. Su transmisión por la Dirección autoriza el arranque dentro de los límites siguientes. Las revisiones externas y los trabajos posteriores se activan únicamente por instrucción humana.

## 1. Finalidad y separación de responsabilidades

Constituir una sede pública de documentación y revisión adversarial aprovechando la presentación documental existente de SVcustos-dataset. Mantener fichas trazables, preparar encargos para dos revisores externos y custodiar sus respuestas por separado. Disponer de registro propio, instrucciones de continuidad y criterios de elevación de hallazgos.

La unidad asume el mantenimiento documental solicitado por la Dirección. No requiere que la unidad ejecutora del laboratorio privado ni la coordinación redacten sus fichas o preparen cada expediente. Las excepciones se elevan a la Dirección con una propuesta concreta. No se delega a esas unidades el trabajo rutinario.

Calidad, Sucesos y RETP del Lenguaje siguen siendo fuentes canónicas. El laboratorio público presenta su estado y evidencia; no asigna estados canónicos, acepta contratos, modifica la secuencia constitutiva ni incorpora cambios al núcleo. Una entrega productora, una revisión externa y una aceptación receptora son hechos distintos.

Los documentos serán técnicos, impersonales y redactados en español académico correcto. No incorporarán narración de conversaciones, sobrenombres personales, comentarios internos ni datos personales innecesarios. «Grok» y «Claude» identificarán exclusivamente la procedencia de las revisiones; los nombres ya presentes en URLs canónicas no se alterarán.

## 2. Proyecto local, repositorio y límites de escritura

La captura aportada sitúa el proyecto local en `[ruta local omitida; véase configuración local]`. Confirmar la ruta real desde el proyecto abierto; no corregir su nombre ni crear otra carpeta por suposición. Registrar la ruta efectiva sólo en la configuración local. No publicar rutas personales del equipo.

**Único repositorio remoto de escritura autorizado:** `juantoniolloretegea/SVcustos-dataset`.

**Rama de integración documental:** `lab/lenguaje-sv-beta`. Corte comprobado al preparar este encargo: `dd50c3e5c74e10526bb28087c8d5e2852d2efb68`. Obtener su punta vigente antes de comenzar y antes de publicar. La presencia de las fuentes en esta rama no acredita por sí sola la configuración actual de GitHub Pages.

| Objeto | Permiso |
|---|---|
| `docs/laboratorio-de-infraestructura-SV/documentacion/revision-publica/**` | Crear y mantener exclusivamente documentación, expedientes, registros y presentación de esta sede. |
| `docs/laboratorio-de-infraestructura-SV/documentacion/index.html` | Añadir un único acceso claramente rotulado a la nueva sede, conservando íntegros fichas, resultados, fechas y enlaces históricos. |
| Resto de SVcustos-dataset | Lectura. No modificar código, playground, binarios, historial Beta, estilos compartidos, licencias, workflows ni configuración de publicación. |
| SV-lenguaje-de-computacion y otros repositorios canónicos públicos | Lectura para documentar y contrastar fuentes. Sin escritura, PR, cambios de estado ni asignación de números canónicos. |
| SV-sala-de-maquinas y cualquier repositorio privado | Fuera del acceso necesario para este encargo. No escribir ni copiar contenido privado a la sede pública. |
| Directorios de otros proyectos locales | Sin escritura. No reutilizar ni alterar la copia de trabajo de otra unidad. |

Las rutas permitidas no habilitan enlaces simbólicos, uniones de directorio ni otros mecanismos que escriban fuera de ellas. Los metadatos Git de la copia propia, temporales y resultados de comprobación se mantienen dentro del proyecto local. No cambiar configuración Git global, credenciales, permisos del sistema ni valores predeterminados de herramientas.

Leer antes las instrucciones AGENTS aplicables. Si el directorio contiene trabajo previo, conservarlo y examinar su origen y estado; no borrar ni sobrescribir. Si no existe copia Git, preparar la copia del repositorio autorizado dentro del proyecto confirmado, sin ocupar otra carpeta existente. La separación de carpetas es una delimitación operativa; no afirmar aislamiento de permisos si no está efectivamente configurado.

Preparar los cambios en una rama local propia de documentación. La publicación queda autorizada únicamente para el conjunto de rutas anterior, conservando todos los avances remotos. Si se utiliza una rama remota temporal, emplear `doc/laboratorio-publico-0001`; si ya existe, comprobar su titularidad documental y contenido antes de reutilizarla. No publicar en `main`, forzar referencias, reescribir historia, eliminar ramas ajenas ni incorporar archivos de otros trabajos. Un conflicto sustantivo se conserva y se eleva a la Dirección.

## 3. Fuentes y estado de partida

Fuentes documentales existentes:

- [Documentación del laboratorio, corte inicial](https://github.com/juantoniolloretegea/SVcustos-dataset/tree/dd50c3e5c74e10526bb28087c8d5e2852d2efb68/docs/laboratorio-de-infraestructura-SV/documentacion).
- [Playground Beta conservado](https://github.com/juantoniolloretegea/SVcustos-dataset/blob/dd50c3e5c74e10526bb28087c8d5e2852d2efb68/docs/index.html).
- [Historial Beta conservado](https://github.com/juantoniolloretegea/SVcustos-dataset/blob/dd50c3e5c74e10526bb28087c8d5e2852d2efb68/docs/historial-beta.html).
- [Entrada canónica de continuidad](https://github.com/juantoniolloretegea/SV-lenguaje-de-computacion/blob/e1ab93d0de3df170a674f5a7887a3c20e96c6816/docs/calidad/tuberias-ia/continuacion-15-09-2026/inicio.md).
- [Estado canónico de Sucesos](https://github.com/juantoniolloretegea/SV-lenguaje-de-computacion/blob/e1ab93d0de3df170a674f5a7887a3c20e96c6816/docs/calidad/Inventario-sv/sucesos/SUCESOS_SV.md).
- [Recepción H2, S32 revisión 5 / RETP-2026-254](https://github.com/juantoniolloretegea/SV-lenguaje-de-computacion/blob/e1ab93d0de3df170a674f5a7887a3c20e96c6816/docs/calidad/tuberias-ia/continuacion-15-09-2026/PARTE_ALCANCE_PRIVACIDAD_SEGURIDAD_Y_OP_CYB_001_2026_09_15.md#s32-recepcion-h2-2026-09-17).

Estos commits son cortes de partida, no una afirmación de vigencia permanente. Al actualizar, identificar las puntas consultadas y su fecha; seguir AGENTS y las lecturas rectoras que prescriba. Consultar el acta de continuidad y su secuencia antes de proponer conclusiones sobre el rumbo.

En el corte inicial: H1-01/H1-02 están subsanados documentalmente; H2 sigue candidato; 18 comprobaciones instrumentales receptoras no equivalen a ensayos de privacidad; 19 casos previstos siguen sin ejecutar; durabilidad especificada, no implementada; S32/BIS-03 continúan abiertos. La nueva tarea S32-COBERTURA-01 está en ejecución según comunicación humana; hasta su recepción no se promoverá su resultado a estado canónico. Distinguir siempre lo comunicado de lo comprobado en una fuente.

No transformar R1/R2 de un reparo o revisión en fases nucleares del mismo nombre: identificar ámbito, documento, versión y apartado. Conservar igualmente la distinción entre E1–E4 del trayecto y E1–E16 de la leyenda.

## 4. Organización documental y registro propio

Crear dentro de la nueva sede las rutas siguientes. Todas las rutas de esta sección son relativas a `revision-publica/`.

| Ruta | Función |
|---|---|
| `LEAME_PRIMERO.md` | Mandato, permisos, fuentes, pasos de inicio y recuperación. |
| `index.html` y estilos propios locales si son necesarios | Acceso público legible a estado, fichas y expedientes; sin servicios externos ni funcionalidad ejecutora. |
| `REGISTRO_LP.csv` y `REGISTRO_LP.md` | Contabilidad propia de actividades y referencias, con presentaciones concordantes. |
| `HISTORIAL_LP.csv` | Transiciones y rectificaciones por adición, sin borrar asientos previos. |
| `ESTADO.md` | Último corte documental, pendientes, trabajo activo y siguiente actuación permitida. |
| `fichas/` | Síntesis derivadas, con evidencia, límites, fecha y revisiones conservadas. |
| `encargos/` | Instrucciones versionadas para revisiones externas. |
| `respuestas/` | Entregas externas originales, separadas por procedencia. |
| `dictamenes/` | Análisis documental de respuestas y recomendaciones a la Dirección. |

Usar identificadores propios `LP-DOC-0001`, `LP-AUD-0001`, `LP-HAL-0001`, sin consumir ni imitar S32 o RETP. Las referencias canónicas se citan, no se renumeran. Registrar identificador, objeto, versión, fuente y fecha de autorización cuando consten, fechas observadas de inicio/entrega, cortes de entrada, evidencia, resultado, límites y siguiente actuación. No inventar marcas temporales ausentes.

Separar autorización, ejecución, entrega y valoración. «No comprobable con el expediente» no significa «conforme» ni «defectuoso». Una rectificación conserva el asiento anterior, su motivo y el enlace a la revisión nueva.

El registro es documental, no una acreditación de seguridad. Una publicación no demuestra que su contenido sea verdadero ni que el estado descrito sea el vigente en otra fecha.

## 5. Encargos y respuestas de los dos revisores

Para cada revisión externa crear:

- `encargos/LP-AUD-NNNN/v1/ENCARGO_COMUN.md`.
- `encargos/LP-AUD-NNNN/v1/FUENTES.tsv` y `MANIFIESTO_SHA256.tsv`.
- `encargos/LP-AUD-NNNN/v1/grok/LEAME.md`.
- `encargos/LP-AUD-NNNN/v1/claude/LEAME.md`.
- `respuestas/LP-AUD-NNNN/grok/entrega-01/`.
- `respuestas/LP-AUD-NNNN/claude/entrega-01/`.
- `dictamenes/LP-AUD-NNNN/revision-01.md` cuando existan respuestas reales.

Los dos encargos remiten al mismo expediente congelado y a la misma rúbrica. Las instrucciones particulares sólo cambian la identificación y la ruta de respuesta. Fijar primero las fuentes y preguntas; publicar y verificar el expediente; después entregar a la Dirección los dos enlaces por commit. No enviar mensajes ni invocar APIs de los revisores. La Dirección decide cuándo transmitirlos y autorizar su ejecución.

Cada encargo especificará objeto, exclusiones, preguntas, fuentes, límites de acceso y formato de respuesta. Exigirá, por hallazgo: identificador, afirmación examinada, commit/ruta/apartado, evidencia, razonamiento o contraejemplo, impacto, alcance, grado de certeza y comprobación propuesta. Distinguir lectura, reproducción propia e inferencia. No afirmar ejecución de código si sólo se ha leído.

Los revisores tendrán lectura del expediente. Si disponen de escritura y la Dirección se la autoriza expresamente, sólo podrán añadir su propia entrega en la ruta asignada, sin modificar tareas, fuentes, registros, otras respuestas o dictámenes. Si carecen de escritura, la Dirección aportará el archivo o texto y U-DOC-PUBLICA lo depositará identificando su procedencia, sin simular acceso del revisor al repositorio.

Conservar la respuesta literal y separada del resumen. Correcciones en entrega-02, nunca sobre entrega-01. Una ausencia de respuesta se registra como tal. Ninguna respuesta puede fabricarse, atribuirse por inferencia ni presentarse como dictamen consensuado. No mostrar deliberadamente a un revisor la respuesta del otro antes de su primera entrega; si ya era pública o hubo exposición, declararlo. Una revisión pública no se calificará de ciega ni plenamente independiente sin justificarlo.

Antes de publicar una respuesta, verificar que es apta para difusión. Si contiene material privado o datos ajenos al alcance, conservar el original sólo localmente y solicitar a la Dirección una versión publicable. Nunca editarlo silenciosamente y llamarlo literal. Si se necesita una versión redactada, conservar su vínculo de procedencia y señalar las omisiones sin revelar el material protegido.

## 6. Publicación y límites de divulgación

Usar únicamente fuentes ya públicas verificadas y material nuevo generado dentro del alcance documental autorizado. Preservar atribuciones y licencias; no cambiarlas ni otorgar licencias nuevas a materiales de terceros.

No publicar contenido privado, credenciales, conversaciones, datos personales, respuestas reservadas ni soluciones de pruebas ciegas. Una referencia pública a un archivo privado no lo convierte en publicable. Solicitar autorización específica sólo si una divulgación adicional resulta necesaria; mientras tanto, marcar la limitación y continuar el resto del trabajo.

Un hash o un enlace privado permite identificar una referencia, pero no da acceso a su contenido. No pedir a los revisores que acrediten lo que no pueden examinar. Las limitaciones públicas no se confundirán con inexistencia de evidencia en la custodia canónica.

Conservar los originales locales y la correspondencia con los commits publicados. Evitar ZIP y descargas duplicadas. Verificar enlaces relativos, identidad de fuentes y acceso a los archivos públicos sin autenticación. Si Pages no refleja el commit o no puede verificarse, ofrecer los enlaces públicos inmutables de GitHub y declarar el despliegue pendiente; no alterar workflows, configuración de Pages ni DNS por este encargo.

## 7. Qué revisar prioritariamente y cuándo recomendar elevación

Revisar afirmaciones de aceptación o cierre sin evidencia; confusión entre prueba instrumental y propiedad material; cambio de significado por nomenclatura; referencias incorrectas; contradicción entre contratos, registros y fichas; pérdida de trazabilidad; permisos excedidos; exposición indebida; restauración o duplicación de efectos sin garantías; propuestas que alteren doctrina, núcleo, licencias o secuencia autorizada.

| Situación | Actuación de U-DOC-PUBLICA |
|---|---|
| Posible exposición sensible o fallo grave de autorización/custodia | Detener la publicación o actuación afectada; avisar a la Dirección sin divulgar detalles explotables. Recomendar revisión prioritaria. No intentar borrar historia o remediar otros sistemas. |
| Contradicción sustantiva, cierre posiblemente inválido, evidencia ausente o no reproducible que afecte una conclusión | Preparar expediente mínimo y recomendar traslado a la coordinación antes de aceptar o continuar el punto afectado. |
| Desacuerdo técnico entre revisores con impacto posible | Exponer ambas posiciones y la evidencia faltante; recomendar revisión cuando no pueda resolverse dentro de la competencia documental. No decidir por mayoría. |
| Errata o enlace defectuoso en una ficha propia, con fuente inequívoca | Corregir dentro de las rutas permitidas y registrar la rectificación. No generar trabajo canónico innecesario. |
| Mejora opcional sin defecto acreditado | Registrar como propuesta, sin convertirla en bloqueo ni encargo automático. |

La nota a la Dirección responderá: qué se ha encontrado, qué se ha comprobado, qué queda incierto, a qué decisión afecta, urgencia motivada y por qué conviene o no elevarlo. Ofrecer enlaces a evidencia y una pregunta concreta para recepción. La Dirección puede trasladar cualquier auditoría por iniciativa propia aunque la recomendación sea no elevar; también decide si autoriza actuaciones correctoras.

La unidad no contactará ni encargará trabajo automáticamente a la coordinación ni a la ejecución privada. En sus informes públicos, «recomendación de elevar» no significará «aceptado por la coordinación».

## 8. Primera actuación autorizada y punto de parada

1. Verificar proyecto, repositorio, rama, permisos efectivos e instrucciones aplicables; conservar el estado inicial y el mandato recibido.
2. Constituir la sede, sus registros propios, plantillas y procedimiento de relevo. Incorporar este encargo como documento de arranque conservado, sin publicar la ruta local del equipo: custodiar su original local y publicar un mandato público derivado que identifique expresamente esa omisión.
3. Preparar una primera ficha del estado canónico comprobado y una ficha del mapa de fuentes públicas. Distinguir documentación histórica del laboratorio, playground experimental y recepción vigente. No actualizar retrospectivamente los resultados históricos.
4. Preparar `LP-AUD-0001/v1`: revisión documental de coherencia entre el estado público vigente, la recepción S32 H2 y las afirmaciones de cobertura, ejecución y pendientes. Incluir la accesibilidad real de las evidencias como objeto de examen. No abarcar la auditoría completa del SV ni el trabajo privado aún en curso. Si existe nueva recepción canónica al iniciar, identificarla y delimitar el cambio de corte.
5. Publicar la sede y el expediente en las rutas autorizadas, comprobar los archivos remotos y añadir el acceso desde el índice documental existente. No invocar modelos externos. Mantener las dos respuestas como pendientes, sin archivos que simulen dictámenes recibidos.
6. Entregar a la Dirección: enlace de entrada; commit de publicación; cambios y rutas; registro; dos enlaces de encargo; comprobaciones realizadas; límites de acceso o despliegue; siguiente actuación propuesta. Distinguir publicación Git de visibilidad efectiva en Pages.

**Punto de parada:** «LP-DOC-0001 entregado; sede documental constituida y LP-AUD-0001 preparado para transmisión humana». La publicación inicial está comprendida en este arranque; las revisiones externas se iniciarán cuando la Dirección las transmita. No se exige otra intervención de la coordinación para el mantenimiento ordinario autorizado.

## 9. Operación posterior y recuperación

Cuando la Dirección solicite «revisar, actualizar y preparar una adversarial», ejecutar el ciclo dentro de este mandato: leer el estado canónico actual, fijar fuentes, actualizar fichas propias, conservar historial, preparar un expediente y dos encargos, verificar publicación y devolver enlaces. No ampliar capacidades, dominios ni permisos por esa fórmula. Una tarea fuera de alcance se presenta a la Dirección con justificación y resultado esperado, antes de ejecutarla.

Tras compactación, interrupción o relevo: leer LEAME_PRIMERO, mandato, registro, último estado y encargo activo; comprobar autorización y commit; identificar la última operación confirmada y comparar copia local/remota. No inferir finalización por la memoria de la conversación ni repetir operaciones con resultado incierto. La indisponibilidad de una fuente se documenta; no se rellena con recuerdos.

Trabajo documental sin campañas experimentales. No instalar herramientas innecesarias, contratar servicios, usar APIs de pago ni ejecutar código de terceros para esta revisión. Si un cotejo instrumental propio es necesario, respetar Rust 1.98.0 y fijar entradas, resultado esperado y alcance antes de ejecutarlo. La disponibilidad de herramientas no autoriza experimentos del SV ni cambios en otros proyectos.

## 10. Criterio de entrega del arranque

La entrega deberá acreditar: escritura limitada a las rutas autorizadas; ausencia de modificaciones en playground, núcleo, fuentes e históricos; registro propio recuperable; enlaces públicos fijados; dos encargos equivalentes con respuestas separadas; distinción entre evidencia accesible y reservada; conservación local; pendientes expresos; procedimiento de elevación bajo decisión humana. Si algún punto no puede acreditarse, declararlo con su alcance y siguiente acción, sin presentar conformidad total.

La autonomía documental se ejerce dentro de estos límites. La soberanía de decisión y de autorización permanece en la Dirección humana.
