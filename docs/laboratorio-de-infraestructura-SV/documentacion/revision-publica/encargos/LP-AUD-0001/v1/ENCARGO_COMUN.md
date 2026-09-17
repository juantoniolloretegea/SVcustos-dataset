# LP-AUD-0001/v1 · Revisión documental de coherencia y cobertura

**Preparación autorizada:** LP-DOC-0001/v1 + A01, 17/09/2026.  
**Ejecución:** únicamente tras transmisión humana de esta revisión por commit. Publicar el encargo no inicia la revisión.

## Objeto y corte

Examinar la coherencia entre el estado público vigente en `7d6ecaa7d56b786ff194379417fef2ec2f0952df`, la recepción S32 H2 revisión 5 / RETP-2026-254, su continuación de cobertura revisión 6 / RETP-2026-255 y las afirmaciones de cobertura, ejecución y pendientes. Examinar también la accesibilidad real de las evidencias.

Corpus común: [FUENTES.tsv](FUENTES.tsv), [MANIFIESTO_SHA256.tsv](MANIFIESTO_SHA256.tsv), [ficha de estado](../../../fichas/estado-canonico/v2.md) y [mapa](../../../fichas/mapa-fuentes/v2.md). Las fuentes canónicas rigen sobre las síntesis. El expediente queda fijado por el commit completo transmitido. No sustituir fuentes por main ni por otra punta móvil. Una recepción posterior pertenece a una nueva revisión.

El corte inicial e1ab93d0de3df170a674f5a7887a3c20e96c6816 se conserva como antecedente. Antes de la primera publicación se incorporó la recepción de cobertura detectada al comprobar la punta canónica. No hubo transmisión ni ejecución de una versión anterior de este encargo. Las fichas v1/v2 documentan el cambio. El manifiesto incluye fuentes y documentos fijos; excluye su propio archivo y los registros administrativos susceptibles de nuevas entradas.

## Exclusiones y acceso

Revisión documental finita. No auditoría completa del SV, implementación, campaña experimental, evaluación jurídica integral ni trabajo privado en curso. No usar credenciales para entrar en repositorios privados, invocar servicios de pago ni modificar fuentes o estados canónicos. Se permite leer las fuentes públicas fijadas y declarar límites de acceso; si otra referencia pública resulta indispensable, identificarla expresamente y justificar su uso sin ampliar el objeto.

El paquete H2 y registros materiales están en custodia privada según la recepción pública. El expediente no permite inspeccionarlos directamente. No exigir su comprobación ni inferir que no existen. No atribuir ejecución de código a una lectura.

## Preguntas comunes

1. ¿Queda delimitada la aceptación documental de H1-01/H1-02 frente al estatuto candidato de H2 y la ausencia de aceptación integral?
2. ¿Se distinguen las 18 comprobaciones instrumentales de los 19 casos previstos no ejecutados y de los ensayos de privacidad?
3. ¿Las afirmaciones sobre durabilidad, confirmación previa al efecto, resultado indeterminado, restauración y reintentos respetan los límites expresos de la recepción?
4. ¿La cobertura residual de los diez flujos conserva las obligaciones de componente, contrato, prueba, situación, seguimiento y condición de habilitación? ¿Se distingue opción condicionada de obligación pendiente y se evita presentar las brechas A–I como seguimientos asignados?
5. ¿Se preservan S32/BIS-03 abiertos, las competencias de Calidad y la secuencia rectora sin transformar reparos R1/R2 o etapas E1–E4 en fases o casos de otro ámbito?
6. ¿Las remisiones identifican commit, ruta, revisión y apartado correctos? ¿Los antecedentes del índice quedan inequívocamente subordinados a la recepción vigente?
7. ¿Qué conclusiones son comprobables con fuentes públicas y cuáles dependen de originales reservados? ¿Se evita equiparar inaccesibilidad con ausencia, conformidad o defecto?
8. ¿La sede derivada conserva trazabilidad, permisos, estados históricos y la separación entre comunicación humana, entrega, revisión y aceptación?

Priorizar cierre sin evidencia, prueba instrumental convertida en propiedad material, contradicción contractual/registral, ambigüedad de nomenclatura, duplicación de efectos sin garantías y divulgación o permisos excedidos.

## Rúbrica y formato obligatorio

Comenzar con commit del encargo, revisor de procedencia, fecha efectiva, fuentes realmente leídas, enlaces inaccesibles, método y cualquier exposición a la otra respuesta. Separar **lectura documental**, **reproducción propia** (si no existe, indicarlo) e **inferencia**.

Por cada hallazgo aportar:

| Campo | Contenido exigido |
|---|---|
| Identificador del revisor | Código local estable, sin asignar S32, RETP ni números canónicos |
| Afirmación examinada | Texto o paráfrasis precisa, con alcance |
| Localización | Commit completo, ruta, revisión y apartado |
| Evidencia | Hechos observados y enlaces accesibles |
| Razonamiento o contraejemplo | Cadena verificable que sustente la objeción |
| Impacto y alcance | Decisión afectada; evitar extrapolación global |
| Grado de certeza | Confirmado documentalmente, plausible o no determinable, con motivo |
| Comprobación propuesta | Evidencia o contraste concreto necesario; sin ejecutar trabajos no autorizados |

Responder las ocho preguntas incluso si no hay hallazgos. Clasificar cada conclusión como respaldada en el alcance público, objeción sustentada o no comprobable con el expediente. Separar hallazgos de mejoras opcionales; no producir una puntuación agregada que oculte limitaciones. «No comprobable» no es «conforme» ni «defectuoso».

Terminar con límites, cuestiones pendientes y recomendación motivada a la Dirección: elevar, no elevar o solicitar evidencia concreta. No atribuir aceptación a coordinación. La ausencia de hallazgos no certifica seguridad o exhaustividad.

## Entrega y preservación

La ruta concreta está en el LEAME del revisor. Si no hay escritura expresamente autorizada, devolver texto o archivo a la Dirección para depósito por U-DOC-PUBLICA, sin simular acceso al repositorio. Con permiso, sólo añadir la entrega propia. Corregir en entrega-02, nunca reemplazar entrega-01. Conservar original literal y resumen separados.

No consultar deliberadamente la respuesta del otro antes de la primera entrega; declarar exposición previa si ocurrió. Esta revisión pública no se considera ciega ni plenamente independiente por defecto. No publicar datos privados ni detalles ajenos al alcance. La Dirección decide continuación y elevación.
