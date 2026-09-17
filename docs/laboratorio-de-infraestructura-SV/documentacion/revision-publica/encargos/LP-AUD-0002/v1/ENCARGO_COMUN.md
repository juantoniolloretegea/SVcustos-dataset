# LP-AUD-0002/v1 · Auditoría preventiva del auxiliar y del cliente Rust

**Preparación:** autorizada por la Dirección el 17/09/2026 bajo LP-DOC-0001/v1 + A01 y la solicitud específica de auditoría. **Ejecución externa y transmisión:** pendientes de revisión y autorización humana del enlace exacto. **Publicación documental:** exclusivamente SVcustos-dataset, rama laboratorio-publico.

## 1. Objeto y límites

Examinar, antes de seleccionar o integrar el cliente, la coherencia y suficiencia de las afirmaciones públicas sobre frontera Rust/FFI, CR08, contención, permisos por dominio y correspondencia entre pruebas y conclusiones.

Corte canónico: `bdb094958da5be00d2c24864c333deaa4f0636cc`, S32 revisión 8 / RETP-2026-257. Antecedente público: `206c405184a33884cdf0d19017b8f31d0f398340`. LP-AUD-0001/v1 conserva su corte S32 revisión 6 y no se sustituye ni modifica.

El corpus público permite auditar la recepción y sus límites. **No permite por sí solo auditar íntegramente el código del cliente, la biblioteca enlazada, CR08 o los registros privados.** La falta de esas evidencias no podrá convertirse en conformidad, aceptación o ausencia de defectos. El revisor deberá delimitar las conclusiones que no puede emitir.

El destinatario prioritario propuesto es Claude. Se conserva un encargo equivalente de Grok conforme al mandato; su preparación no dispone una segunda ejecución. Ningún revisor se invoca o contacta automáticamente.

## 2. Lectura y fijación del expediente

1. Registrar el commit completo del enlace transmitido y la autorización humana recibida.
2. Leer [fuentes fijadas](FUENTES.tsv), [manifiesto](MANIFIESTO_SHA256.tsv), [análisis preparatorio](ANALISIS_PREPARATORIO.md), [matriz de evidencia](MATRIZ_EVIDENCIA.md), [referencias reservadas](REFERENCIAS_RESERVADAS.md) y [protocolo de entrega](PROTOCOLO_ENTREGA.md).
3. Leer AGENTS, Pilares, Perfiles y Transición en ese orden; después Acta 001, parte S32 §§13–16, registros S32/RETP-256/257 y cabeceras vigentes. Diferenciar antecedentes y actualización.
4. Para los perímetros públicos, examinar los dos commits e índices fijados. No atribuir nueva inspección a los siete commits privados que no se han podido leer.
5. Declarar qué se ha leído realmente, qué no ha podido recuperarse y si hubo exposición a otra revisión. No presentar el análisis preparatorio como hallazgo propio.

No reemplazar commits por main o latest. Una actualización material de fuentes requiere nueva revisión y autorización. Se permite consultar documentación oficial adicional indispensable para interpretar una afirmación, identificándola por versión y fecha; no se amplía el corpus experimental por intuición.

## 3. Preguntas obligatorias

| Código local | Pregunta y criterio documental |
|---|---|
| Q01 · Identidad y alcance | ¿Se distinguen AUX-DUCKDB-NCBI-01 y AUX-CLIENTE-RUST-02, sus versiones, productores, receptores, paquetes y plataformas? ¿Las 12 485 entradas y el banco 19/19 se atribuyen sólo a su recepción? |
| Q02 · Frontera Rust/FFI | ¿Se evita inferir aislamiento del motor a partir de guardas Rust? Identificar qué prueba o documento sería necesario para juzgar carga nativa, ABI, memoria, fallos del proceso y capacidades efectivas. Sin fuente exacta no certificar el código ni la versión de DuckDB. |
| Q03 · CR08 / RCR-01 | ¿Eliminar todos los eventos sustenta la propiedad de detectar la omisión de cada evento requerido? Delimitar el complemento discriminante, control válido, única diferencia, causa de rechazo y oráculo previo; no cerrar RCR-01 por el 19/19 ni por proponer un ensayo. |
| Q04 · RCR-02 / RCR-03 | ¿Se preserva la diferencia entre recuperación instrumental resuelta y límites históricos de captura/independencia? ¿Se conserva el fallo CR06 y su reparación sin reconstrucción retrospectiva? |
| Q05 · Contención | ¿AUX-C01–C04 tienen un resultado acreditado o siguen siendo condiciones pendientes? Separar supervisor y permisos, carga/frontera nativa, transporte externo y recursos/salida. Cargo --offline no demuestra aislamiento de red del sistema operativo. |
| Q06 · Autorización por dominio | ¿Se requiere autorización ligada a dominio, finalidad y expediente, recurso, operación y destino? ¿Se evita trasladar permisos entre inmunología y ciberseguridad, exportar secretos o inferir permiso de egreso por agregación? No inventar un contrato ya constituido. |
| Q07 · Prueba y conclusión | ¿Se separan los siete subcasos del auxiliar, 19/19 del banco del cliente, 18 controles H2 y los 19 TLC no ejecutados? ¿Se distinguen compilación, recuperación, identidad, causalidad, contención y aceptación integral? |
| Q08 · Trazabilidad y escritura | ¿Qué acreditan los nueve commits examinados y qué queda fuera? Reexaminar sólo los públicos accesibles; limitar la atribución de los privados a la recepción. Verificar rama/ruta asignada para la propia entrega y declarar impedimentos de escritura. |
| Q09 · Ramas y protecciones | ¿Se limita protected=false al endpoint observado? No convertir inventario, edad o muestreo en prueba exhaustiva de permisos, autorización o abandono; no recomendar borrado automático ni cambiar reglas. |
| Q10 · Continuidad y aceptación | ¿Se mantienen S32/BIS-03 abiertos, H2 candidato, durabilidad no implementada, aceptación íntegra/selección pendientes y exclusiones de producción, datos reales y entrenamiento federado? Precisar qué evidencia falta antes de una decisión posterior. |

Las hipótesis de ataque o casos propuestos son revisión documental; no autorizan ejecutarlos. No realizar campañas, recompilar el cliente, instalar herramientas, descargar ZIP, usar NCBI, credenciales, datos reales, Qwen o servicios externos de pago.

## 4. Rúbrica y entrega

Utilizar [PLANTILLA_RESPUESTA.md](PLANTILLA_RESPUESTA.md). Responder Q01–Q10 aunque no se hallen defectos. Por cada conclusión usar una de estas categorías y explicar su alcance: respaldo documental, objeción sustentada, no comprobable con el expediente, propuesta de mejora.

Cada hallazgo incluirá identificador local del revisor, afirmación examinada, commit completo/ruta/revisión/apartado, evidencia, razonamiento o contraejemplo, impacto, alcance, grado de certeza y comprobación propuesta. No asignar números S32, RETP o LP-HAL canónicos por cuenta propia.

Separar lectura, reproducción propia e inferencia. Si no hubo reproducción, escribir «No he ejecutado ni reproducido el código o el banco». No afirmar verificación de paquetes a partir de huellas citadas por terceros. Un resumen no sustituye al original literal.

Añadir una matriz de evidencia inaccesible con objeto/referencia, causa observada o desconocida, efecto sobre cada conclusión y evidencia mínima requerida. Las reservas ya documentadas se identificarán como tales, sin presentarlas como descubrimiento nuevo.

El dictamen debe distinguir:
- conformidad de la presentación documental examinada;
- suficiencia del diseño que pueda juzgarse con fuentes disponibles;
- verificación material del cliente y controles, que permanece no comprobable cuando falte evidencia;
- recomendación motivada a la Dirección, sin aceptación canónica ni selección del cliente.

Una puntuación global o «sin hallazgos» no sustituye estas distinciones. La recomendación de elevar no significa que se haya transmitido o aceptado el asunto por otra unidad.

## 5. Escritura y parada

Cumplir el [protocolo de rama, permisos y entrega](PROTOCOLO_ENTREGA.md). Sólo con autorización humana expresa de escritura podrá añadirse la entrega propia en la ruta del LEAME particular, dentro de laboratorio-publico. Nunca usar main, la rama beta, otra rama, un fork o una PR como alternativa a una limitación de la herramienta.

Si no se puede escribir, declararlo expresamente y devolver a la Dirección el texto o archivo de la respuesta. No simular commit o publicación. La ausencia de escritura no impide completar la lectura permitida; la falta de evidencia sí limita las conclusiones.

Detenerse tras devolver la respuesta o publicar únicamente su entrega autorizada y verificada. No corregir el cliente, CR08, registros canónicos, otros encargos, ramas o protecciones.
