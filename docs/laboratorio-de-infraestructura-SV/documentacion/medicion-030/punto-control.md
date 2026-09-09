# Punto de control de rendimiento y vigilancia de las evoluciones Beta del Lenguaje SV

**Registro:** LAB-2026-030 · **Fecha:** 9 de septiembre de 2026.

**Estado:** medición exploratoria externa recibida y revisada; reproducción independiente pendiente. No hay necesidad de refactorización acreditada por este expediente.

## 1. Mandato, alcance y gobierno

Juan Antonio Lloret Egea autoriza registrar el punto corregido y su ficha en la documentación del laboratorio, como referencia para las futuras evoluciones Beta. La finalidad es preservar el trabajo y detectar costes o regresiones antes de que se conviertan en problemas. No se establece una ventana obligatoria de refactorización ni se autoriza modificar el núcleo por razones de organización, tamaño de fichero o preferencia estética.

Una refactorización sólo podrá proponerse ante un defecto concreto acreditado, con su causa, consecuencia, alternativa mínima y evidencia antes/después; requiere autorización humana expresa y separada. Los resultados favorables respaldan conservar la realización dentro del alcance medido. Ninguna prueba finita garantiza que nunca aparezca una necesidad futura; un dato ausente o no resoluble no se contabiliza como conformidad ni como defecto del núcleo.

Esta inscripción no modifica gramática, IR, operaciones, perfiles, constituciones IMM/CYB, R1, plataforma ni secuencia de cierre. Los relojes del laboratorio son instrumentos externos de medida, no primitivas del SV. La continuación del Lenguaje es la localización diagnóstica ES/EN ya autorizada, antes de la recepción profesional R1.

## 2. Identidad y procedencia

| Objeto | Identidad y alcance |
|---|---|
| Informe recibido | «Medición de rendimiento del núcleo SV y procedimiento de la ventana de refactorización», atribuido a Claude por la Dirección, 09/09/2026. |
| Original bajo custodia | `030-informe-recibido.md`, 12 461 bytes; SHA-256 `6aa75bdd7be1a7f98e2bd6bc77a87d1fc8d1d1c6f12ca7a9a62ee5029c2cd084`. Se conserva sin corregirlo retrospectivamente. |
| Corte declarado de medición | Lenguaje `0f434dbb6f6ea8f5f4e270c58c4ad4c5490d8c84`; árbol comprobado `604a4f8a44cf9dae7e027008c9be2a2b3168935e`. |
| Binario declarado | `rust/target/release/sv-native`, 1 008 952 bytes; SHA-256 declarado `cb0db4112cea29ed01972f88a549b0add8bb43a261034f0e8f37cb04974b8e7a`. Esta recepción no verifica los bytes del binario utilizado por el autor. |
| Construcción declarada | `release`; rustc 1.98.0 (88d9e12ae 2026-08-18); cargo 1.98.0 (797e8a9bc 2026-08-05). Comando completo y opciones adicionales no entregados. |
| Entorno declarado | Linux 6.18.44-fc-v24 x86_64; Intel Xeon a 2,10 GHz; dos CPU; contenedor compartido, sin afinidad ni control de otras cargas. No se han verificado independientemente modelo completo ni cuotas efectivas. |
| Protocolo declarado | Cinco calentamientos y 40 ejecuciones por punto; mediana y desviación absoluta mediana (MAD). Muestras individuales y programa adicional de análisis no entregados. |
| Corte de continuidad del Lenguaje | `39203889e2896389907e9a2ddffec0f60dc259e5`, PR #89 en borrador; realización diagnóstica pendiente. No es el corte medido. |
| Bases de esta recepción | Laboratorio `673a81abf3046eba4437c0b109c3667d00fc7f08`; publicación Beta `97c91ec2b64e1cee6b2fe9697eb051dd5c49ddf0`. |

El estado histórico de PR #88 consignado en el original no describe el estado posterior de integración. La inscripción conserva el corte de la medición, sin trasladar sus tiempos a otra cabeza.

Se consultaron los Pilares completos, el acta de perfiles completa y la transición secuencial con sus relevos hasta §30 en el corte de continuidad del Lenguaje; el contrato diagnóstico y RETP-111/112; y el índice y régimen de evidencia pública del laboratorio. La medición externa no adquiere autoridad doctrinal.

## 3. Magnitud realmente observada

El arnés citado mide la duración de `subprocess.run([sv-native, fuente])`, con stdout y stderr capturados. La CLI lee la fuente, compila, serializa la IR y escribe el resultado antes de terminar. La duración incluye creación y terminación de proceso, lectura, compilación, serialización y transporte de salida. La generación de fuentes y el cálculo de la huella de salida quedan fuera del intervalo cronometrado del arnés.

Por tanto, la referencia de 106 bytes, mediana declarada **1,450 ms** y MAD declarada **0,096 ms**, no mide exclusivamente el arranque. Restarla estima una diferencia respecto de esa carga; no aísla el núcleo. Los tiempos siguientes son las diferencias publicadas por el autor, no tiempos totales reproducidos en esta recepción. No se reconstruyen las medianas brutas sumando la referencia, porque el tratamiento de los ceros no está documentado.

El umbral **0,480 ms = 5 × MAD de la referencia** se conserva como criterio exploratorio del original, no como presupuesto SV, intervalo de confianza ni límite de detección acreditado. La incertidumbre de la resta incluye cada carga y su referencia. Los ceros requieren aclarar redondeo, recorte o censura. El ruido de un entorno compartido puede afectar tanto a valores absolutos como a la forma de la curva.

## 4. Tablas recibidas de la medición de hoy

La columna de clasificación sólo aplica el criterio exploratorio anterior. «Supera» no significa conformidad de rendimiento ni suficiencia estadística.

| Familia | Escala | Bytes de fuente | Diferencia publicada (ms) | Supera 0,480 ms |
|---|---:|---:|---:|---|
| N02, cantidad de objetos | 1 | 27 | 0,000 | No |
| N02, cantidad de objetos | 100 | 2790 | 0,111 | No |
| N02, cantidad de objetos | 500 | 14390 | 1,009 | Sí |
| N02, cantidad de objetos | 2000 | 58890 | 3,814 | Sí |
| N02, cantidad de objetos | 5000 | 148890 | 9,397 | Sí |
| N03, nodos del marco | 1 | 720 | 0,000 | No |
| N03, nodos del marco | 100 | 22257 | 0,828 | Sí |
| N03, nodos del marco | 250 | 55707 | 2,185 | Sí |
| N03, nodos del marco | 1000 | 222957 | 9,102 | Sí |
| N03, nodos del marco | 2000 | 450957 | 19,804 | Sí |
| N01, dígitos de parameter_id | 1 | 106 | 0,000 | No |
| N01, dígitos de parameter_id | 1000 | 1105 | 0,000 | No |
| N01, dígitos de parameter_id | 5000 | 5105 | 0,000 | No |
| N01, dígitos de parameter_id | 20000 | 20105 | 0,079 | No |
| N01, dígitos de parameter_id | 50000 | 50105 | 0,348 | No |
| N02, longitud de identificador | 8 | 33 | 0,000 | No |
| N02, longitud de identificador | 1024 | 1049 | 0,000 | No |
| N02, longitud de identificador | 4096 | 4121 | 0,000 | No |
| N02, longitud de identificador | 16384 | 16409 | 0,065 | No |
| N02, longitud de identificador | 65536 | 65561 | 0,654 | Sí |

## 5. Recálculo y lectura corregida

Se ajusta por mínimos cuadrados `ln(diferencia_ms) = intercepto + exponente × ln(tamaño)` sólo con los puntos que superan 0,480 ms. Se publican por separado ajustes frente a escala y frente a bytes. Los cálculos proceden de cifras redondeadas del informe, no de sus muestras. El verificador público permite repetir esta aritmética.

| Familia | Puntos | Exponente / escala | Exponente / bytes | Coste inicial → final (ns/byte) | Recorrido de escala |
|---|---:|---:|---:|---:|---:|
| N02, cantidad de objetos | 3 | 0,968297 | 0,954035 | 70,118 → 63,114 | ×10 |
| N03, nodos del marco | 4 | 1,053915 | 1,050236 | 37,202 → 43,915 | ×20 |
| N01, dígitos | 0 | No estimado | No estimado | No caracterizado | No caracterizado |
| N02, identificador | 1 | No estimado | No estimado | Un punto no define curva | No caracterizado |

Las dos primeras familias muestran crecimiento aproximadamente proporcional en el intervalo ensayado. No demuestran complejidad lineal asintótica ni ausencia de cuellos de botella fuera del ensayo. El tramo útil es de una década de escala en N02 y aproximadamente 1,301 décadas en N03; no son dos órdenes de magnitud. En N03, el coste por byte crece aproximadamente **18,047 %** entre extremos; su significación requiere la dispersión de las muestras.

Los caudales derivados de los últimos puntos son aproximadamente 15,8 y 22,8 MB/s decimales de fuente por tiempo incremental. No se presentan como capacidad general del compilador ni del núcleo. El identificador de longitud 65536 sí supera el criterio exploratorio, pero un solo punto no caracteriza su crecimiento.

El generador N03 mantiene un grafo sin aristas, una CellSpec sintética de b=3 y vacíos los resultados de evaluación, puertas, supervisión y criticidades. El generador N01 varía parameter_id, no toda la aritmética Nat. Los programas generados son EN. No se ha medido aquí ensamblaje ES/EN, ejecución profesional IMM/CYB, LIG, álgebra material, WASI ni navegador. La sonda sintética no constituye geometría, agente o arquitectura de un dominio.

**Dictamen:** este expediente no acredita un cuello de botella que justifique refactorizar. Tampoco acredita suficiencia universal. Mantener la realización y observar sus evoluciones es la decisión respaldada en este alcance.

## 6. Seguimiento mínimo para futuras evoluciones Beta

Antes de medir una candidata, fijar entradas, instrumento, oráculos y criterios aplicables en un corte identificable. No escoger después los tamaños o límites para obtener un resultado favorable. La ausencia de presupuesto impide declarar su cumplimiento; no impide detectar una regresión reproducible frente a una referencia comparable.

| Control | Medida mínima | Decisión y límite |
|---|---|---|
| Identidad y validez | Corte, árbol, fuentes y binarios con huellas; comandos y opciones; aceptación/rechazo y resultado esperado | Una entrada o un artefacto distinto se registra por separado; no se atribuye una mejora a haber omitido trabajo o guardas. |
| CLI nativa | Duración total, muestras, calentamientos, mediana, MAD y percentiles con número de muestras y método | Conservar totales; las diferencias son auxiliares y conservan incertidumbre y signo. |
| Recursos | CPU de usuario/sistema y memoria residente máxima; bytes de entrada y salida, tamaño de binario | El arnés original ya recoge CPU/RSS en ejecuciones separadas. Declarar esa separación y sus políticas de salida; un tiempo favorable no sustituye estas medidas. |
| Escalado | Objetos, longitudes, nodos y, cuando afecten a la modificación, aristas, estados, operaciones y ensamblajes | No extrapolar una familia vacía o sintética a todo el trabajo. Conservar puntos intermedios y residuos del ajuste. |
| Comparación | Referencia y candidata bajo condiciones comparables, cargas idénticas e intercalación/orden declarado | Investigar una diferencia reproducible y su efecto práctico; no imponer un porcentaje universal sin justificarlo antes. Entorno local o CI controlado pueden servir. |
| Perfiles ES/EN | Casos equivalentes y ensamblaje mixto, incluidos rechazo temprano y tardío cuando sean pertinentes | Mismo juicio y causa canónica; prosa, tamaño y huella pueden diferir legítimamente. La localización añade trabajo y se declara ese cambio funcional. |
| Destinos | Nativo, WASI y navegador por separado; inicialización/carga y ejecución estable diferenciadas | No trasladar una basal entre destinos ni declarar probado uno por el resultado de otro. |
| Corrección | Corpus y mutaciones vigentes, oráculos de decisión, procedencia y presentación | Los recuentos históricos no congelan el corpus. Investigar supervivientes y aplicabilidad; el porcentaje de mutación no es una certificación de seguridad. |
| Resolución insuficiente | Aumentar señal o usar, si procede, un banco dentro del proceso con entrada y resultado efectivamente consumidos | Es otro instrumento; delimitar compilación, serialización, asignación/liberación y cachés. No mezclar sus cifras con las de proceso. |

Estos controles acompañan cambios pertinentes y recepciones Beta; no ordenan repetir indiscriminadamente todas las campañas por una edición documental. La evidencia afectada se decide por el alcance del cambio. No se introduce dependencia productiva ni se requiere instalar una biblioteca de medición en el núcleo.

Si aparece una desviación, primero se verifica el instrumento, se reproduce, se localiza el coste y se documenta la consecuencia. La corrección mínima y su prueba discriminante preceden a cualquier propuesta de reorganización amplia. Medir un crecimiento superlineal no autoriza por sí solo una refactorización; medir crecimiento lineal tampoco descarta un coste excesivo.

## 7. Pendientes y alcance de la comprobación pública

Pendientes del informe recibido: 40 muestras de cada punto y de la referencia; orden de ejecución; tiempos totales; significado de ceros; dispersión por carga; script exacto ampliado y de análisis; opciones de construcción; fuentes y salidas capturadas; confirmación independiente del binario y entorno; CPU/RSS de esa ejecución.

Los tamaños de archivos, proporción de código de pruebas y avisos de compilador del original son datos de estructura declarados, no métricas de velocidad ni de memoria en producción. No se incorporan como alarma de refactorización ni se presentan aquí como censo reproducido.

El paquete público de este registro contiene manifiesto, especificación, verificador y estado, más este punto Markdown. Su alcance es identidad documental, transcripción y recálculo de tablas, y ausencia de atribuciones de ejecución o suficiencia no acreditadas. No reproduce tiempos, no ejecuta SV y no convierte datos declarados en medidas independientes. La huella del manifiesto queda en el asiento de publicación externo, sin autorreferencia circular. Los paneles de campañas anteriores mantienen sus recuentos.

## 8. Fuentes y continuidad

- [Arnés exacto R0-8](https://github.com/juantoniolloretegea/SV-lenguaje-de-computacion/blob/0f434dbb6f6ea8f5f4e270c58c4ad4c5490d8c84/tests/r0_8_scale.py) y [CLI del mismo corte](https://github.com/juantoniolloretegea/SV-lenguaje-de-computacion/blob/0f434dbb6f6ea8f5f4e270c58c4ad4c5490d8c84/rust/sv_native/src/main.rs).
- [Pilares](https://github.com/juantoniolloretegea/SV-lenguaje-de-computacion/blob/39203889e2896389907e9a2ddffec0f60dc259e5/docs/calidad/PILARES_Y_RESTRICCIONES_DE_DISENO_DEL_LENGUAJE_DE_COMPUTACION_SV_2026_09_05.md), [perfiles](https://github.com/juantoniolloretegea/SV-lenguaje-de-computacion/blob/39203889e2896389907e9a2ddffec0f60dc259e5/docs/calidad/ACTA_TECNICA_DE_PERFILES_CONTRATOS_Y_ENSAMBLAJE_DEL_LENGUAJE_SV_2026_09_06.md) y [secuencia rectora](https://github.com/juantoniolloretegea/SV-lenguaje-de-computacion/blob/39203889e2896389907e9a2ddffec0f60dc259e5/docs/dominios/inmunologia/ACTA_DE_CONFORMIDAD_DE_TRANSICION_SECUENCIAL_DESDE_OP-IMM-001_AL_LENGUAJE_SV_2026_09_03.md#adenda-secuencia-20260906).
- [Contrato de localización ES/EN](https://github.com/juantoniolloretegea/SV-lenguaje-de-computacion/blob/39203889e2896389907e9a2ddffec0f60dc259e5/docs/calidad/CONTRATO_DIAGNOSTICO_ESTRUCTURADO_Y_LOCALIZACION_ES_EN_2026_09_09.md) e [inventario RETP-112](https://github.com/juantoniolloretegea/SV-lenguaje-de-computacion/blob/39203889e2896389907e9a2ddffec0f60dc259e5/docs/calidad/INVENTARIO_DE_EMISORES_DIAGNOSTICOS_Y_MIGRACION_ES_EN_2026_09_09.md).
- [Variabilidad de medición, Google Benchmark](https://google.github.io/benchmark/reducing_variance.html) y [precauciones de medición, LLVM](https://llvm.org/docs/Benchmarking.html). Referencias metodológicas, sin adopción de dependencias.
- [Documentación del laboratorio](https://juantoniolloretegea.github.io/SVcustos-dataset/laboratorio-de-infraestructura-SV/documentacion/) y [ficha de este punto](https://juantoniolloretegea.github.io/SVcustos-dataset/laboratorio-de-infraestructura-SV/documentacion/punto-control-rendimiento.html).

Tras este asiento se retoma la localización desde PR #89. Esta recepción documental no prueba la localización ni autoriza su despliegue, no vincula referencias protegidas R1 y no constituye una operación de los dominios.
