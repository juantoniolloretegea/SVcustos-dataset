# Régimen de evidencia pública verificable del laboratorio

Fecha: 8 de septiembre de 2026. Disposición estructural autorizada por la Dirección. Registro vinculado: LAB-2026-027, en Markdown y CSV. Aplicación: próxima tanda y recepción normativa de antecedentes. Custodia privada conservada.

Toda campaña del laboratorio de infraestructura SV emitirá, junto a su custodia privada, un paquete público de verificación. Una campaña sin ese paquete disponible no es recibible por ninguna sede: no se promueve, no se cita como evidencia normativa y no cierra una fila. La ausencia del paquete es defecto de la campaña, no del auditor. Su presencia es una condición necesaria; no demuestra por sí sola la suficiencia de todas las propiedades alegadas.

## Cuatro objetos obligatorios

En una misma ruta pública se publicarán:

| Objeto | Obligación |
| --- | --- |
| `manifiesto.json` | Nombre relativo, tamaño en bytes y SHA-256 de cada artefacto; confirmación de fuente e identificador de ejecución. El artefacto privado permanece privado. Cada tanda conserva un inventario íntegro, incluido el archivo cerrado y su manifiesto interno. |
| `testigos.json` | Especificación separada del resultado: obligación, condición atacada, oráculo y control negativo de cada testigo. Las observaciones no se utilizan para fabricar expectativas. |
| `verificar.mjs` | Verificador sin acceso privado. Deriva sus predicados de la especificación; comprueba aritmética, identidades, fuentes y coherencia de `cases`, `pending` y `state`; rechaza las mutaciones negativas declaradas. |
| `estado.json` | Resultados y observaciones públicas seleccionadas, sometidos al verificador. Distingue comprobación pública, declaración identificada bajo custodia, fallo, bloqueo y ausencia de ejecución. |

La confirmación Git de publicación y la huella del manifiesto se fijarán en el asiento de recepción. Los archivos se conservarán también en custodia privada. El manifiesto puede identificar los otros tres archivos; su propia identidad debe apoyarse en una referencia externa, evitando una autorreferencia criptográfica circular.

## Expectativas y momento de constitución

Los resultados esperados proceden de la obligación y del testigo, nunca de lo que emita el ejecutor. Antes de medir se fijarán en una confirmación identificada la especificación, las entradas, los oráculos, las mutaciones negativas y cualquier umbral, ventana o presupuesto. Alterar esas condiciones después de conocer los resultados invalida la tanda a esos efectos. El expediente conservará tanto la declaración anterior a la ejecución como el corte efectivamente ejecutado.

No basta que dos realizaciones o dos destinos produzcan el mismo resultado. Tampoco basta que dos diagnósticos sean iguales, diferentes o tengan identificadores únicos. Cada control debe alcanzar la obligación que pretende comprobar. La sensibilidad del verificador público se declara separadamente de la sensibilidad del ejecutor privado.

## Identidad y sucesión

Un identificador nombra un testigo y sólo uno. Al repetirse en otra tanda, incorpora el sufijo de esa tanda. Se conserva el nombre histórico como `legacy_id`, sin utilizarlo como identidad global.

Un testigo superado por otro declara `superseded_by`, los sucesores exactos, la obligación que éstos cubren y sus límites. Un resultado FAIL, ERROR o BLOCKED retirado de pendientes necesita esa declaración; en caso contrario vuelve a `pending`. La sucesión no convierte el fallo histórico en PASS. Las obligaciones generales que sobreviven al testigo acotado permanecen en el inventario de pendientes.

## Custodia y alcance demostrable

No se publican los registros 005, 012, 016, 018 ni 020, las actas privadas, claves, capturas completas, contenido protegido ni rutas de host. Se publican únicamente los cuatro objetos y las proyecciones expresamente revisadas.

Una huella permite comprobar identidad si se dispone de los bytes. Sin ellos no acredita su contenido, su ejecución ni la correspondencia fuente–binario. El verificador declara qué observaciones puede contrastar públicamente y qué resultados sólo quedan identificados bajo custodia. Un consumidor no puede ampliar ese alcance por el simple resultado conforme del verificador.

Este régimen adopta el principio de contraste externo que motivó el retorno G/H: fuentes identificadas, expectativas separadas y ataques negativos reproducibles. No transfiere automáticamente al laboratorio cifras, resultados ni cierres del expediente inmunológico.

## Aplicación y recepción

1. Antes de la próxima tanda, constituir y fijar la especificación y los presupuestos, con fuente y fecha anteriores a la medición.
2. Ejecutar únicamente la campaña autorizada y cerrar su custodia privada con inventario completo.
3. Emitir los cuatro objetos públicos. El estado no puede fijar ni corregir sus propias expectativas.
4. Ejecutar el verificador y sus controles negativos en un directorio que sólo contenga los cuatro objetos. Conservar orden, retorno, salida y huellas de esa verificación.
5. Publicar en la superficie documental existente y verificar acceso anónimo y correspondencia de bytes. Registrar la confirmación de publicación y la huella del manifiesto.
6. La sede receptora comprueba el paquete sobre esa identidad exacta. Si falta un objeto, falla el verificador, no se puede acceder sin privilegios o la evidencia no alcanza la propiedad invocada, la campaña no es recibible para esa pretensión.

La regla no depende de GitHub Actions y alcanza las ejecuciones locales. No modifica infraestructura, hospedaje, permisos privados ni el reparto de competencias de las catorce filas. El cumplimiento queda sometido a la recepción anterior; no se afirma que exista una publicación automática desde el ejecutor privado.

Las campañas anteriores no se repiten por esta instrucción. Se regularizan cuando alguna sede las invoca como evidencia normativa. La regularización documental actual no fabrica un compromiso público anterior a las ejecuciones históricas. Los registros 016, 018 y 020 no quedan regularizados por haber conciliado ahora las tres tandas presentes en `estado.json`.

## Regularización inmediata

LAB-2026-027 conserva el panel con 55 PASS, 1 FAIL, 1 BLOCKED y 1 ERROR. Desambigua las identidades por tanda, documenta las sucesiones de A05 y A07 sin borrar los resultados antiguos y conserva aparte los 17 casos de #8: 16 PASS y 1 ERROR. Los 16 PASS de esa conciliación no vuelven a sumarse al total 55 del panel.

El primer paquete regularizado identifica 74 testigos y 6.472 artefactos de custodia de las ejecuciones 34010741314, 34021807895 y 34022055251. Ninguna de esas campañas vuelve a ejecutarse. Las pruebas nuevas corresponden exclusivamente a la consistencia y sensibilidad del verificador público.
