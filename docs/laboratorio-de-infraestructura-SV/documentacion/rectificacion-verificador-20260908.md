# Rectificación del verificador público · 8 de septiembre de 2026

Esta revisión sucede al paquete publicado en `SVcustos-dataset@801030958e2d0af7a4e7e90b7b8371dd6f0f1930`. Aplica el régimen de evidencia pública vigente a H-1, H-2 y H-3. No modifica las campañas ni sustituye sus resultados.

## Obligaciones del contrato

1. **Aprobación.** Esta regularización sólo admite `approval = NOT_GRANTED`. Se rechazan la aprobación concedida, un valor arbitrario y la ausencia del campo. Un panel sin fallos tampoco otorga autoridad humana. Una aprobación futura requiere su acto competente y un contrato que lo identifique; este verificador no la constituye.
2. **Límites.** LVP-01…LVP-06 tienen identidad y texto fijados en `testigos.json` y en el contrato ejecutado. El estado debe reproducir ambos. Se rechaza retirar un límite, repetir su identidad o debilitar su contenido, incluso cuando se modifica simultáneamente la especificación publicada.
3. **Fuentes.** Las referencias de la regularización deben ser confirmaciones Git de cuarenta cifras hexadecimales, no nulas, y concordar entre manifiesto, especificación y estado. Esta comprobación no acredita públicamente la existencia o el contenido de la confirmación privada.

Los cuatro límites anteriores conservan su texto íntegro. LVP-05 precisa el alcance de las referencias privadas. LVP-06 exige distinguir integridad, autenticidad externa y aprobación humana.

## Pruebas causales

Antes de corregir, el control reserializado y las seis variantes externas de H-1/H-2/H-3 fueron aceptados. Después, el control permanece aceptado y las seis variantes se rechazan por la causa declarada, con retorno 1 y sin informe de conformidad en la salida normal.

La autoprueba permanente contiene 32 mutaciones dirigidas: las veinte anteriores y doce de esta revisión. Cada copia se serializa de nuevo y sus huellas se recalculan en el manifiesto antes de atravesar la comprobación completa. Un rechazo por huella discordante no satisface una expectativa de rechazo semántico. Las 32 mutaciones se rechazan por su causa exacta; se conservan los 65 controles de ausencia de observables.

Además, un control cambia concordantemente la referencia privada en los tres archivos usando un valor de formato válido. La verificación sin referencia externa conserva expresamente `private_source_authenticated_by_public_reader = false`. La comprobación contra la huella externa original rechaza el paquete sustituido. Este control demuestra el límite; no se cuenta como autenticación pública de la custodia.

## Referencia externa

El auditor puede aportar una huella obtenida de una confirmación de publicación cuya procedencia haya comprobado:

```sh
node verificar.mjs --autoprueba --manifest-sha256=HUELLA_SHA256_DE_CONFIANZA
```

También puede añadir al comando la URL del directorio publicado. El verificador ejecutado debe coincidir con los bytes identificados por el paquete. Descargar un verificador y su manifiesto de una misma fuente sin autenticar no crea por sí solo una referencia de confianza.

Huella del manifiesto de esta revisión: `cfb20291ef34e0a17203880c8c675303a0bde0ef7fa40aaeb6291e3d84bb8c83`.

## Alcance conservado

El panel conserva 55 PASS, 1 FAIL, 1 BLOCKED y 1 ERROR; la conciliación separada de la ejecución #8 conserva 16 PASS y 1 ERROR. Permanecen las 74 identidades de testigo y las 6.472 identificaciones de artefactos privados. No se ha repetido ninguna campaña. La revisión no cierra garantías completas, la fila 7 ni el núcleo. La custodia y su visibilidad permanecen intactas.

El asiento de custodia LAB-2026-029 conserva el par CSV/Markdown, las fuentes de las comprobaciones, sus salidas y las huellas del paquete. El paquete público permite reproducir la autoprueba sin acceso a ese asiento.
