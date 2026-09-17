# Registro propio LP

Cierre del arranque: publicación comprobada; revisiones externas no iniciadas. Fechas no observadas permanecen vacías. Presentación concordante con el CSV; transiciones en HISTORIAL_LP.csv.

## LP-DOC-0001

| Campo | Valor |
|---|---|
| id | LP-DOC-0001 |
| objeto | Constitución de sede documental |
| version | v1 + A01 |
| fuente_autorizacion | Mandato v1 y adenda A01 transmitidos por la Dirección |
| fecha_autorizacion | 2026-09-17; hora no registrada |
| inicio_observado | 2026-09-17; hora inicial no registrada |
| entrega_observada | 2026-09-17T10:05:11.5552280Z; publicación comprobada y disponible para entrega |
| cortes_entrada | SVcustos dd50c3e5c74e10526bb28087c8d5e2852d2efb68; Lenguaje e1ab93d0de3df170a674f5a7887a3c20e96c6816; corte actualizado Lenguaje 7d6ecaa7d56b786ff194379417fef2ec2f0952df |
| evidencia | ENTREGA.md; VERIFICACION_PUBLICACION.tsv; commit 8e03fc2e0f150dd94d375049909a0980eb7161cc |
| resultado | Sede publicada y verificada; LP-AUD-0001 preparado |
| limites | Sin ejecución externa ni despliegue HTML; sin acceso privado |
| siguiente_actuacion | Entregar enlaces a la Dirección; detenerse hasta nueva instrucción |

## LP-AUD-0001

| Campo | Valor |
|---|---|
| id | LP-AUD-0001 |
| objeto | Coherencia de estado público y recepción S32 H2 |
| version | v1 |
| fuente_autorizacion | LP-DOC-0001/v1 + A01: preparación; ejecución por transmisión humana futura |
| fecha_autorizacion | 2026-09-17; hora no registrada |
| inicio_observado |  |
| entrega_observada |  |
| cortes_entrada | Lenguaje e1ab93d0de3df170a674f5a7887a3c20e96c6816; presentación dd50c3e5c74e10526bb28087c8d5e2852d2efb68; corte actualizado Lenguaje 7d6ecaa7d56b786ff194379417fef2ec2f0952df |
| evidencia | encargos/LP-AUD-0001/v1/ENCARGO_COMUN.md; FUENTES.tsv; MANIFIESTO_SHA256.tsv |
| resultado | Expediente publicado y verificado; revisiones no iniciadas; respuestas ausentes |
| limites | Lectura documental; originales H2 reservados; sin dictamen |
| siguiente_actuacion | La Dirección transmite cada enlace por commit cuando decida iniciar la revisión |

## LP-DOC-0002

| Campo | Valor |
|---|---|
| id | LP-DOC-0002 |
| objeto | Actualización y preparación de auditoría preventiva del auxiliar |
| version | v1 |
| fuente_autorizacion | Solicitud humana de 17/09/2026 bajo LP-DOC-0001/v1 + A01 |
| fecha_autorizacion | 2026-09-17; hora de autorización no registrada |
| inicio_observado | 2026-09-17T16:31:28.6900000Z; primer sello conservado de preparación |
| entrega_observada | 2026-09-17T16:45:14.0449070Z; publicación comprobada y disponible para entrega |
| cortes_entrada | Lenguaje bdb094958da5be00d2c24864c333deaa4f0636cc; laboratorio 206c405184a33884cdf0d19017b8f31d0f398340 |
| evidencia | encargos/LP-AUD-0002/v1/ANALISIS_PREPARATORIO.md; fichas/estado-canonico/v3.md; entregas/LP-DOC-0002/ENTREGA.md; commit 116b69eae43037a9159a1d7c483b0e585c4445c2 |
| resultado | Expediente publicado y verificado; disponible para revisión humana |
| limites | Sin acceso privado, reproducción material o transmisión externa |
| siguiente_actuacion | La Dirección revisa y autoriza la transmisión del enlace exacto; no iniciar revisores |

## LP-AUD-0002

| Campo | Valor |
|---|---|
| id | LP-AUD-0002 |
| objeto | Rust/FFI, CR08, contención, dominios y prueba-conclusión |
| version | v1 |
| fuente_autorizacion | Preparación solicitada por la Dirección; transmisión y ejecución externa pendientes |
| fecha_autorizacion | 2026-09-17; sólo preparación |
| inicio_observado |  |
| entrega_observada |  |
| cortes_entrada | S32 revisión 8 / RETP-2026-257 en bdb094958da5be00d2c24864c333deaa4f0636cc |
| evidencia | encargos/LP-AUD-0002/v1/ENCARGO_COMUN.md; FUENTES.tsv; MANIFIESTO_SHA256.tsv; PROTOCOLO_ENTREGA.md |
| resultado | Publicado y verificado; no transmitido; Claude prioritario propuesto; respuestas ausentes |
| limites | No acredita código privado, permisos de Claude ni aceptación material |
| siguiente_actuacion | Revisión y autorización humana del enlace exacto antes de transmitir; declaración expresa de acceso y escritura por el revisor |
