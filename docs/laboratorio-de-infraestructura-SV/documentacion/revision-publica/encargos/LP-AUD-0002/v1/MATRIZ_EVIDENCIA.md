# Matriz de afirmaciones y evidencia exigible

**Fuentes receptoras:** F07 (parte §§15–16), F08–F10 (Sucesos), F11–F12 (RETP), F06/F13 (cabeceras). Ver [inventario](FUENTES.tsv). Las demandas siguientes son criterios de revisión, no pruebas ejecutadas.

| Preguntas | Afirmación examinable públicamente | Evidencia directa que falta para juicio material | Resultado permitido sin ella |
|---|---|---|---|
| Q01/Q07 | Recuperación Linux, identidad de 12 485 entradas y banco 19/19 según §16.2 | Fuente exacta del cliente, dependencias/versiones, manifiesto, entradas, comandos y salida literal | Concordancia documental de la recepción; no recuperación propia |
| Q02 | FFI nativa intraproceso | Fuente de la frontera y ABI, biblioteca cargada, modelo de fallos, contrato de memoria y pruebas discriminantes | Identificar dependencia y límite; no certificar seguridad de memoria o aislamiento |
| Q03 | RCR-01 abierta; CR08/missing elimina todos los eventos | Contrato CR08 completo, caso positivo, mutación exacta, observador, causa de rechazo, oráculo precomprometido y resultado | Evaluar insuficiencia lógica declarada; no declarar complemento superado |
| Q04 | RCR-03 resuelta; RCR-02 histórica | Registro inicial, cadena de custodia y evidencia independiente disponibles | Conservar la reserva; no reconstruir lo ausente |
| Q05 | AUX-C01–C04 pendientes | Contratos, mecanismos impuestos, permisos efectivos, límites de recursos/egreso y pruebas positivas/negativas | Enumerar condiciones y carencias; no dar por implantada contención |
| Q06 | Perfiles de dominio/finalidad/expediente requeridos | Política decidida por autoridad competente, mapa de permisos, destinos y minimización, contratos y observación de denegaciones | Revisar separación de competencias; no conceder permisos ni habilitar fuentes |
| Q08 | Cotejo de nueve commits en alcance | Los dos públicos son accesibles; para siete privados faltan autorización/árboles originales | Reexaminar sólo dos; atribuir el resto a la recepción sin ampliar a toda la historia |
| Q09 | protected=false observado en endpoint | Reglas/rulesets y permisos efectivos del actor y herramienta, examen autorizado | Describir campo observado; no afirmar ausencia exhaustiva de protección |
| Q10 | Selección, aceptación y producción pendientes | Decisión humana y evidencias exigibles para el alcance retenido | Mantener reservas; no seleccionar o integrar el cliente |

**Acceso del revisor:** una columna final de su respuesta indicará leído, inaccesible, no suministrado o no examinado para cada evidencia pertinente. Registrar motivo y consecuencias; nunca inferir conformidad de una casilla vacía.
