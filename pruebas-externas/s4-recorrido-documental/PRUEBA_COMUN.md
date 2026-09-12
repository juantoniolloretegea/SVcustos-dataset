# S4 · Prueba externa común de lectura, cobertura y presentación

Banco S4-EXTERNA-DOCUMENTAL/1. Este documento contiene todas las reglas, fuentes, casos, rúbrica y plantilla. Puede resolverse con el archivo adjunto o el texto completo, sin acceso al laboratorio ni permisos de escritura.

## Encargo común

Resuelva E01–E12. Por caso, entregue decisión, causa o resolución, consecuencia material, fuentes exactas, cita o diferencia verificable y justificación breve. Distinga lectura documental y actuación, y texto custodiado y archivo observado.

Indique modelo y versión visibles, plataforma, acceso por enlace/adjunto/texto, identificador de banco y si pudo leerlo íntegro. Declare modalidad: lectura_documental o ejecucion_propia. No necesita ejecutar Rust para la vía documental. Si ejecuta, aporte comandos, salidas, entorno y artefactos; no atribuya ejecución propia a leer resultados anteriores. Puede consultar antecedentes públicos y debe identificar las referencias adicionales utilizadas. Banco conocido, no evaluación ciega ni reserva inédita.

Use estas decisiones: DATO, PERMISO_REVOCADO, ENTREGAR_LECTURA, RECHAZAR, ACEPTAR_ARCHIVO, DETECTAR_ALTERACION, RECHAZAR_LECTURA, ERROR_LECTURA. En E01/E02, causa_o_resolucion recoge el estado y se indican contenido y llamadas de política. En los demás casos, recoja la causa técnica o null si no hay rechazo/error; contenido y llamadas pueden ser null por no ser objeto de esa respuesta.

Devuelva el JSON de la plantilla o una tabla con todos sus campos. Para tiempos, tokens, coste o herramientas sin medida disponible, use null y explique la ausencia. Aporte evidencias de actividad observables. No se penaliza la falta de herramientas declarada honestamente. Una respuesta correcta sin trazabilidad verificable no acredita conformidad.

## Reglas del banco

R01. A01 consulta inequívocamente LEER/CASO-A/IGG/ACTUAL/VALOR. El análisis completo produce una única ruta [1,1,1,1,1] y precede a política. Esta consulta alcanza una vez la puerta de política.

R02. P-IE004/1 admite LEER/CASO-A/IGG; K-IE004/1 contiene el literal artificial 8.40 para IGG/ACTUAL. El conductor fija la vigencia, ajena a la propuesta. La vigencia negativa resuelve PERMISO_REVOCADO sin contenido, fuente ni alcance. No constituye revocación profesional ni historia de autorización viva.

R03. Leer el recibo es distinto de actuar. Un recibo de resolución negativa puede leerse fielmente sin conceder permiso. El conductor elige la referencia antes de recibir la propuesta; se comprueban versión, operación e identidad antes del contenido. Una identidad ajena causa Identidad.

R04. La cobertura de estas posiciones exige caso original y entrada exacta de su montaje. Una cita verdadera puede ser insuficiente o ajena. Se distinguen FaltaCaso, FaltaVigencia, CasoDistinto y VigenciaDistinta. La propuesta no constituye la evidencia requerida.

R05. IE004-PRESENTACION-ESPACIOS/1 admite sólo espacio, TAB, LF y CR fuera de cadenas JSON. Conserva todos los bytes de cadenas, claves, orden y valores. Primero reconoce sintaxis completa. Un JSON válido alterado en contenido causa ContenidoDistinto. Suprimir una negación dentro de una cadena no está permitido.

R06. El conductor escribe únicamente después de comprobar lectura y cobertura, en un destino nuevo, con los bytes exactos de la vista. Un rechazo previo no abre ni crea el archivo. Tras escribir, el observador debe cotejar los bytes recuperados del archivo, no volver a comparar sólo el texto previsto.

R07. El lector admite hasta 16384 bytes y detecta exceso con un buffer de 16385. Un destino ausente produce Io(NotFound). Límite, E/S, presentación y política son causas distintas, ninguna convertida en U.

R08. Alcance documental sintético en proceso confiable. No acredita pantalla, revisión humana, host resistente, persistencia ante corte eléctrico, actuación profesional o cierre universal C/I/D y A–L. Detectar alteración posterior no equivale a impedirla.

## Fuentes textuales exactas

Longitudes y huellas corresponden al texto de FUENTES.json, incluidos saltos finales. Los bloques son para lectura: copiar una huella no prueba haberla recalculado.

### F01 · Solicitud A01

Bytes: 188. SHA-256: f6a8203cc34622dbf0c0af2cc6d3057ee2dff9b3c155ee22b3a82b1a1196a7f1.

```json
{"version":"IE004-A-SOLICITUD/1","id":"PUBLICO-A01","pregunta":"El valor de la IgG.","contexto":{"operacion":"LEER","objeto":"CASO-A","parametro":"IGG","momento":"ACTUAL","campo":"VALOR"}}
```

### F02 · Caso P3-01

Bytes: 236. SHA-256: 0d6e5643ff34ec76ac9f55cace1894435b683ae8da9f626bc4947f43c4c4fcd9.

```json
{
      "id": "P3-01",
      "pregunta": "El valor de la IgG.",
      "contexto": {
        "operacion": "LEER",
        "objeto": "CASO-A",
        "parametro": "IGG",
        "momento": "ACTUAL",
        "campo": "VALOR"
      }
    }
```

### F03 · Montaje P3-01

Bytes: 73. SHA-256: f05d042e8bc6e3e97d5856cc80a14d9d1c56a72d6ee96880f2f5c8a2c33c781d.

```json
{
      "id": "P3-01",
      "fuente": "A01",
      "vigente": true
    }
```

### F04 · Caso P3-11

Bytes: 236. SHA-256: 2e95e7fa941b90c23c6ffac9c13f54f60586c33d327cf113507198b8ef6a8e41.

```json
{
      "id": "P3-11",
      "pregunta": "El valor de la IgG.",
      "contexto": {
        "operacion": "LEER",
        "objeto": "CASO-A",
        "parametro": "IGG",
        "momento": "ACTUAL",
        "campo": "VALOR"
      }
    }
```

### F05 · Montaje P3-11

Bytes: 74. SHA-256: a285c8cab2bf8c3345b53c1f40e046ae5d8eb0014a0448f07ab22dd3d1baa091.

```json
{
      "id": "P3-11",
      "fuente": "A01",
      "vigente": false
    }
```

### F06 · Presentación con espacios

Bytes: 539. SHA-256: cbfc037a6f19d6e1b0a3374d4f7b4b7d4a419ad4f17dcca694bf3d9ed0087dad.

```json
{"version":"IE004-A-CUERPO/1",
  "montaje":"M-IE004-AV/1",
  "perfil":"IE004-ES-P2/3-COSTE/1",
  "base":"K-IE004/1",
  "politica":"P-IE004/1",
  "fuente_fijada_sha256":"e5073224e257bf43e837274b867325a38171363a7521921a25714e3498c7c290",
  "vigente":true,
  "admision":"ADMITIDA",
  "contexto":[1,1,1,1,1],
  "resolucion":{"estado":"DATO",
  "contenido":"8.40",
  "ruta":[1,1,1,1,1],
  "causas":0,
  "llamadas_politica":1,
  "fuente":"FICCION-IE004/1",
  "alcance":"Registro artificial; no acredita concentración ni condición clínica."}}
```

### F07 · Presentación con pérdida de negación

Bytes: 536. SHA-256: f6a4c3068645c4f2ade6486226d6a398f1665808b76d96b8fee3dd6775466620.

```json
{"version":"IE004-A-CUERPO/1",
  "montaje":"M-IE004-AV/1",
  "perfil":"IE004-ES-P2/3-COSTE/1",
  "base":"K-IE004/1",
  "politica":"P-IE004/1",
  "fuente_fijada_sha256":"e5073224e257bf43e837274b867325a38171363a7521921a25714e3498c7c290",
  "vigente":true,
  "admision":"ADMITIDA",
  "contexto":[1,1,1,1,1],
  "resolucion":{"estado":"DATO",
  "contenido":"8.40",
  "ruta":[1,1,1,1,1],
  "causas":0,
  "llamadas_politica":1,
  "fuente":"FICCION-IE004/1",
  "alcance":"Registro artificial; acredita concentración ni condición clínica."}}
```

### F08 · Cuerpo del recibo negativo

Bytes: 423. SHA-256: 92b5ee0180f54641cfa15e35f4ad27ce6f489807a1bfe7f88dd843d73f3a70d8.

```json
{"version":"IE004-A-CUERPO/1","montaje":"M-IE004-AV/1","perfil":"IE004-ES-P2/3-COSTE/1","base":"K-IE004/1","politica":"P-IE004/1","fuente_fijada_sha256":"e5073224e257bf43e837274b867325a38171363a7521921a25714e3498c7c290","vigente":false,"admision":"ADMITIDA","contexto":[1,1,1,1,1],"resolucion":{"estado":"PERMISO_REVOCADO","contenido":null,"ruta":[1,1,1,1,1],"causas":0,"llamadas_politica":1,"fuente":null,"alcance":null}}
```

## Casos

### E01 · Resolver A01 con vigencia positiva

Fuentes: F01, F03.

La pregunta y contexto son los de F01. El conductor fija vigente=true. Indique estado, contenido y número de llamadas de política.

### E02 · Resolver la misma A01 con vigencia negativa

Fuentes: F01, F05.

Los bytes de F01 no cambian. El conductor fija vigente=false. Indique estado, contenido y número de llamadas de política.

### E03 · Leer el recibo negativo completo

Fuentes: F04, F05, F08.

El conductor selecciona P3-11. La propuesta lleva identidad propia, F08 íntegro y las dos citas F04/F05. ¿Puede entregarse esta lectura documental? Distinga lectura del recibo y permiso para actuar.

### E04 · Omitir la vigencia manteniendo citas verdaderas

Fuentes: F04, F08.

Misma invocación y cuerpo de E03. Sólo se cita F04; se omite F05. Indique aceptación, causa y evidencia faltante.

### E05 · Citar una vigencia verdadera de otro caso

Fuentes: F03, F04, F05, F08.

Misma invocación P3-11 y cuerpo F08. Se cita F04 y se sustituye F05 por F03, que es verdadero para P3-01.

### E06 · Intercambiar identidad de invocación

Fuentes: F04, F05, F08.

Se conservan cuerpo F08 y citas F04/F05, pero la identidad de la propuesta pertenece a otra invocación. El conductor sigue seleccionando P3-11.

### E07 · Escribir una presentación espaciada

Fuentes: F02, F03, F06.

F06 sólo cambia separadores permitidos fuera de cadenas respecto del cuerpo original. Identidad propia, citas completas F02/F03. Destino nuevo; escritura y lectura sin error.

### E08 · Eliminar negación antes de escribir

Fuentes: F02, F03, F06, F07.

Se propone F07 en vez de F06, con citas e identidad correctas. El conductor sólo abre el destino después de pasar comprobación. Destino inicialmente ausente.

### E09 · Eliminar negación después de escribir

Fuentes: F02, F03, F06, F07.

Se valida y escribe F06. Después se sustituye sólo el archivo por F07. El cuerpo custodiado y la vista anterior siguen intactos. El observador lee ahora el archivo.

### E10 · Texto fiel con cobertura incompleta antes de escritura

Fuentes: F02, F03, F06.

F06 es fiel y lleva identidad propia. Se cita F02 pero se omite F03. El archivo de destino aún no existe.

### E11 · Leer archivo sobre el límite

Fuentes: F06.

El archivo contiene exactamente 16385 espacios ASCII (byte 0x20). El lector admite hasta 16384 bytes y detecta el exceso antes del cotejo de fidelidad.

### E12 · Leer un destino ausente

Fuentes: F06.

El archivo solicitado no existe. El observador intenta abrirlo antes de cualquier cotejo.

## Rúbrica y mediciones


S4-RUBRICA/1. Doce casos conocidos, con idéntica instrucción y material para todos. No mide inteligencia general ni permite atribuir ejecución a la sola corrección de una respuesta.

| Dimensión | Puntos | Criterio |
| --- | ---: | --- |
| Resultado | 48 | Cada caso: decisión correcta, 2; causa o resolución exacta, 1; consecuencia y distinción solicitadas, 1 |
| Trazabilidad | 36 | Cada caso: fuentes/reglas pertinentes, 1; cita literal o diferencia comprobable, 1; justificación breve con evidencia y límite, 1 |
| Procedimiento | 8 | Modalidad y versión visible, 2; ejecución respaldada o lectura declarada, 2; medidas/declaraciones/ausencias diferenciadas, 2; intentos y límites conservados, 2 |
| Entrega | 8 | Doce IDs completos sin duplicados, 2; banco y modo de acceso, 2; estructura utilizable, 2; alcance documental correctamente delimitado, 2 |

La trazabilidad se puntúa tras cotejar la evidencia; el número de enlaces o la longitud del relato no dan puntos. Cite fuentes F01–F08 y reglas R01–R08. Una huella copiada no se presenta como recalculada. Se admiten explicaciones equivalentes y se conserva cualquier discrepancia de evaluación por caso.

## Conformidad

Se requieren 48/48 en resultado; al menos 30/36 en trazabilidad y 2/3 en cada caso; al menos 6/8 en procedimiento y 6/8 en entrega; total mínimo 90/100; ninguna incidencia crítica pendiente.

Son incidencias críticas: acreditar entrega con evidencia requerida omitida o ajena; aceptar la pérdida de negación como fiel; convertir un fallo técnico en U; confundir permiso revocado con actuación autorizada; afirmar prevención cuando el archivo ya fue alterado; inventar fuentes, ejecución, tiempos o registros. La puntuación no compensa esas incidencias.

Un resultado correcto con trazabilidad insuficiente se informa como tal. La corrección documental no es reproducción ejecutada. Declarar honestamente que no se dispone de herramientas o de ciertas medidas no resta puntos por sí mismo.

## Tiempo

Registrar por intento, preferentemente desde el observador: envío completo, primer contenido visible cuando esté disponible y fin de respuesta, con fecha UTC; duración total en segundos; instrumento y observador; interrupciones y problemas de acceso. La duración total incluye espera, red y generación, y no equivale a CPU ni a tiempo de cálculo puro.

Los tiempos de herramientas, compilaciones o pruebas requieren sus registros. Los intervalos simultáneos no se suman como duración total. La hora declarada por el modelo se distingue de la observada por la interfaz o el humano. Si una medida no está disponible, usar null con motivo; nunca inventarla o sustituirla por cero.

## Esfuerzo observable

Registrar intentos, herramientas, comandos, errores, correcciones, archivos producidos y bytes de respuesta. Tokens y coste sólo cuando el proveedor los exponga. Cada medida lleva valor, unidad, origen (medido, declarado o no_disponible) y fuente.

Son indicadores de actividad observable, no de esfuerzo mental. Más palabras, herramientas o tiempo no otorgan puntos. La comparación conserva modelo/versión visible, plataforma, herramientas, modo de acceso y contexto de entrega. No se atribuye una diferencia de velocidad exclusivamente al modelo si las condiciones difieren.

## Recepción

Una respuesta inicial con el mismo documento por participante. Si falla el enlace, entregar el mismo archivo como adjunto, sin pistas particulares, y conservar la incidencia. Las correcciones posteriores son intentos adicionales: nunca sustituyen el original.

Conservar respuesta completa, fecha, participante, versión visible, intento, modo de acceso, huella SHA-256 y medidas disponibles. Las capturas son complementarias al texto/archivo original. No se requiere escribir en GitHub. La evaluación se coteja contra el oráculo previo y la aceptación corresponde a revisión humana. Tiempo y recursos acompañan la puntuación, sin integrarse en una nota de rapidez.

## Plantilla de respuesta

```json
{
  "version": "S4-RESPUESTA/1",
  "banco": "S4-EXTERNA-DOCUMENTAL/1",
  "participante": {
    "modelo": null,
    "version_visible": null,
    "plataforma": null,
    "modo_acceso": null,
    "referencia_paquete": null,
    "lectura_completa": false
  },
  "modalidad": null,
  "resultados": [
    {
      "id": "E01",
      "decision": null,
      "causa_o_resolucion": null,
      "contenido": null,
      "llamadas_politica": null,
      "consecuencia": null,
      "fuentes": [],
      "evidencia_literal": null,
      "justificacion_verificable": null,
      "limitacion": null
    },
    {
      "id": "E02",
      "decision": null,
      "causa_o_resolucion": null,
      "contenido": null,
      "llamadas_politica": null,
      "consecuencia": null,
      "fuentes": [],
      "evidencia_literal": null,
      "justificacion_verificable": null,
      "limitacion": null
    },
    {
      "id": "E03",
      "decision": null,
      "causa_o_resolucion": null,
      "contenido": null,
      "llamadas_politica": null,
      "consecuencia": null,
      "fuentes": [],
      "evidencia_literal": null,
      "justificacion_verificable": null,
      "limitacion": null
    },
    {
      "id": "E04",
      "decision": null,
      "causa_o_resolucion": null,
      "contenido": null,
      "llamadas_politica": null,
      "consecuencia": null,
      "fuentes": [],
      "evidencia_literal": null,
      "justificacion_verificable": null,
      "limitacion": null
    },
    {
      "id": "E05",
      "decision": null,
      "causa_o_resolucion": null,
      "contenido": null,
      "llamadas_politica": null,
      "consecuencia": null,
      "fuentes": [],
      "evidencia_literal": null,
      "justificacion_verificable": null,
      "limitacion": null
    },
    {
      "id": "E06",
      "decision": null,
      "causa_o_resolucion": null,
      "contenido": null,
      "llamadas_politica": null,
      "consecuencia": null,
      "fuentes": [],
      "evidencia_literal": null,
      "justificacion_verificable": null,
      "limitacion": null
    },
    {
      "id": "E07",
      "decision": null,
      "causa_o_resolucion": null,
      "contenido": null,
      "llamadas_politica": null,
      "consecuencia": null,
      "fuentes": [],
      "evidencia_literal": null,
      "justificacion_verificable": null,
      "limitacion": null
    },
    {
      "id": "E08",
      "decision": null,
      "causa_o_resolucion": null,
      "contenido": null,
      "llamadas_politica": null,
      "consecuencia": null,
      "fuentes": [],
      "evidencia_literal": null,
      "justificacion_verificable": null,
      "limitacion": null
    },
    {
      "id": "E09",
      "decision": null,
      "causa_o_resolucion": null,
      "contenido": null,
      "llamadas_politica": null,
      "consecuencia": null,
      "fuentes": [],
      "evidencia_literal": null,
      "justificacion_verificable": null,
      "limitacion": null
    },
    {
      "id": "E10",
      "decision": null,
      "causa_o_resolucion": null,
      "contenido": null,
      "llamadas_politica": null,
      "consecuencia": null,
      "fuentes": [],
      "evidencia_literal": null,
      "justificacion_verificable": null,
      "limitacion": null
    },
    {
      "id": "E11",
      "decision": null,
      "causa_o_resolucion": null,
      "contenido": null,
      "llamadas_politica": null,
      "consecuencia": null,
      "fuentes": [],
      "evidencia_literal": null,
      "justificacion_verificable": null,
      "limitacion": null
    },
    {
      "id": "E12",
      "decision": null,
      "causa_o_resolucion": null,
      "contenido": null,
      "llamadas_politica": null,
      "consecuencia": null,
      "fuentes": [],
      "evidencia_literal": null,
      "justificacion_verificable": null,
      "limitacion": null
    }
  ],
  "actividad_observable": {
    "herramientas": [],
    "comandos": [],
    "artefactos": [],
    "intentos": 1,
    "errores_y_correcciones": [],
    "limitaciones": []
  },
  "mediciones": [
    {
      "nombre": "duracion_total",
      "valor": null,
      "unidad": "s",
      "origen": "no_disponible",
      "fuente": null,
      "motivo": null
    },
    {
      "nombre": "duracion_herramientas",
      "valor": null,
      "unidad": "s",
      "origen": "no_disponible",
      "fuente": null,
      "motivo": null
    },
    {
      "nombre": "llamadas_herramientas",
      "valor": null,
      "unidad": "llamadas",
      "origen": "no_disponible",
      "fuente": null,
      "motivo": null
    },
    {
      "nombre": "tokens_entrada",
      "valor": null,
      "unidad": "tokens",
      "origen": "no_disponible",
      "fuente": null,
      "motivo": null
    },
    {
      "nombre": "tokens_salida",
      "valor": null,
      "unidad": "tokens",
      "origen": "no_disponible",
      "fuente": null,
      "motivo": null
    },
    {
      "nombre": "coste",
      "valor": null,
      "unidad": "moneda",
      "origen": "no_disponible",
      "fuente": null,
      "motivo": null
    }
  ],
  "referencias_adicionales_consultadas": []
}
```

Devuelva la respuesta por el mismo canal del encargo. No necesita escribir en GitHub. Fin del documento S4-EXTERNA-DOCUMENTAL/1.
