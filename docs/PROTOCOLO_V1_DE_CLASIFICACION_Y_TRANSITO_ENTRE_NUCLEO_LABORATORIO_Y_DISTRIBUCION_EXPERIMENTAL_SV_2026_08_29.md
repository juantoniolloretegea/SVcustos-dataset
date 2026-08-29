# Protocolo V1 de clasificación y tránsito entre núcleo, laboratorio y distribución experimental del Lenguaje SV

**Fecha:** 29 de agosto de 2026  
**Versión:** 1  
**Estado:** vigente para trabajos posteriores a Beta B2; documento evolutivo  
**Ámbito:** `SV-lenguaje-de-computacion`, `SV-matematica-semantica-cuaternaria` y `SVcustos-dataset`

## 1. Objeto

Este protocolo fija la clasificación mínima y la secuencia de tránsito de los cambios relacionados con el Lenguaje de Computación SV entre tres sedes con funciones distintas:

1. el repositorio soberano del Lenguaje, donde se constituyen y evolucionan las propiedades básicas y estructurales;
2. el laboratorio, donde pueden ensayarse y verificarse capacidades laterales expresamente necesarias sin convertir el laboratorio en una segunda realización soberana del Lenguaje;
3. la distribución experimental pública, donde pueden exponerse capacidades laterales ya verificadas sin adquirir por ello autoridad semántica ni estatuto de producción.

Su finalidad es impedir que una necesidad estructural del Lenguaje sea resuelta lateralmente en el laboratorio o en una interfaz pública, y asegurar que toda capacidad experimental conserve procedencia, alcance, evidencia y límites explícitos.

Este protocolo no sustituye los contratos técnicos, las condiciones de cierre de fase, los controles de seguridad ni los registros de calidad aplicables a cada cambio.

## 2. Función de cada sede

### 2.1. Repositorio soberano del Lenguaje

`SV-lenguaje-de-computacion` es la sede de origen de toda modificación que afecte a la identidad, sintaxis, representación, semántica, validación, autoridad o garantías del Lenguaje.

Una necesidad pertenece al núcleo cuando modifica o puede modificar, entre otras materias:

- gramática, léxico o análisis sintáctico;
- identidades canónicas y perfiles lingüísticos cuando alteren reconocimiento o canonicalización;
- IR y reglas de bienformación;
- validación semántica o ejecución;
- identidad o contrato de diagnósticos;
- autoridad, mediación, persistencia o invariantes de seguridad;
- ABI semántica del núcleo;
- semántica del ensamblaje, módulos, dependencias o punto de entrada;
- significado de huellas, procedencia o identidad de fuente cuando formen parte del contrato del Lenguaje.

Estas necesidades deberán crecer en una rama propia de este repositorio. El laboratorio no se utilizará como lugar de implementación sustitutiva de una modificación estructural todavía no constituida en el Lenguaje.

### 2.2. Laboratorio

`SV-matematica-semantica-cuaternaria` actúa, para este ámbito, como sede de experimentación, medición, contraste y evidencia de capacidades laterales o de verificaciones independientes.

Puede recibir una capacidad cuando concurran conjuntamente estas condiciones:

1. existe una necesidad técnica actual y explícita;
2. la frontera respecto del núcleo está delimitada;
3. la capacidad puede realizarse sin crear una gramática, analizador, validador, semántica o autoridad paralelos;
4. el corte exacto del Lenguaje del que depende está identificado;
5. existen criterios de aceptación y límites verificables.

El laboratorio puede detectar defectos del núcleo. No debe corregirlos lateralmente como si fueran propiedades del experimento.

### 2.3. Distribución experimental pública

`SVcustos-dataset` puede alojar ramas de distribución pública experimental vinculadas al Lenguaje, pero no constituye una sede de definición del Lenguaje.

Su función se limita a exponer o transportar capacidades laterales previamente verificadas, por ejemplo:

- un entorno de edición o futuro IDE que consuma interfaces canónicas sin reimplementar el compilador;
- una interfaz de demostración;
- documentación de uso o navegación;
- una futura Wiki que actúe como manual;
- mecanismos de distribución de un artefacto cuya identidad se verifique antes de su ejecución;
- presentación bilingüe o localización de interfaz que no modifique el código fuente ni su perfil.

Una modificación en `SVcustos-dataset` no puede subsanar por sí misma una carencia de gramática, IR, semántica, validación o garantías del núcleo.

## 3. Regla de clasificación previa

Antes de abrir trabajo material deberá clasificarse la necesidad.

| Clase | Ejemplos | Sede de origen | Laboratorio | `SVcustos-dataset` |
|---|---|---|---|---|
| Básica o estructural | gramática, IR, validador, semántica, diagnósticos, autoridad, persistencia, ABI nuclear, semántica multifuente | rama de `SV-lenguaje-de-computacion` | sólo contraste o verificación posterior | sólo distribución posterior; nunca origen ni reparación |
| Lateral necesaria | IDE, editor, visualización, transporte verificado de artefactos, interfaz de demostración | laboratorio, una vez fijada la dependencia del núcleo | sí | sí, tras luz verde del laboratorio |
| Documental de uso | manual, Wiki, navegación, ayuda de usuario | según el estado de la capacidad documentada | puede prepararse | puede publicarse si distingue con claridad lo estable de lo experimental |
| Propuesta futura | capacidad no requerida todavía | registro de propuestas | sin implementación salvo autorización expresa | no se despliega como capacidad disponible |

Ante una clasificación dudosa se aplicará la opción conservadora: **si una modificación puede alterar el significado o la aceptación de un programa, deberá tratarse como estructural y comenzar en el repositorio soberano del Lenguaje**.

## 4. Ruta A — necesidad básica o estructural

Una necesidad estructural seguirá, como mínimo, esta secuencia:

```text
necesidad material identificada
        ↓
corte exacto de main fijado
        ↓
rama específica en SV-lenguaje-de-computacion
        ↓
objeto + contrato + criterios de aceptación + garantías afectadas
        ↓
implementación y pruebas
        ↓
conformidad + regresiones de las fases afectadas
        ↓
registro de límites o deuda restante
        ↓
revisión e integración gobernada
```

No se copiará una corrección estructural todavía no integrada a un laboratorio o a `SVcustos-dataset` con el fin de hacer pasar una Beta o una demostración pública.

Cuando una modificación estructural sea necesaria para que una capacidad lateral pueda existir, la capacidad lateral quedará detenida hasta que el núcleo disponga de un candidato definido y verificable en su sede propia.

## 5. Ruta B — capacidad lateral necesaria

Una capacidad lateral podrá abrirse en laboratorio cuando el Lenguaje ya proporcione la base necesaria y la nueva función no modifique esa base.

La secuencia mínima será:

```text
necesidad lateral explícita
        ↓
frontera de no afectación al núcleo
        ↓
corte fuente soberano identificado
        ↓
rama de laboratorio
        ↓
prototipo o realización auxiliar
        ↓
pruebas funcionales + seguridad + rendimiento cuando sean materiales
        ↓
identidad de artefactos y evidencia reproducible
        ↓
luz verde de laboratorio
        ↓
rama experimental pública en SVcustos-dataset
        ↓
comprobación de identidad, límites y comportamiento visible
```

La realización lateral deberá reutilizar las interfaces o artefactos canónicos disponibles. No deberá introducir un segundo analizador sintáctico, un segundo validador ni tablas semánticas paralelas que puedan divergir del núcleo.

La rama pública deberá declarar de manera verificable, cuando resulte aplicable:

- estado experimental;
- identificador de la Beta;
- corte fuente soberano;
- referencia de laboratorio;
- ejecución o evidencia asociada;
- identidad criptográfica y tamaño del artefacto ejecutable;
- versión de gramática, IR y proyección utilizadas;
- límites expresos de la capacidad.

## 6. Tratamiento de incidencias descubiertas fuera del núcleo

Toda anomalía detectada durante una campaña de laboratorio o en una distribución experimental deberá someterse a discriminación causal antes de corregirse.

### 6.1. Prueba de origen

Siempre que sea técnicamente posible se compararán, mediante la misma sonda y condiciones equivalentes:

1. el corte soberano exacto del que parte el experimento;
2. el candidato experimental.

La clasificación será:

```text
falla sólo el candidato experimental
    → regresión atribuible al experimento

falla también el corte soberano de partida
    → hueco o deuda heredada del núcleo

no puede discriminarse con evidencia suficiente
    → estado no resuelto; no procede promoción
```

### 6.2. Regresión experimental

Si el defecto pertenece exclusivamente a la capa lateral, podrá corregirse en la rama experimental correspondiente y deberá repetirse la batería afectada.

Si la corrección exige modificar una propiedad estructural del Lenguaje, cesará la ruta lateral y se abrirá la Ruta A.

### 6.3. Hueco heredado del núcleo

Si el defecto ya existe en el corte soberano:

1. no se reparará lateralmente en el laboratorio ni en `SVcustos-dataset`;
2. se conservará la evidencia que discrimina su origen;
3. se registrará la deuda en la sede de calidad del Lenguaje cuando afecte a una obligación o cierre ya declarado;
4. se abrirá el frente correctivo correspondiente en una rama del repositorio soberano;
5. si afecta a la base de una fase cerrada o posterior, se suspenderá la continuidad que dependa de esa base hasta su recertificación;
6. la sonda que reveló el hueco deberá convertirse en prueba permanente cuando proceda.

Una Beta puede, por tanto, quedar libre de regresiones propias y al mismo tiempo descubrir una deuda heredada que impida una decisión posterior de producción.

### 6.4. Incidencia histórica o documental

Un vector histórico obsoleto, una denominación antigua o una desincronización documental deberán registrarse y reconciliarse en su ámbito propio. No se modificará la semántica para hacer coincidir artificialmente una realización vigente con material histórico ya superado.

## 7. Significado de la luz verde

La expresión «luz verde» deberá acompañarse siempre de su perímetro.

### 7.1. Luz verde de laboratorio

Significa únicamente que la capacidad experimental satisface sus criterios declarados, que las regresiones atribuibles a ella están resueltas y que las deudas preexistentes detectadas han quedado separadas y gobernadas.

### 7.2. Luz verde de distribución experimental

Significa que la interfaz pública corresponde al candidato y artefactos identificados, conserva sus límites y no introduce una semántica propia.

### 7.3. Producción

Ninguna de las dos luces verdes anteriores equivale a integración en producción.

Una decisión de producción requiere un acto separado y deberá comprobar, al menos:

- el estado del núcleo y de toda deuda bloqueante;
- las regresiones exigibles de las fases afectadas;
- la comparación exacta de cambios que se pretende integrar;
- la correspondencia entre fuente, artefacto y distribución;
- la documentación pública y los registros de calidad aplicables;
- la inexistencia de una vía lateral que esté sustituyendo una obligación estructural pendiente;
- una aprobación expresa de integración.

No se trasladará automáticamente código desde `SVcustos-dataset` al núcleo. Si una capacidad experimental revela una necesidad que debe convertirse en propiedad del Lenguaje, esa propiedad deberá constituirse mediante una rama propia de `SV-lenguaje-de-computacion` y superar sus controles correspondientes.

## 8. IDE, Wiki y otras capacidades laterales

Un IDE puede desarrollarse lateralmente mientras permanezca como consumidor del compilador y de sus interfaces canónicas. Si requiere nuevos símbolos, diagnósticos estructurados, módulos, dependencias, semántica de proyecto o cualquier otra propiedad inexistente del Lenguaje, esas propiedades deberán constituirse primero mediante la Ruta A.

Una Wiki o manual de GitHub puede actuar como capa documental de uso, pero no define la gramática ni la semántica. Deberá distinguir las capacidades estables de las experimentales y no documentar como disponible aquello que todavía sea únicamente una propuesta.

El mismo criterio se aplicará a resaltado sintáctico, completado, navegación, visores de IR, trazas y demás herramientas: deberán derivarse de fuentes canónicas o interfaces gobernadas y no mantener vocabularios o reglas paralelos.

## 9. Lista de comprobación antes de iniciar trabajo

Antes de abrir una rama de laboratorio o una modificación en `SVcustos-dataset` deberán poder responderse afirmativamente las preguntas que correspondan:

1. ¿La necesidad es actual y explícita, y no sólo una posibilidad futura?
2. ¿Se ha comprobado primero si afecta al núcleo del Lenguaje?
3. Si afecta al núcleo, ¿existe ya una rama específica en `SV-lenguaje-de-computacion` y se ha detenido la vía lateral?
4. Si es lateral, ¿está declarada la frontera que impide modificar semántica, gramática, IR o validación?
5. ¿Está fijado el corte soberano exacto del que depende?
6. ¿Existen criterios de aceptación y límites verificables?
7. Antes de publicar en `SVcustos-dataset`, ¿el laboratorio ha alcanzado luz verde en ese perímetro?
8. ¿La distribución pública identifica de forma verificable su fuente, artefactos y estado experimental?
9. Si apareció una anomalía, ¿se discriminó primero si era regresión o deuda heredada?
10. ¿Se ha evitado convertir una luz verde experimental en autorización implícita de producción?

Una respuesta negativa en los puntos estructurales obliga a detener el tránsito hasta resolverla.

## 10. Trazabilidad de las copias

La copia de control de este protocolo se conserva en el laboratorio y puede espejarse en la rama experimental pública correspondiente de `SVcustos-dataset`.

Las copias de una misma versión deberán ser textualmente idénticas. La identidad de contenido podrá verificarse mediante la huella del objeto Git o una huella criptográfica equivalente.

La existencia de esta copia de control no otorga al laboratorio autoridad sobre la semántica del Lenguaje; únicamente fija el procedimiento de tránsito entre sedes.

## 11. Evolución del protocolo

La versión 1 se emite después de la experiencia de Beta B2, en la que una campaña experimental técnicamente satisfactoria reveló además deuda preexistente del núcleo. Ese hecho demuestra la necesidad de separar de manera explícita:

```text
corrección del experimento
≠ corrección del núcleo
≠ autorización de producción
```

El protocolo podrá evolucionar cuando aparezcan necesidades materiales nuevas. Una versión posterior deberá:

1. identificarse con un número de versión superior;
2. conservar la versión anterior para trazabilidad;
3. declarar qué reglas modifica o amplía;
4. volver a sincronizar sus copias de control y espejo.

Hasta que exista una versión sucesora expresamente constituida, **V1 permanece como procedimiento vigente de clasificación y tránsito**.
