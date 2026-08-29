# Protocolo V2.1 de clasificación, tránsito, validación, promoción y despliegue del Lenguaje SV

**Fecha:** 29 de agosto de 2026  
**Versión:** 2.1  
**Estado:** vigente; sustituye a V2 como procedimiento operativo de referencia  
**Predecesor:** `PROTOCOLO_V2_DE_CLASIFICACION_TRANSITO_VALIDACION_Y_PROMOCION_DEL_LENGUAJE_SV_2026_08_29.md`  
**Ámbito:** `SV-lenguaje-de-computacion`, `SV-matematica-semantica-cuaternaria`, `SVcustos-dataset` y la capa autorizada de publicación web del Lenguaje SV

## 1. Objeto

Este protocolo fija la clasificación y el tránsito de los cambios relacionados con el Lenguaje de Computación SV desde su sede soberana, a través de los trabajos experimentales o laterales que resulten necesarios, hasta su eventual publicación estable.

V2.1 conserva las reglas de V2 y añade las condiciones de cierre de la capa material de despliegue. En particular, establece que una integración correcta en el repositorio soberano no equivale por sí sola a una publicación efectiva, que el paquete estático realmente desplegado debe quedar identificado y conservado, y que los hipervínculos destinados a lectores deben comprobarse por el resultado que presentan y no sólo por la existencia del archivo de destino.

El propósito es mantener una separación inequívoca entre:

1. definición y realización soberanas del Lenguaje;
2. laboratorio privado;
3. distribución experimental pública;
4. candidata de integración;
5. versión estable integrada;
6. paquete material desplegado;
7. comprobación posterior a publicación;
8. cierre de Calidad.

## 2. Función de cada sede

### 2.1. Repositorio soberano

`SV-lenguaje-de-computacion` es la sede de definición, realización, integración, Calidad e historial estable del Lenguaje.

Toda modificación que pueda alterar el significado, reconocimiento, aceptación, rechazo, identidad canónica o garantías de un programa debe originarse y cerrarse en una rama de este repositorio.

Pertenecen a este ámbito, entre otras materias:

- gramática, léxico y análisis sintáctico;
- perfiles fuente y canonicalización;
- representación intermedia y reglas de bienformación;
- validación y semántica;
- contrato de diagnósticos;
- autoridad, mediación y persistencia;
- ABI del núcleo;
- ensamblaje multifuente, módulos y dependencias;
- invariantes de seguridad;
- artefactos ejecutables publicados como realización estable.

### 2.2. Laboratorio privado

`SV-matematica-semantica-cuaternaria` es la sede privada de experimentación, medición, contraste, regresión, evidencia, reconstrucción y conservación registral.

El laboratorio no constituye una realización soberana paralela y no puede reparar lateralmente una carencia estructural del núcleo.

### 2.3. Distribución experimental pública

`SVcustos-dataset` puede alojar una Beta o cara experimental pública cuando resulte necesario exponer capacidades laterales o de entorno sin convertirlas en definición soberana del Lenguaje.

La distribución experimental:

- no define gramática, IR, semántica ni autoridad;
- debe identificar su procedencia y estado;
- no convierte una propuesta futura en capacidad estable;
- no se utiliza para corregir por atajo una deuda del núcleo;
- conserva sólo el material necesario para la distribución y revisión pública de la Beta.

### 2.4. Capa de publicación web

La aplicación o servicio que presenta el Lenguaje al público es una capa material de despliegue. Puede estar técnicamente separada del repositorio soberano.

Su función es servir una realización ya autorizada. No adquiere autoridad para definir el Lenguaje ni para introducir correcciones independientes.

## 3. Laboratorio cerrado por defecto

### 3.1. Autorización de apertura

El laboratorio permanece cerrado por defecto a trabajo material.

Ninguna actuación operativa puede abrir por iniciativa propia una campaña, rama experimental, ejecución, modificación o reactivación del laboratorio. Toda apertura material exige autorización expresa de la Dirección para el trabajo concreto.

Se considera apertura material, entre otras acciones:

- crear una rama experimental o de campaña;
- crear o modificar un flujo de ejecución;
- lanzar pruebas, mediciones o campañas experimentales;
- modificar una realización experimental;
- generar o incorporar nuevos artefactos de laboratorio;
- reactivar un frente cerrado, suspendido o hibernado;
- trasladar una capacidad al laboratorio o desde éste a una distribución experimental;
- ampliar el alcance de una campaña ya autorizada.

Una autorización no se extiende automáticamente a trabajos distintos ni a fases posteriores.

### 3.2. Lectura sin apertura

La consulta de evidencia ya existente en modo de solo lectura no constituye apertura material cuando no modifica archivos, ramas, artefactos ni ejecuciones.

Si esa consulta revela la necesidad de ejecutar, modificar o ampliar trabajo experimental, la actuación se detendrá hasta obtener autorización expresa.

### 3.3. Cierre de campaña

Toda campaña autorizada terminará con un estado explícito:

- cerrada satisfactoriamente;
- cerrada con deuda o límites registrados;
- suspendida;
- hibernada;
- no resuelta.

El cierre de una campaña no abre la siguiente.

## 4. Clasificación previa de cualquier necesidad

Antes de iniciar trabajo material debe clasificarse la necesidad.

| Clase | Ejemplos | Sede de origen |
|---|---|---|
| Básica o estructural | gramática, IR, validador, semántica, autoridad, ABI nuclear, reglas multifuente | rama de `SV-lenguaje-de-computacion` |
| Lateral necesaria | editor, visualización, interfaz, transporte, herramienta auxiliar | laboratorio autorizado cuando la frontera con el núcleo esté cerrada |
| Documental de uso | manual, navegación, ayuda | sede correspondiente al estado real de la capacidad |
| Propuesta futura | capacidad todavía no requerida | registro de propuestas; no se implementa como disponible |

Ante duda se aplica la regla conservadora:

> Si una modificación puede cambiar el significado, aceptación, rechazo o identidad de un programa, se trata como estructural y comienza en `SV-lenguaje-de-computacion`.

## 5. Ruta de una necesidad estructural

La secuencia mínima es:

```text
necesidad material identificada
        ↓
corte exacto de main
        ↓
rama específica en SV-lenguaje-de-computacion
        ↓
contrato y criterios de aceptación
        ↓
implementación
        ↓
pruebas unitarias e integración
        ↓
conformidad y regresiones de fases afectadas
        ↓
seguridad y rendimiento cuando sean materiales
        ↓
registro de límites o deuda
        ↓
revisión
        ↓
integración gobernada
```

Una corrección estructural pendiente no se copiará a una Beta para hacer pasar artificialmente una distribución experimental.

## 6. Ruta de una necesidad lateral

Una capacidad lateral puede entrar en laboratorio sólo cuando:

1. exista autorización expresa;
2. la necesidad esté delimitada;
3. el núcleo ya proporcione la base necesaria;
4. se haya fijado la frontera de no afectación al núcleo.

Secuencia:

```text
autorización
        ↓
necesidad lateral concreta
        ↓
frontera con el núcleo
        ↓
corte soberano identificado
        ↓
trabajo de laboratorio
        ↓
pruebas funcionales
        ↓
seguridad/rendimiento si procede
        ↓
evidencia y huellas
        ↓
cierre de laboratorio
        ↓
autorización de distribución experimental
        ↓
Beta pública en SVcustos-dataset
```

Las herramientas laterales reutilizarán las interfaces canónicas y no mantendrán gramáticas, analizadores, validadores o tablas semánticas paralelos con autoridad propia.

## 7. Campaña previa a promoción desde Beta

Una Beta no se promueve por funcionamiento visual ni por haber cerrado únicamente sus pruebas específicas.

Cuando resulte aplicable, la revisión previa cubrirá:

1. rendimiento y coste material;
2. seguridad estructural;
3. arquitectura y construcción Rust;
4. equivalencia diferencial sobre corpus comprometido;
5. línea basal y escalas;
6. paridad nativa, WASI y navegador;
7. garantías históricas dependientes;
8. deuda histórica y coherencia entre documentación, corpus y realizaciones.

Las cifras, corpus, versiones y huellas se fijarán en el corte revisado. Un comprobador obsoleto no se corregirá alterando el candidato para acomodarlo.

## 8. Discriminación de incidencias

Todo fallo detectado durante una campaña debe clasificarse antes de modificar el producto.

### 8.1. Regresión del candidato

Existe cuando el corte soberano de referencia satisface una obligación y el candidato deja de satisfacerla.

### 8.2. Deuda heredada

Existe cuando el mismo defecto se reproduce en el corte soberano anterior y en el candidato.

Una deuda heredada:

- no se atribuye a la Beta que la descubre;
- no se repara lateralmente dentro de la Beta;
- se registra en Calidad si afecta a una obligación soberana;
- puede bloquear la continuidad de fases dependientes.

### 8.3. Incidencia de infraestructura de prueba

Existe cuando el fallo procede del comprobador, arnés, ruta de archivos, recuento esperado, mecanismo de transporte o entorno de construcción y el producto no ha cambiado materialmente.

La corrección de la infraestructura debe quedar separada de cualquier corrección del producto.

## 9. Construcción de una candidata soberana

Cuando la Beta queda verde en su perímetro y existe autorización para preparar la integración:

1. se parte del `main` soberano vigente;
2. se materializa la realización verificada en una rama de integración;
3. se conserva por separado la procedencia Beta;
4. no se mezcla el historial Beta con el historial estable;
5. se retira del producto estable el andamiaje exclusivamente experimental;
6. las propuestas futuras permanecen como propuestas;
7. los transformadores privados o herramientas auxiliares no se publican como parte del producto salvo necesidad técnica propia;
8. se compara el radio de cambios contra `main`.

## 10. Higiene de la rama candidata

Antes de revisión deben excluirse:

- directorios de construcción;
- archivos temporales;
- fragmentos de transferencia;
- flujos auxiliares ya consumidos;
- cambios incidentales de bloqueos de dependencias sin causa material;
- artefactos no destinados a preservación;
- residuos de estados anteriores.

Las construcciones de validación usarán, cuando sea posible, rutas temporales fuera del árbol versionado.

## 11. Identidad y transporte de artefactos

Los transportes de artefactos deben comprobar su identidad.

Para un artefacto relevante se conservará, según corresponda:

- SHA-256;
- tamaño;
- corte fuente;
- versión de la herramienta;
- identidad de WebAssembly;
- lista o manifiesto de fragmentos;
- prueba de reconstrucción.

Si un transporte altera datos, no se aceptará una reparación heurística. Sólo podrá admitirse una reconstrucción que restituya una identidad criptográfica previamente conocida y comprobable.

## 12. Revisión integral de la candidata

Antes de autorizar producción se repetirá sobre el `head` exacto de la candidata:

- compilación y pruebas del núcleo;
- conformidad;
- regresiones comprometidas;
- equivalencia del modo heredado cuando proceda;
- perfiles lingüísticos;
- ensamblaje;
- identidad del WebAssembly;
- navegación y documentación;
- ausencia de residuos experimentales;
- limpieza del árbol.

Las pruebas que hayan corrido sobre un corte anterior no sustituyen las del corte final.

## 13. Revisión funcional y visual

La revisión previa a publicación comprobará, como mínimo:

- carga inicial;
- cambio de idioma de interfaz;
- selección explícita del perfil fuente;
- compilación válida en los perfiles soportados;
- rechazo por perfil incorrecto;
- ausencia de transformación automática del código modificado;
- ensamblaje multifuente;
- descarga de fuentes cuando esté disponible;
- disposición en escritorio y anchuras reducidas;
- contraste y diferenciación visual de perfiles;
- navegación documental;
- ausencia de textos residuales de Beta o candidata en la versión estable.

## 14. Hipervínculos: existencia y resultado presentado

La comprobación de un hipervínculo no termina al confirmar que el archivo de destino existe.

Debe verificarse que el usuario recibe la representación prevista.

En particular:

- un documento HTML destinado a lectura renderizada no debe enlazarse mediante una vista de repositorio que muestre su código fuente;
- las rutas relativas deben resolverse desde la ubicación real de despliegue;
- los enlaces a lectores documentales deben abrir el documento solicitado;
- los enlaces externos deberán conservar el contexto y destino esperados;
- tras una publicación manual se repetirán las comprobaciones sobre el dominio efectivo.

Si una URL técnicamente existente presenta al usuario una representación distinta de la prevista, se considera defecto de navegación y se corrige antes del cierre de Calidad.

## 15. Propuesta de integración y autorización de producción

La integración se presenta mediante una propuesta revisable sobre el `head` exacto.

Antes de fusionar:

1. la comparación de cambios debe corresponder al alcance previsto;
2. las comprobaciones obligatorias deben estar verdes;
3. no deben existir conflictos materiales;
4. el estado visual y funcional debe haber sido revisado;
5. la autorización de producción debe ser expresa.

La luz verde técnica no sustituye la autorización de producción.

## 16. Fusión e identidad estable

La fusión fija una identidad soberana estable, pero no demuestra por sí sola que el servicio público esté sirviendo esa identidad.

Tras fusionar se comprobarán:

- nuevo `main`;
- relación entre el corte certificado y el corte integrado;
- ejecución de los controles posteriores;
- metadatos de producción;
- historial y registros que deban actualizarse.

## 17. Despliegue material

### 17.1. Regla general

Cuando la capa pública se actualice por un mecanismo distinto de la fusión del repositorio, la promoción no se considera materialmente cerrada hasta verificar ese despliegue.

### 17.2. Paquete estático

Si el despliegue utiliza un paquete estático cargado manualmente:

1. se genera desde el estado soberano autorizado;
2. `index.html` y los recursos requeridos quedan en la estructura esperada por el servicio;
3. se incluyen los artefactos ejecutables y metadatos necesarios;
4. se genera un manifiesto SHA-256 del contenido;
5. se comprueba el paquete ya cerrado, no sólo su directorio de origen;
6. se registra la huella SHA-256 del ZIP;
7. sólo ese ZIP verificado se entrega a la capa de publicación.

### 17.3. Carga manual

Cuando la publicación estable se realice mediante carga manual en la aplicación autorizada de Cloudflare, la operación consistirá en cargar directamente el ZIP estático final ya verificado.

No se corregirá el contenido directamente en la consola de despliegue. Si aparece una diferencia, el paquete se regenerará desde el estado soberano y se volverá a desplegar.

La ruta administrativa concreta, las cuentas y otros datos operativos privados no forman parte de este protocolo público y se conservan únicamente en la documentación registral privada del laboratorio.

## 18. Verificación posterior al despliegue

Después de la carga se comprobará el dominio público efectivo.

Como mínimo:

- la versión visual corresponde a la autorizada;
- el perfil español e inglés son distinguibles conforme al diseño vigente;
- los perfiles compilan ejemplos válidos;
- el perfil incorrecto rechaza;
- el ensamblaje funciona;
- las descargas funcionan;
- los documentos auxiliares cargan;
- los hipervínculos muestran la representación prevista;
- el historial Beta se presenta renderizado;
- el historial estable sigue separado;
- no aparecen residuos de candidata o Beta en la interfaz estable.

Si el servicio público no coincide con el estado autorizado, el despliegue queda abierto aunque `main` esté correcto.

## 19. Conservación registral del despliegue

El laboratorio conservará, para cada despliegue estable manual relevante:

1. el ZIP exacto cargado;
2. su SHA-256;
3. un documento de procedimiento que permita reproducir la operación;
4. la identidad soberana de la que procede;
5. cuando resulte material, evidencia de la verificación posterior.

La copia registral no adquiere autoridad sobre el Lenguaje; conserva evidencia del artefacto efectivamente destinado a publicación.

## 20. Historial Beta e historial estable

Son registros distintos.

### 20.1. Historial Beta

Registra:

- capacidades ensayadas;
- procedencia;
- estado experimental;
- límites;
- propuestas y pendientes propios de la Beta;
- relación de una Beta con su eventual promoción.

### 20.2. Historial estable

Registra únicamente estados efectivamente integrados y publicados como versiones estables, junto con las necesidades futuras que el régimen estable decida registrar.

La promoción de una Beta no borra su historial experimental ni convierte todos sus pendientes en deuda estable.

## 21. Calidad y cierre de una promoción

El cierre de una promoción se realizará en dos niveles documentales cuando así se determine:

1. una primera acta de Calidad que recoja la conformidad técnica del proceso de actualización, integración y despliegue;
2. una segunda acta posterior que recoja una verificación independiente del estado resultante.

La segunda verificación no sustituye a la primera ni puede anticiparse como prueba de un estado todavía no cerrado.

Las actas públicas recogerán hechos técnicos, evidencia reproducible, decisiones, alcance, límites y estado, sin incorporar procesos internos de deliberación.

## 22. Condición de cierre

Una promoción no se considera cerrada mientras falte cualquiera de estos elementos que resulte aplicable:

- campaña previa suficiente;
- clasificación de anomalías;
- candidata limpia;
- pruebas sobre el `head` final;
- revisión funcional;
- revisión visual;
- comprobación de hipervínculos;
- autorización expresa;
- fusión exacta;
- despliegue material;
- verificación del dominio efectivo;
- conservación registral;
- actualización de Calidad.

## 23. Fallo cerrado

Ante incertidumbre sobre identidad, alcance, transporte, despliegue o correspondencia entre repositorio y servicio público se aplica fallo cerrado:

- no se infiere equivalencia;
- no se promueve por aproximación;
- no se corrige directamente en la capa pública;
- se identifica el último corte verificable;
- se repite la comprobación necesaria.

## 24. Evolución del protocolo

V2.1 conserva V1 y V2 como antecedentes históricos.

Una futura revisión deberá:

1. conservar esta versión;
2. declarar de forma expresa qué cambia;
3. mantener la separación entre núcleo, laboratorio, distribución experimental y despliegue;
4. conservar la regla de autorización expresa para abrir el laboratorio;
5. actualizar de forma sincronizada las copias vigentes destinadas a las sedes operativas correspondientes.

## 25. Estado

Con V2.1 queda fijado un procedimiento completo desde la clasificación de una necesidad hasta la comprobación del servicio público efectivamente desplegado.

La regla de cierre es:

> **El repositorio fija la identidad soberana; el despliegue material fija lo que recibe el usuario; Calidad sólo cierra cuando ambas identidades han sido comprobadas y trazadas.**
