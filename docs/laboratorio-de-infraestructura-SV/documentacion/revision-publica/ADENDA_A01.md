# LP-DOC-0001 · Adenda A01: rama pública independiente

**Fecha:** 17/09/2026. **Unidad destinataria:** U-DOC-PUBLICA.  
**Documento complementado:** ENCARGO_LP_DOC_0001_ARRANQUE_v1.md.  
**Origen:** precisión expresa de la Dirección humana sobre la separación de ramas y el acceso público de los revisores.

## 1. Prevalencia y autorización

Esta adenda sustituye exclusivamente las instrucciones de rama, integración y publicación afectadas del encargo v1, especialmente §§2, 6 y 8. El encargo original se conserva como antecedente. El mandato vigente es **LP-DOC-0001/v1 + A01**. Su transmisión por la Dirección autoriza esta modificación; no autoriza más rutas, repositorios ni capacidades.

El trabajo documental compatible puede continuar. Antes de la siguiente escritura remota, incorporar esta adenda y comprobar la rama destino. No reiniciar ni repetir trabajo ya comprobado por el mero cambio de rama.

## 2. Repositorio y rama únicos de publicación

| Objeto | Disposición vigente |
|---|---|
| Repositorio público | `juantoniolloretegea/SVcustos-dataset` |
| Rama remota de trabajo, custodia y publicación documental | **`laboratorio-publico`** |
| Rama `main` | Sólo lectura; no publicar ni integrar cambios. |
| Rama `lab/lenguaje-sv-beta` | Fuente documental histórica; sólo lectura desde la recepción de esta adenda. |
| Rama temporal `doc/laboratorio-publico-0001` mencionada en v1 | Deja de ser destino de publicación. Si existe, conservar sus cambios y registrar su estado; no borrarla ni reescribirla. |
| Repositorio privado de ejecución y demás repositorios | Conservan las restricciones de v1; sin escritura ni divulgación de contenido privado. |

Crear `laboratorio-publico` si no existe, a partir del corte público documental `dd50c3e5c74e10526bb28087c8d5e2852d2efb68` de `lab/lenguaje-sv-beta`, después de verificar su identidad y disponibilidad. Ese corte permite reutilizar la presentación existente; no constituye el estado científico actual, que se obtiene de las fuentes canónicas identificadas en cada ficha.

Si la rama ya existe, examinar su contenido y ascendencia. No recrearla, resetearla, forzarla ni sobrescribir trabajo ajeno. Conciliar sólo cambios compatibles y dentro del alcance; una discrepancia sustantiva se eleva a la Dirección. Mantener una copia de trabajo propia.

Las únicas rutas de contenido autorizadas siguen siendo, **dentro de `laboratorio-publico`**:

- `docs/laboratorio-de-infraestructura-SV/documentacion/revision-publica/**`.
- `docs/laboratorio-de-infraestructura-SV/documentacion/index.html`, únicamente para añadir el acceso previsto a la nueva sede en esta rama.

No abrir una integración hacia `main` ni hacia `lab/lenguaje-sv-beta`. No actualizar el índice de la rama histórica. Los archivos heredados al crear la rama son antecedentes conservados; no quedan habilitados para edición por estar presentes en ella.

## 3. Acceso de los revisores y custodia

Encargos, fuentes públicas, manifiestos, respuestas publicables y dictámenes de este laboratorio residirán en `SVcustos-dataset`, rama `laboratorio-publico`. No se exigirá acceso al repositorio privado para recoger o responder al encargo.

Mantener las dos sedes superiores `encargos/` y `respuestas/` bajo `revision-publica/`, y las rutas separadas por revisor establecidas en v1. Ambos revisores recibirán el mismo expediente congelado; cada uno tendrá su ruta de entrega independiente.

La referencia operativa puede señalar la rama. El encargo transmitido, sus fuentes y las respuestas se enlazarán al **commit completo efectivamente publicado**, con su ruta exacta. No presentar como existentes enlaces de archivos todavía no publicados ni usar una punta móvil como identidad de la revisión autorizada.

Verificar que el repositorio continúa público y que los archivos necesarios pueden recuperarse sin autenticación. Si un entorno revisor no puede abrirlos, registrar la limitación y facilitar una copia pública equivalente identificada, sin atribuirle una lectura no realizada. La disponibilidad pública permite lectura; no concede permiso de escritura. Las reglas de recepción mediante la Dirección y depósito por U-DOC-PUBLICA permanecen vigentes.

Una rama separada organiza versiones, pero no constituye una frontera de confidencialidad: tratar todo lo publicado en ella como público. No copiar contenidos privados para completar el expediente. Si una conclusión depende de evidencia reservada, precisar qué puede y qué no puede verificar un revisor externo.

## 4. Diferencia entre archivos públicos y GitHub Pages

La publicación de archivos en `laboratorio-publico` permite entregar enlaces públicos de GitHub. **No implica que la web existente de GitHub Pages cambie de rama ni muestre automáticamente esos archivos.** GitHub Pages depende de su fuente de publicación o del workflow configurado.

Conservar la configuración y el despliegue actuales: no cambiar Pages, DNS, workflows ni trasladar el playground. En esta fase, proporcionar como acceso operativo el índice Markdown y los expedientes públicos por commit. La presentación HTML de la nueva rama queda preparada; su despliegue web se declarará pendiente si no existe una vía ya autorizada y comprobada que lo publique sin modificar otras ramas.

Si la Dirección requiere posteriormente la presentación web desplegada de esta rama, presentar una propuesta acotada que identifique destino, efectos sobre el sitio existente y archivos o configuración a cambiar. No ejecutar esa modificación como consecuencia implícita de esta adenda.

Referencia técnica: [Configuración de la fuente de publicación de GitHub Pages](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site).

## 5. Conciliación del trabajo iniciado

1. Conservar el encargo v1, esta adenda y el trabajo local existente. Registrar la recepción de A01 y la última operación confirmada.
2. Si no se publicó todavía, trasladar únicamente los cambios documentales propios permitidos a `laboratorio-publico`, conservando su procedencia. No arrastrar modificaciones ajenas.
3. Si ya se publicó en la rama indicada por v1, registrar commit, rutas y alcance como antecedente autorizado antes de A01; dejar de escribir allí. Incorporar los cambios pertinentes a la nueva rama. No borrar ni revertir silenciosamente lo anterior. Cualquier modificación correctora de aquella rama requiere instrucción específica de la Dirección.
4. Actualizar el mandato público derivado, LEAME_PRIMERO, registro propio, estado y enlaces para que indiquen `laboratorio-publico`. Conservar el historial de la modificación.
5. Cotejar antes de publicar que el destino y todas las rutas están autorizados; después comprobar el commit remoto y los enlaces públicos. Informar separadamente sobre archivos publicados, acceso comprobado y despliegue HTML.

**Punto de parada conservado:** sede documental y LP-AUD-0001 preparados; entregar a la Dirección los enlaces públicos fijados de los dos encargos. Las revisiones externas se activan por transmisión humana. Permanecen vigentes las reglas de respuestas separadas, conservación local, registro propio, recomendación de elevación y autorización humana de actuaciones posteriores.
