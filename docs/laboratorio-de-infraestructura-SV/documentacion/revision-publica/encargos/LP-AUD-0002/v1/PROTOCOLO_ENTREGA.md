# Protocolo obligatorio de rama, permisos y entrega

## Destino único

Repositorio: **juantoniolloretegea/SVcustos-dataset**.
Rama remota: **laboratorio-publico**.
Ruta: exclusivamente la carpeta de respuesta propia indicada en el LEAME del revisor.

El enlace por commit congela la entrada. No convierte ese commit en una rama de escritura: una entrega autorizada se añade sobre la punta vigente de laboratorio-publico, conservando avances concurrentes.

No se permite escribir en main, lab/lenguaje-sv-beta, doc/laboratorio-publico-0001, cualquier otra rama, fork o repositorio. No crear una rama o PR alternativa porque una herramienta no pueda seleccionar laboratorio-publico. No fusionar, resetear, forzar referencias, borrar ramas, cambiar protecciones o utilizar enlaces/uniones que salgan del destino permitido.

## Autorización y comprobación antes de escribir

La lectura pública no concede escritura. La Dirección debe autorizar expresamente la escritura de esta entrega; sin ella, devolver el informe a la Dirección.

Si hay autorización, comprobar antes de cada escritura:
1. Identidad exacta del repositorio, visibilidad y rama destino admitida por la herramienta.
2. Punta remota actual y ausencia de cambios ajenos en la copia propia.
3. Que cada archivo propuesto pertenece a la carpeta propia; no modificar fuentes, encargos, registros, fichas, otras respuestas o dictámenes.
4. Que la herramienta especifica laboratorio-publico de forma inequívoca y no aplica main por defecto. Si no puede acreditarse, no escribir.
5. Que la entrega es publicable, sin originales privados, secretos, credenciales o datos ajenos al alcance.

La separación de ramas es una regla operativa, no un aislamiento de seguridad certificado. Los permisos que ofrezca un conector no amplían la autorización humana.

## Declaración expresa de impedimentos

La respuesta incluirá siempre estos campos, aunque no haya escritura:

- Autorización humana de escritura: recibida / no recibida; referencia y alcance.
- Capacidad de publicación en laboratorio-publico: comprobada / no disponible / no comprobada.
- Publicación realizada: sí / no.
- Repositorio y rama de destino efectivos, o «ninguno; no se escribió».
- Impedimento observado: restricción de herramienta/sistema, permisos, autenticación, rama no seleccionable, conflicto u otro; diagnóstico suficiente sin secretos. Si no se conoce la causa, decir «causa no determinada».
- Escrituras fuera del destino permitido: ninguna / incidente observado y alcance conocido.
- Commit y archivos publicados, sólo si existen y fueron comprobados.

**Si Claude no puede escribir en el destino, deberá indicar expresamente:**

> No he podido publicar mi respuesta en juantoniolloretegea/SVcustos-dataset, rama laboratorio-publico, ruta asignada. Motivo observado: [motivo o causa no determinada]. No he utilizado otra rama ni otro repositorio como alternativa. Entrego [texto/archivo] a la Dirección para su depósito por U-DOC-PUBLICA.

Si falta autorización en vez de capacidad, escribir «No he publicado porque no he recibido autorización humana expresa de escritura», sin atribuir un bloqueo técnico inexistente. Los demás revisores aplicarán la misma declaración a su propia entrega.

La imposibilidad de publicar no se oculta detrás de «tarea completada», un enlace inventado o un supuesto commit. No se debe probar escritura en un destino no autorizado para diagnosticarla.

## Acceso privado y devolución

Declarar por separado cada evidencia privada que no se pudo examinar, el motivo conocido y las conclusiones que quedan no comprobables. Una referencia o hash no sustituye su lectura. El presente encargo no autoriza acceso privado; cualquier ampliación exige instrucción específica y una nueva delimitación. No reproducir contenido privado en la respuesta pública.

Sin escritura autorizada y disponible, devolver el informe íntegro a la Dirección. U-DOC-PUBLICA podrá custodiar y depositar la entrega cuando reciba esa instrucción, identificando procedencia y hash, sin simular acceso del revisor.

Con escritura autorizada y disponible, añadir sólo entrega-01; verificar remotamente commit, ruta y contenido. Si ya existe respuesta, no sobrescribirla: aclarar su procedencia y usar entrega-02 sólo como corrección autorizada. Conservar todas las versiones.

Antes de la primera entrega no buscar deliberadamente la respuesta del otro revisor. Declarar cualquier exposición previa. Entregar y detenerse; no iniciar reparaciones ni contactar otras unidades.
