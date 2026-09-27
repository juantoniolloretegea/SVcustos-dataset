# Comprobaciones documentales · 27/09/2026

**Actividad:** LP-DOC-0003/v1. Revisión en dos pasadas, sin ejecución experimental.

| Pasada | Comprobación | Resultado |
|---|---|---|
| Primera | Organización, mandato y rama | Se mantienen fichas versionadas, expedientes congelados y publicación exclusiva en laboratorio-publico |
| Primera | Fuentes públicas | Veinte documentos fijados por commit, recuperados sin autenticación y con SHA-256 y tamaño conservados |
| Primera | Cambios de estado | S32 revisión 9; S39 revisión 24; cierres de candidatos, recepción MCP y preparación del 120B separados |
| Segunda | Sucesos CSV/Markdown | Resultados S32/S39 concordantes; sin promover el estado de ningún registro canónico |
| Segunda | Alcance editorial | Sin cambios en encargos, respuestas, dictámenes ni fichas anteriores; sin contenido privado ni rutas personales en los documentos nuevos |
| Segunda | Estructura y enlaces | Enlaces locales comprobados; referencias públicas de conservación accesibles; secciones HTML equilibradas |
| Segunda | Registro | CSV y Markdown concordantes; historial ampliado sin sustituir asientos previos |

Los controles del verificador documental comprueban rutas, enlaces, manifiestos y concordancia. No son ensayos de software ni acreditación de seguridad. La presentación HTML conserva sus estilos y su estado de despliegue no se modifica.

La entrada de lectura se actualiza; LP-AUD-0002 permanece congelado en S32 revisión 8. La referencia a GPT-OSS-120B acredita únicamente su preparación documental en el corte consultado.

---


# Comprobaciones del arranque

**Fecha:** 17/09/2026. **Método:** inspección documental, Git, recuperación HTTP anónima y cálculo SHA-256. No ejecución de código SV ni de terceros.

| Control | Resultado y alcance |
|---|---|
| Mandato original local | Copia idéntica al adjunto por SHA-256; original conservado. El derivado público omite la ruta local explícitamente |
| A01 | Original conservado; destino exclusivo laboratorio-publico; sin publicación anterior en otras ramas |
| Proyecto y copia propia | Confirmados localmente; directorios de destino sin atributo de enlace o unión; ruta efectiva sólo en configuración local |
| Rama de origen | dd50c3e5c74e10526bb28087c8d5e2852d2efb68 verificado; rama preparatoria local conservada |
| Instrucciones | No se localizaron AGENTS en SVcustos-dataset ni en los directorios superiores comprobados. Leído AGENTS del Lenguaje y sus tres rectores, acta de continuidad y recepción aplicable |
| Fuentes | Diecisiete archivos públicos recuperados sin autenticación con HTTP 200; identidades y fecha en FUENTES.tsv; originales locales conservados |
| Acceso reservado | No se consultaron repositorios privados. La recepción pública no habilita recuperar sus originales |
| Revisores | Mismo expediente y rúbrica; diferencias de LEAME limitadas a procedencia y ruta. Ambas respuestas ausentes |
| Publicación Git | Commit 8e03fc2e0f150dd94d375049909a0980eb7161cc comprobado: 31 archivos con HTTP 200 e identidad SHA-256; tres enlaces GitHub accesibles. Evidencia en ENTREGA.md y VERIFICACION_PUBLICACION.tsv |
| HTML / Pages | Presentación preparada; despliegue pendiente conforme a A01. Sin cambios de configuración |
| Controles previos a publicación | Cotejo de rutas, inserción única en índice, enlaces locales, manifiesto, concordancia CSV/Markdown y ausencia de rutas locales en archivos nuevos. Resultado detallado conservado en registro local de comprobación |

El diff desde el corte de base debe contener únicamente la sede y el acceso añadido al índice padre. Los controles documentales no son acreditación de seguridad ni reproducción de las evidencias receptoras. Un fallo de acceso futuro se registra sin inferir inexistencia del objeto.
