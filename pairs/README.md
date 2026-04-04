# pairs/ — Células SV en par (Documento 5)

**Fecha y Versión: V.1 del conjunto**  
**Fecha:** 4 de abril de 2026  
**Versión del conjunto:** V.1 del conjunto  
**Autor del corpus:** Juan Antonio Lloret Egea  
**ORCID:** 0000-0002-6634-3351  
**Institución:** ITVIA — IA eñ™  
**ISSN:** 2695-6411  
**Licencia:** CC BY-NC-ND 4.0  
**Titularidad y autoría:** © Juan Antonio Lloret Egea, 2026. Este conjunto se distribuye con atribución explícita de autoría y bajo la licencia indicada, sin autorización para apropiación de la paternidad intelectual del Sistema Vectorial SV.  

---

## Visualización del espacio vectorial

> ▶ **Previsualización del vídeo** — Al hacer clic se descarga el archivo MP4, calidad original.

[![SVcustos — Células SV en par · Documento 5 de 8](https://raw.githubusercontent.com/juantoniolloretegea/SVcustos-dataset/main/assets/thumbnail_par.png)](https://raw.githubusercontent.com/juantoniolloretegea/SVcustos-dataset/main/assets/SVcustos_Documento5_video.mp4)
*Arquitectura dual-cell · SV(36,6) + SV(9,3) · 45 parámetros · Documento 5 de 8*

## Relación con la serie documental

Este subproyecto implementa el componente experimental descrito en el
**Documento 5 – «Células SV en par: n = 36 + n = 9»** de la serie
«De SVcustos, el marco (framework) de intrusión, hasta SVperitus:
agentes especializados».

En dicho documento se define la célula compuesta:

- **Entrada:** salidas individuales de SV(36,6) y SV(9,3).
- **Regla de decisión:**

> SV(par) = max(SV(36,6), SV(9,3))
> con severidad INTRUSIÓN > INDETERMINADO > NORMAL.

## Composición determinista

La regla `max()` es estrictamente algebraica y determinista: no requiere
un tercer clasificador entrenado. Cada célula mantiene su propio dataset
de entrenamiento y su propia instancia de ResNet34. La decisión del par
se resuelve con una operación de comparación sobre las salidas ternarias.

Este subproyecto **no redefine** una nueva célula de 45 parámetros ni
entrena un tercer modelo sobre imágenes compuestas; se limita a modelar
de forma explícita la decisión conjunta a partir de las salidas
individuales, preservando las propiedades algebraicas (conmutatividad,
asociatividad e idempotencia) de la composición ternaria.

## Contenido

| Fichero | Función |
|---------|---------|
| `generate_pair_dataset.py` | Genera pares de salidas (cls₃₆, cls₉), aplica `max()` y opcionalmente produce visualizaciones duales |
| `../config/pair_n36_n9.yaml` | Configuración del par: células referenciadas, regla y orden de severidad |
| `../config/n9.yaml` | Configuración de la célula de integridad SV(9,3) |
| `../config/n36.yaml` | Configuración de la célula principal SV(36,6) |

## Estructura de salida
```text
data/pairs/n36_n9/
  INTRUSION/
  INDETERMINADO/
  NORMAL/
```

Cada imagen contiene ambos polígonos (n=36 y n=9) y está etiquetada con
la clasificación final del par.

## Relación con SVperitus

La arquitectura de este subproyecto (células simples + composición ternaria)
está pensada para servir más adelante como plantilla para combinaciones
de células en el marco **SVperitus**, aunque SVperitus se desarrollará en
un repositorio o subproyecto independiente.
