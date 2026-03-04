#!/usr/bin/env python3
"""
generate_pair_dataset.py — Documento 5 de 8
Genera el dataset del par SV(36,6) + SV(9,3).

Regla de composición:
    SV(par) = max(cls_36, cls_9)
    con INTRUSIÓN (2) > INDETERMINADO (1) > NORMAL (0)

Este script NO entrena un tercer clasificador. Genera pares de vectores
ternarios, clasifica cada célula independientemente con su umbral, aplica
max() y opcionalmente produce visualizaciones duales.

Uso:
    python pairs/generate_pair_dataset.py
    python pairs/generate_pair_dataset.py --config config/pair_n36_n9.yaml
    python pairs/generate_pair_dataset.py --visualize
"""

import argparse
import os
import sys
import math
import random
import yaml
import numpy as np

# Severidad: NORMAL=0, INDETERMINADO=1, INTRUSION=2
SEVERITY = {"NORMAL": 0, "INDETERMINADO": 1, "INTRUSION": 2}
LABELS = {0: "NORMAL", 1: "INDETERMINADO", 2: "INTRUSION"}


def load_config(path="config/pair_n36_n9.yaml"):
    """Carga la configuración del par."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def generate_ternary_vector(n, rng):
    """Genera un vector ternario aleatorio de longitud n."""
    return [rng.choice([0, 1, 2]) for _ in range(n)]


def classify_cell(vector, threshold):
    """
    Clasifica un vector ternario según la regla estricta.
    Valores: 0=normal, 1=intrusión, 2=indeterminado(U)
    """
    n1 = sum(1 for v in vector if v == 1)  # parámetros activos
    n0 = sum(1 for v in vector if v == 0)  # parámetros normales
    if n1 >= threshold:
        return SEVERITY["INTRUSION"]
    elif n0 >= threshold:
        return SEVERITY["NORMAL"]
    else:
        return SEVERITY["INDETERMINADO"]


def compose_pair(cls_36, cls_9):
    """
    Regla de composición: max(cls_36, cls_9).
    Algebraica, determinista, conmutativa, asociativa, idempotente.
    """
    return max(cls_36, cls_9)


def generate_pair_samples(config, n_samples, rng):
    """Genera n_samples pares clasificados."""
    n36 = config["cells"]["principal"]["n"]
    n9 = config["cells"]["integrity"]["n"]
    t36 = config["cells"]["principal"]["threshold_intrusion"]
    t9 = config["cells"]["integrity"]["threshold_intrusion"]

    samples = []
    for _ in range(n_samples):
        v36 = generate_ternary_vector(n36, rng)
        v9 = generate_ternary_vector(n9, rng)
        cls_36 = classify_cell(v36, t36)
        cls_9 = classify_cell(v9, t9)
        cls_pair = compose_pair(cls_36, cls_9)
        samples.append({
            "v36": v36,
            "v9": v9,
            "cls_36": cls_36,
            "cls_9": cls_9,
            "cls_pair": cls_pair,
            "label": LABELS[cls_pair],
        })
    return samples


def print_distribution(samples):
    """Muestra la distribución de clases."""
    from collections import Counter
    counts = Counter(s["label"] for s in samples)
    total = len(samples)
    print(f"\nDistribución del par ({total} muestras):")
    for label in ["INTRUSION", "INDETERMINADO", "NORMAL"]:
        n = counts.get(label, 0)
        pct = 100 * n / total if total > 0 else 0
        print(f"  {label:16s}: {n:6d}  ({pct:.1f}%)")

    # Discrepancias
    disc = sum(1 for s in samples if s["cls_36"] != s["cls_9"])
    print(f"\n  Discrepancias entre células: {disc} ({100*disc/total:.1f}%)")


def print_composition_table(samples):
    """Muestra la tabla 3x3 de composición observada."""
    from collections import Counter
    combos = Counter((LABELS[s["cls_36"]], LABELS[s["cls_9"]]) for s in samples)
    labels_order = ["INTRUSION", "INDETERMINADO", "NORMAL"]
    print("\nTabla de composición observada (SV(36) filas × SV(9) columnas):")
    header = f"{'':16s} | {'INTRUSION':>12s} | {'INDETERMINADO':>14s} | {'NORMAL':>8s}"
    print(header)
    print("-" * len(header))
    for row in labels_order:
        cells = []
        for col in labels_order:
            cells.append(f"{combos.get((row, col), 0):>12d}")
        print(f"{row:16s} | {' | '.join(cells)}")


def save_dataset(samples, output_dir, config):
    """Guarda los pares como CSV para trazabilidad."""
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "pair_dataset.csv")
    with open(csv_path, "w") as f:
        f.write("v36,v9,cls_36,cls_9,cls_pair,label\n")
        for s in samples:
            v36_str = "".join(str(x) for x in s["v36"])
            v9_str = "".join(str(x) for x in s["v9"])
            f.write(f"{v36_str},{v9_str},"
                    f"{LABELS[s['cls_36']]},{LABELS[s['cls_9']]},"
                    f"{s['label']},{s['label']}\n")
    print(f"\nDataset guardado en: {csv_path}")
    return csv_path


def main():
    parser = argparse.ArgumentParser(
        description="Genera dataset del par SV(36,6)+SV(9,3) — Documento 5"
    )
    parser.add_argument(
        "--config", default="config/pair_n36_n9.yaml",
        help="Ruta al fichero de configuración del par"
    )
    parser.add_argument(
        "--samples", type=int, default=3000,
        help="Número total de pares a generar (default: 3000)"
    )
    parser.add_argument(
        "--visualize", action="store_true",
        help="Generar visualizaciones duales (requiere matplotlib)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Semilla para reproducibilidad (default: 42)"
    )
    args = parser.parse_args()

    # Cargar configuración
    config = load_config(args.config)
    print(f"Configuración: {args.config}")
    print(f"Par: {config['pair']['name']}")
    print(f"Regla: {config['pair']['rule']}(cls_36, cls_9)")

    # Generar muestras
    rng = random.Random(args.seed)
    samples = generate_pair_samples(config, args.samples, rng)

    # Estadísticas
    print_distribution(samples)
    print_composition_table(samples)

    # Guardar
    output_dir = config.get("dataset", {}).get("output_dir", "data/pairs/n36_n9")
    save_dataset(samples, output_dir, config)

    if args.visualize:
        print("\n[Visualización dual pendiente de implementación]")
        print("Cada imagen contendrá polígono n=36 (izq) + polígono n=9 (der)")
        print("etiquetada con la clasificación del par.")


if __name__ == "__main__":
    main()
