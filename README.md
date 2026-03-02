#!/usr/bin/env python3
"""
SVcustos — Generador de dataset para entrenamiento de ResNet
=============================================================
Genera imágenes de polígonos polares ternarios organizadas en
la estructura ImageFolder de PyTorch/torchvision.

Uso:
  python generate_dataset.py --level n16
  python generate_dataset.py --level n25
  python generate_dataset.py --level n36

Cada ejecución con la misma config y seed produce exactamente
las mismas imágenes (reproducibilidad total).

DOI:   https://doi.org/10.21428/39829d0b.981b7276
ORCID: https://orcid.org/0000-0002-6634-3351
Autor: Juan Antonio Lloret Egea
Licencia: CC BY-NC-ND 4.0
"""

import argparse
import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml
from pathlib import Path


# ══════════════════════════════════════════════════════════
# VISUAL CONSTANTS (identical across all levels)
# ══════════════════════════════════════════════════════════
BG       = "#0D1B2A"    # Fondo oscuro
GOLD     = "#C9A84C"    # Línea del polígono (neutra para todas las clases)
FILL     = "#C9A84C"    # Relleno del polígono (semitransparente)
WHITE    = "#F0F4F8"
GRAY     = "#8899AA"

# Colores por VALOR (no por clase) — la CNN aprende de esto
VAL_COLORS = {
    0: "#2ECC71",   # verde = valor 0 (normal)
    1: "#E74C3C",   # rojo  = valor 1 (activo/intrusión)
    2: "#F4D03F",   # amarillo = valor U (indeterminado)
}

# Anillos de referencia
RING_COLORS = {
    1: "#2ECC71",   # radio 1 → valor 0
    2: "#E74C3C",   # radio 2 → valor 1
    3: "#F4D03F",   # radio 3 → valor U
}

CLASS_NAMES = ["INTRUSION", "INDETERMINADO", "NORMAL"]


# ══════════════════════════════════════════════════════════
# VECTOR GENERATION
# ══════════════════════════════════════════════════════════
def classify_vector(combo, threshold):
    """Classify a ternary vector using the strict rule."""
    n1 = sum(1 for v in combo if v == 1)
    n0 = sum(1 for v in combo if v == 0)
    if n1 >= threshold:
        return "INTRUSION"
    elif n0 >= threshold:
        return "NORMAL"
    else:
        return "INDETERMINADO"


def generate_vector_for_class(n, threshold, target_class, rng):
    """Generate a random ternary vector guaranteed to belong to target_class."""
    while True:
        if target_class == "INTRUSION":
            # Guarantee n1 >= threshold
            n1 = rng.integers(threshold, n + 1)
            remaining = n - n1
            n0 = rng.integers(0, remaining + 1)
            nU = remaining - n0
        elif target_class == "NORMAL":
            # Guarantee n0 >= threshold
            n0 = rng.integers(threshold, n + 1)
            remaining = n - n0
            n1 = rng.integers(0, min(remaining + 1, threshold))  # n1 < threshold
            nU = remaining - n1
        else:  # INDETERMINADO
            # Neither n1 >= threshold nor n0 >= threshold
            n1 = rng.integers(0, threshold)
            max_n0 = min(n - n1, threshold - 1)
            n0 = rng.integers(0, max_n0 + 1)
            nU = n - n1 - n0

        # Build and shuffle the vector
        combo = [1] * n1 + [0] * n0 + [2] * nU
        rng.shuffle(combo)

        # Verify classification (safety check)
        if classify_vector(combo, threshold) == target_class:
            return combo


def combo_to_index(combo):
    """Convert ternary vector to its index in the 3^n space."""
    idx = 0
    for v in combo:
        idx = idx * 3 + v
    return idx + 1


# ══════════════════════════════════════════════════════════
# POLAR POLYGON RENDERER (neutral style — no class info)
# ══════════════════════════════════════════════════════════
def render_polar_image(combo, image_size=224):
    """
    Render a ternary vector as a polar polygon image.

    The image uses NEUTRAL styling (same colors for all classes)
    so the CNN must learn from the GEOMETRIC PATTERN, not from
    color-coded class information.

    Points are colored by VALUE (0/1/U), which is structural
    information, not class labeling.
    """
    n = len(combo)
    dpi = 100
    fig_size = image_size / dpi

    fig = plt.figure(figsize=(fig_size, fig_size), dpi=dpi, facecolor=BG)
    ax = fig.add_axes([0.05, 0.05, 0.90, 0.90], polar=True, facecolor=BG)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)

    # Map: 0→radius 1, 1→radius 2, U→radius 3
    radius_map = {0: 1, 1: 2, 2: 3}
    radii = [radius_map[v] for v in combo]

    # Close polygon
    angles_closed = np.append(angles, angles[0])
    radii_closed = radii + [radii[0]]

    # Reference rings (subtle, structural)
    theta_ring = np.linspace(0, 2 * np.pi, 200)
    for r, color in RING_COLORS.items():
        ax.plot(theta_ring, np.full(200, r),
                color=color, linewidth=0.6, linestyle='--',
                alpha=0.25, zorder=1)

    # Axis lines
    for a in angles:
        ax.plot([a, a], [0, 3.3], color=GRAY, alpha=0.20, linewidth=0.4)

    # Polygon fill + outline (NEUTRAL color — same for ALL classes)
    ax.fill(angles_closed, radii_closed,
            alpha=0.15, color=FILL, zorder=2)
    ax.plot(angles_closed, radii_closed,
            color=GOLD, linewidth=1.5, alpha=0.85, zorder=3)

    # Points colored by VALUE (structural, not class)
    for i, (a, r) in enumerate(zip(angles, radii)):
        ax.scatter(a, r, c=VAL_COLORS[combo[i]], s=25, zorder=5,
                   edgecolors=WHITE, linewidths=0.5)

    # Clean axes
    ax.set_ylim(0, 3.8)
    ax.set_yticks([])
    ax.yaxis.set_visible(False)
    ax.set_xticks([])
    ax.grid(False)
    ax.spines['polar'].set_visible(False)

    # Render to numpy array
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    img = img.reshape(h, w, 4)[:, :, :3]  # drop alpha
    plt.close(fig)

    return img


# ══════════════════════════════════════════════════════════
# DATASET GENERATION PIPELINE
# ══════════════════════════════════════════════════════════
def generate_dataset(config_path):
    """Generate the complete ImageFolder dataset from a config file."""
    # Load config
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    n = cfg["level"]["n"]
    threshold = cfg["level"]["threshold_intrusion"]
    samples_per_class = cfg["dataset"]["samples_per_class"]
    split = cfg["dataset"]["split"]
    seed = cfg["dataset"]["seed"]
    image_size = cfg["dataset"]["image_size"]
    doc = cfg["level"]["document"]

    print(f"═══ SVcustos Dataset Generator ═══")
    print(f"  Level: n={n} (Documento {doc})")
    print(f"  Threshold: n₁ ≥ {threshold} (INTRUSIÓN) / n₀ ≥ {threshold} (NORMAL)")
    print(f"  Space: 3^{n} = {3**n:,} vectors")
    print(f"  Samples: {samples_per_class}/class × 3 = {samples_per_class * 3} images")
    print(f"  Split: {split}")
    print(f"  Image size: {image_size}×{image_size} px")
    print(f"  Seed: {seed}")
    print()

    rng = np.random.default_rng(seed)

    # Output directory
    base_dir = Path(__file__).parent / "data" / f"n{n}"

    # Create directory structure
    splits = {
        "train": int(samples_per_class * split["train"]),
        "val": int(samples_per_class * split["val"]),
    }
    splits["test"] = samples_per_class - splits["train"] - splits["val"]

    for split_name in ["train", "val", "test"]:
        for cls_name in CLASS_NAMES:
            (base_dir / split_name / cls_name).mkdir(parents=True, exist_ok=True)

    print(f"  Directory: {base_dir}")
    print(f"  Split: train={splits['train']}, val={splits['val']}, test={splits['test']} per class")
    print()

    start = time.time()
    total_generated = 0

    for cls_name in CLASS_NAMES:
        print(f"  Generating {samples_per_class} {cls_name} images...")
        cls_start = time.time()

        # Generate all vectors for this class
        vectors = []
        seen = set()
        while len(vectors) < samples_per_class:
            combo = generate_vector_for_class(n, threshold, cls_name, rng)
            key = tuple(combo)
            if key not in seen:
                seen.add(key)
                vectors.append(combo)

        # Shuffle and split
        indices = list(range(samples_per_class))
        rng.shuffle(indices)

        split_assignment = {}
        offset = 0
        for split_name, count in splits.items():
            for i in indices[offset:offset + count]:
                split_assignment[i] = split_name
            offset += count

        # Render and save
        for i, combo in enumerate(vectors):
            split_name = split_assignment[i]
            idx = combo_to_index(combo)

            img = render_polar_image(combo, image_size)

            # Filename encodes the vector index for traceability
            filename = f"sv_n{n}_{cls_name.lower()}_idx{idx}_{i:04d}.png"
            filepath = base_dir / split_name / cls_name / filename

            plt.imsave(str(filepath), img)
            total_generated += 1

            if (i + 1) % 100 == 0:
                elapsed = time.time() - cls_start
                rate = (i + 1) / elapsed
                print(f"    {i+1}/{samples_per_class} ({rate:.0f} img/s)")

        cls_elapsed = time.time() - cls_start
        print(f"    Done: {samples_per_class} images in {cls_elapsed:.1f}s")
        print()

    total_elapsed = time.time() - start

    # Summary
    print(f"═══ GENERATION COMPLETE ═══")
    print(f"  Total images: {total_generated}")
    print(f"  Time: {total_elapsed:.1f}s ({total_generated/total_elapsed:.0f} img/s)")
    print()

    # Verify counts
    print(f"  Verification:")
    for split_name in ["train", "val", "test"]:
        for cls_name in CLASS_NAMES:
            count = len(list((base_dir / split_name / cls_name).glob("*.png")))
            expected = splits[split_name]
            status = "✅" if count == expected else "❌"
            print(f"    {status} {split_name}/{cls_name}: {count} images")

    # Disk usage
    total_bytes = sum(f.stat().st_size for f in base_dir.rglob("*.png"))
    print(f"\n  Disk usage: {total_bytes / 1024 / 1024:.1f} MB")
    print(f"  Location: {base_dir.resolve()}")


# ══════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SVcustos — Generate polar polygon dataset for ResNet training"
    )
    parser.add_argument(
        "--level", required=True,
        choices=["n16", "n25", "n36"],
        help="Target level (n16, n25, n36)"
    )
    parser.add_argument(
        "--config-dir", default="config",
        help="Directory containing YAML configs (default: config/)"
    )
    args = parser.parse_args()

    config_path = Path(args.config_dir) / f"{args.level}.yaml"
    if not config_path.exists():
        print(f"Error: config not found at {config_path}")
        sys.exit(1)

    generate_dataset(config_path)
