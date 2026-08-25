"""Проверка композитора глазами: синтетический кадр рядом с настоящим.

Синтез повторяет разметку настоящего кадра — те же чемпионы, стороны и
позиции, — но собран целиком из ассетов: фон из текстур карты, значки из
circle-иконок. Так сравнение бок-о-бок отвечает на главный вопрос ступени 2:
похож ли синтетический кадр на настоящий по устройству.

Запуск: .venv/Scripts/python tools/preview_synthetic.py
Результат: out/synthetic-preview/<кадр>-<вариант>.png (настоящий | синтез).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from tcvm.cdragon import base_circle_bgra, patch_of
from tcvm.ddragon import latest_version
from tcvm.formats import bgra_to_rgb, default_corpus_dir, load_frame
from tcvm.matching import ICON_SIDE
from tcvm.synthesis import (
    ALLY_RING_BGR,
    ENEMY_RING_BGR,
    MINION_ALLY_BGR,
    MINION_ENEMY_BGR,
    MINION_SPACING,
    STRUCTURE_ALLY_BGR,
    STRUCTURE_ENEMY_BGR,
    STRUCTURE_SIDE,
    compose_background,
    draw_minion_column,
    load_darkness_mask,
    load_map_layer,
    load_minimap_icon,
    place_icon,
    ringed_icon,
    tinted_icon,
    to_uint8_bgr,
    visibility_mask,
)

DEFAULT_CORPUS = None  # ищется при запуске: см. formats.default_corpus_dir
DEFAULT_MAP_DIR = Path(__file__).resolve().parents[1] / "data" / "cdragon"
ANNOTATIONS = Path(__file__).resolve().parents[1] / "annotations"
STRUCTURES_PATH = ANNOTATIONS / "map-structures.json"
DARKNESS_PATH = ANNOTATIONS / "map-darkness.png"
CANONICAL_SIDE = 320
# Иконка и размер по типу постройки; размеры сняты при съёме позиций.
STRUCTURE_ICONS = {
    "turret": ("turret_5plate", STRUCTURE_SIDE),
    "nexus_turret": ("tower", 16),
    "inhibitor": ("inhibitor", 14),
    "nexus": ("nexus", 18),
}


def synthesize_like(frame, map_dir: Path, variant: str, patch: str) -> np.ndarray:
    """Собирает синтетический кадр по разметке настоящего.

    Источники обзора — союзные чемпионы и союзные башни из той же разметки:
    зона видимости в синтезе повторяет настоящую лишь приблизительно, но
    сравнение бок-о-бок остаётся честным по устройству.
    """
    layer = load_map_layer(map_dir / patch / "map11", variant)
    sight = [
        (r.x + r.width // 2, r.y + r.height // 2)
        for r in (frame.labels.regions if frame.labels else ())
        if r.affiliation == "Ally" and r.kind in ("Champion", "Tower")
    ]
    darkness = load_darkness_mask(DARKNESS_PATH, CANONICAL_SIDE)
    canvas = compose_background(
        layer, CANONICAL_SIDE, visibility_mask(CANONICAL_SIDE, sight), darkness
    )

    # Постройки и миньоны рисуются до чемпионов: значки чемпионов лежат поверх.
    # Постройки — все, из канонических позиций; в кадре 01 союзники на юго-западе.
    icons_dir = map_dir / patch / "minimap-icons"
    structures = json.loads(STRUCTURES_PATH.read_text(encoding="utf-8"))["structures"]
    for structure in structures:
        icon_name, side = STRUCTURE_ICONS[structure["type"]]
        tint = STRUCTURE_ALLY_BGR if structure["side"] == "southwest" else STRUCTURE_ENEMY_BGR
        icon = tinted_icon(load_minimap_icon(icons_dir, icon_name), tint)
        place_icon(canvas, icon, (structure["x"], structure["y"]), side)

    dot = load_minimap_icon(icons_dir, "minionmapcircle")
    for region in frame.labels.regions if frame.labels else ():
        center = (region.x + region.width // 2, region.y + region.height // 2)
        if region.kind == "MinionWave":
            tint = MINION_ALLY_BGR if region.affiliation == "Ally" else MINION_ENEMY_BGR
            along_height = region.height >= region.width
            start = (center[0], region.y + 3) if along_height else (region.x + 3, center[1])
            direction = (0.0, 1.0) if along_height else (1.0, 0.0)
            count = max(region.width, region.height) // MINION_SPACING + 1
            draw_minion_column(canvas, tinted_icon(dot, tint), start, direction, count)

    for region in frame.labels.champions if frame.labels else ():
        ring = ALLY_RING_BGR if region.affiliation == "Ally" else ENEMY_RING_BGR
        icon = ringed_icon(base_circle_bgra(region.champion_id, patch), ring)
        center = (region.x + region.width // 2, region.y + region.height // 2)
        place_icon(canvas, icon, center, ICON_SIDE)
    return to_uint8_bgr(canvas)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=None)
    parser.add_argument("--map-dir", type=Path, default=DEFAULT_MAP_DIR)
    parser.add_argument("--frame", default="static-tower-as-champion-01")
    parser.add_argument("--variant", default="base_baron1")
    parser.add_argument("--out", type=Path, default=Path("out/synthetic-preview"))
    args = parser.parse_args()
    corpus = args.corpus or default_corpus_dir()
    args.out.mkdir(parents=True, exist_ok=True)

    patch = patch_of(latest_version())
    frame = load_frame(corpus / f"{args.frame}.tempocue-vision")
    synthetic = synthesize_like(frame, args.map_dir, args.variant, patch)

    side = CANONICAL_SIDE * 2
    sheet = Image.new("RGB", (side * 2 + 8, side), (24, 24, 24))
    real = Image.fromarray(bgra_to_rgb(frame.pixels), "RGB").resize((side, side), Image.NEAREST)
    synth = Image.fromarray(synthetic[..., ::-1], "RGB").resize((side, side), Image.NEAREST)
    sheet.paste(real, (0, 0))
    sheet.paste(synth, (side + 8, 0))
    path = args.out / f"{args.frame}-{args.variant}.png"
    sheet.save(path)
    print(f"Настоящий | синтез: {path.resolve()}")


if __name__ == "__main__":
    main()
