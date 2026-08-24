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
from pathlib import Path

import numpy as np
from PIL import Image

from tcvm.cdragon import base_circle_bgra, patch_of
from tcvm.ddragon import latest_version
from tcvm.formats import bgra_to_rgb, load_frame
from tcvm.matching import ICON_SIDE
from tcvm.synthesis import (
    ALLY_RING_BGR,
    ENEMY_RING_BGR,
    compose_background,
    load_map_layer,
    place_icon,
    ringed_icon,
    to_uint8_bgr,
    visibility_mask,
)

DEFAULT_CORPUS = Path(
    r"C:\Users\deokn\.codex\worktrees\739a\PROJECT\tests\TempoCue.Vision.Tests\ReplayCorpus"
)
DEFAULT_MAP_DIR = Path(__file__).resolve().parents[1] / "data" / "cdragon"
CANONICAL_SIDE = 320


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
    canvas = compose_background(layer, CANONICAL_SIDE, visibility_mask(CANONICAL_SIDE, sight))
    for region in frame.labels.champions if frame.labels else ():
        ring = ALLY_RING_BGR if region.affiliation == "Ally" else ENEMY_RING_BGR
        icon = ringed_icon(base_circle_bgra(region.champion_id, patch), ring)
        center = (region.x + region.width // 2, region.y + region.height // 2)
        place_icon(canvas, icon, center, ICON_SIDE)
    return to_uint8_bgr(canvas)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--map-dir", type=Path, default=DEFAULT_MAP_DIR)
    parser.add_argument("--frame", default="static-tower-as-champion-01")
    parser.add_argument("--variant", default="base_baron1")
    parser.add_argument("--out", type=Path, default=Path("out/synthetic-preview"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    patch = patch_of(latest_version())
    frame = load_frame(args.corpus / f"{args.frame}.tempocue-vision")
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
