"""Извлечение рамки миникарты из настоящих кадров в annotations/map-border.png.

Карта занимает в каноническом кадре не всю сторону: вокруг неё рамка
интерфейса. В генераторе её не было вовсе, и синтетика отличалась от живого
кадра по всей кайме — а это шестая часть площади.

Рамка снимается с корпуса как медиана по кадрам вне области карты. Право так
делать даёт замер: расхождение кадров в кайме равно нулю — элемент статичный
и одинаковый во всех записях, как маска тьмы. Внутрь области карты
инструмент не смотрит вовсе, поэтому чемпионы и постройки в рамку не попадают.

    .venv/Scripts/python tools/extract_border.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from tcvm.formats import bgra_to_rgb, load_corpus
from tcvm.generator import CANONICAL_SIDE
from tcvm.synthesis import MAP_PLACEMENT, map_rect

DEFAULT_CORPUS = Path(
    r"C:\Users\deokn\.codex\worktrees\739a\PROJECT\tests\TempoCue.Vision.Tests\ReplayCorpus"
)
ANNOTATIONS = Path(__file__).resolve().parents[1] / "annotations"
BORDER_PATH = ANNOTATIONS / "map-border.png"
OPAQUE = 255


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    args = parser.parse_args()

    frames = load_corpus(args.corpus)
    stack = np.stack([bgra_to_rgb(frame.pixels).astype(np.float32) for frame in frames])
    median = np.median(stack, axis=0)
    spread = np.median(np.abs(stack - median), axis=0).mean(axis=2)

    left, top, right, bottom = map_rect(CANONICAL_SIDE, MAP_PLACEMENT)
    outside = np.ones((CANONICAL_SIDE, CANONICAL_SIDE), dtype=bool)
    outside[top:bottom, left:right] = False

    print(f"Кадров: {len(frames)}; область карты {left}..{right} по x, {top}..{bottom} по y")
    print(
        f"Расхождение кадров в кайме: медиана {np.median(spread[outside]):.2f}, "
        f"p90 {np.percentile(spread[outside], 90):.2f}, максимум {spread[outside].max():.2f}"
    )

    rgba = np.zeros((CANONICAL_SIDE, CANONICAL_SIDE, 4), dtype=np.uint8)
    rgba[..., :3] = np.clip(median + 0.5, 0, 255).astype(np.uint8)
    rgba[outside, 3] = OPAQUE
    Image.fromarray(rgba, "RGBA").save(BORDER_PATH)
    print(f"Рамка записана: {BORDER_PATH} ({outside.mean():.3f} площади кадра)")


if __name__ == "__main__":
    main()
