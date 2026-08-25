"""Съём канонических позиций построек с настоящего кадра.

Постройки стоят в фиксированных точках карты, поэтому их позиции снимаются
один раз: полный скан кадра тонированными шаблонами иконок (башня с пятью
пластинами в обоих командных цветах), подавление соседних совпадений и
оверлей для проверки глазами. Результат дописывается вручную (сооружения баз
плохо разделяются шаблоном) и фиксируется в annotations/map-structures.json.

Сторона в файле — геометрическая (southwest/northeast), а не ally/enemy:
сторона игрока — ось рандомизации, а геометрия карты постоянна.

Запуск: .venv/Scripts/python tools/extract_structures.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from tcvm.cdragon import patch_of
from tcvm.ddragon import latest_version
from tcvm.formats import bgra_to_rgb, default_corpus_dir, load_frame
from tcvm.matching import masked_ncc
from tcvm.synthesis import (
    STRUCTURE_ALLY_BGR,
    STRUCTURE_ENEMY_BGR,
    STRUCTURE_SIDE,
    VISIBLE_ALPHA,
    load_minimap_icon,
    tinted_icon,
)

DEFAULT_CORPUS = None  # ищется при запуске: см. formats.default_corpus_dir
DEFAULT_MAP_DIR = Path(__file__).resolve().parents[1] / "data" / "cdragon"
# Порог найденной башни: лучший NCC настоящей башни против шаблона — 0,74
# (иконка на кадре сидит на фоне и свечении, идеала не бывает); порог ниже с
# запасом, ложные кандидаты отсеиваются глазами по оверлею.
MATCH_THRESHOLD = 0.55
SUPPRESS_RADIUS = 8


def template_of(icons_dir: Path, tint: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Шаблон башни в канонич. размере и маска его видимых пикселей."""
    icon = tinted_icon(load_minimap_icon(icons_dir, "turret_5plate"), tint)
    scaled = Image.fromarray(icon[..., [2, 1, 0, 3]], "RGBA").resize(
        (STRUCTURE_SIDE, STRUCTURE_SIDE), Image.BILINEAR
    )
    bgra = np.asarray(scaled)[..., [2, 1, 0, 3]]
    return bgra, bgra[..., 3] > VISIBLE_ALPHA


def scan(frame_pixels: np.ndarray, template: np.ndarray, mask: np.ndarray) -> list:
    """Полный скан кадра: NCC шаблона в каждой позиции, локальные максимумы."""
    side = template.shape[0]
    height, width = frame_pixels.shape[:2]
    scores = np.full((height, width), -1.0)
    for y in range(0, height - side):
        for x in range(0, width - side):
            crop = frame_pixels[y : y + side, x : x + side]
            scores[y + side // 2, x + side // 2] = masked_ncc(crop, template, mask)

    found = []
    working = scores.copy()
    while True:
        peak = np.unravel_index(np.argmax(working), working.shape)
        score = working[peak]
        if score < MATCH_THRESHOLD:
            break
        found.append((int(peak[1]), int(peak[0]), float(score)))
        y0 = max(0, peak[0] - SUPPRESS_RADIUS)
        x0 = max(0, peak[1] - SUPPRESS_RADIUS)
        working[y0 : peak[0] + SUPPRESS_RADIUS, x0 : peak[1] + SUPPRESS_RADIUS] = -1.0
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=None)
    parser.add_argument("--map-dir", type=Path, default=DEFAULT_MAP_DIR)
    parser.add_argument("--frame", default="static-tower-as-champion-01")
    parser.add_argument("--out", type=Path, default=Path("out/structures"))
    args = parser.parse_args()
    corpus = args.corpus or default_corpus_dir()
    args.out.mkdir(parents=True, exist_ok=True)

    patch = patch_of(latest_version())
    icons_dir = args.map_dir / patch / "minimap-icons"
    frame = load_frame(corpus / f"{args.frame}.tempocue-vision")

    overlay = Image.fromarray(bgra_to_rgb(frame.pixels), "RGB").resize(
        (640, 640), Image.NEAREST
    )
    draw = ImageDraw.Draw(overlay)
    for side_name, tint, color in (
        ("southwest", STRUCTURE_ALLY_BGR, (0, 255, 70)),
        ("northeast", STRUCTURE_ENEMY_BGR, (255, 160, 0)),
    ):
        template, mask = template_of(icons_dir, tint)
        found = scan(frame.pixels, template, mask)
        print(f"{side_name}: найдено {len(found)}")
        for x, y, score in sorted(found, key=lambda f: (f[1], f[0])):
            print(
                f'  {{ "type": "turret", "side": "{side_name}", "x": {x}, "y": {y} }},'
                f"  # балл {score:.2f}"
            )
            draw.ellipse(
                [x * 2 - 10, y * 2 - 10, x * 2 + 10, y * 2 + 10], outline=color, width=2
            )
            draw.text((x * 2 + 10, y * 2 - 10), f"{score:.2f}", fill=color)

    path = args.out / f"{args.frame}-structures.png"
    overlay.save(path)
    print(f"Оверлей: {path.resolve()}")


if __name__ == "__main__":
    main()
