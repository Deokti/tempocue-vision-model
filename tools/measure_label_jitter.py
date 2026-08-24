"""Измерение точности ручной разметки корпуса: где центр значка на самом деле.

Полнота детектора считается по расстоянию между предсказанием и разметкой.
Но разметка ставилась руками, и её собственная погрешность входит в это
расстояние целиком. Пока она не измерена, нельзя сказать, чья ошибка перед
нами — модели или разметки, и любой допуск совпадения выбран наугад.

Способ измерения — тот же, которым сверялась отрисовка (ступень 1): значок
рисуется из circle-арта игры и совмещается с кадром перебором целых сдвигов
по максимуму нормированной кросс-корреляции. Найденное положение считается
истиной, разница с разметкой — погрешностью разметки.

    .venv/Scripts/python tools/measure_label_jitter.py

Грязные вырезы (annotations/corpus-exclusions.json) исключены: у них
совмещение само по себе ненадёжно, и измерять им нечего.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np

from tcvm.cdragon import base_circle_bgra, patch_of
from tcvm.formats import load_corpus
from tcvm.matching import ICON_SIDE, INNER_RADIUS, circular_mask, find_best_match
from tcvm.render import RenderParams, render_icon

DEFAULT_CORPUS = Path(
    r"C:\Users\deokn\.codex\worktrees\739a\PROJECT\tests\TempoCue.Vision.Tests\ReplayCorpus"
)
EXCLUSIONS_PATH = Path(__file__).resolve().parents[1] / "annotations" / "corpus-exclusions.json"
PATCH_VERSION = "16.16.1"

# Наивная отрисовка: подбор параметров здесь не нужен, совмещению хватает
# формы значка — те же параметры, с которых начинает tools/fit_render.py.
NAIVE = RenderParams(1.0, ICON_SIDE, "box", "srgb", 0.0, "bilinear", 0.0, 0.0)
INNER_MASK = circular_mask(ICON_SIDE, INNER_RADIUS)
# Радиус поиска: разметка не может ошибаться на полразмера значка, а больший
# радиус начинает притягивать соседний значок в плотных группах.
ALIGN_RADIUS = 5
# Пороги для сводки: нынешний допуск совпадения детектора и вдвое больший.
REPORT_DISTANCES = (3.0, 5.0)


def load_exclusions() -> set[tuple[str, str]]:
    data = json.loads(EXCLUSIONS_PATH.read_text(encoding="utf-8"))
    return {(e["championId"], e["frame"]) for e in data["exclusions"]}


def describe(values: list[float], unit: str = "px") -> str:
    return (
        f"медиана {statistics.median(values):.2f} {unit}, "
        f"p90 {np.percentile(values, 90):.2f}, максимум {max(values):.2f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    args = parser.parse_args()

    patch = patch_of(PATCH_VERSION)
    excluded = load_exclusions()
    icons: dict[str, np.ndarray] = {}
    offsets: list[float] = []
    per_axis: list[tuple[int, int]] = []
    scores: list[float] = []
    skipped = 0

    for frame in load_corpus(args.corpus):
        for region in frame.labels.champions if frame.labels else ():
            champion = region.champion_id
            if (champion, frame.name) in excluded:
                continue
            if champion not in icons:
                try:
                    icons[champion] = render_icon(base_circle_bgra(champion, patch), NAIVE)
                except (OSError, ValueError) as error:
                    print(f"  нет circle-иконки у {champion}: {error}")
                    icons[champion] = None
            if icons[champion] is None:
                skipped += 1
                continue

            label = (region.x + region.width // 2, region.y + region.height // 2)
            try:
                x, y, score = find_best_match(
                    frame.pixels, icons[champion], INNER_MASK, label, ALIGN_RADIUS
                )
            except ValueError:
                skipped += 1
                continue
            dx, dy = x - label[0], y - label[1]
            offsets.append(float(np.hypot(dx, dy)))
            per_axis.append((dx, dy))
            scores.append(score)

    print(f"Совмещено вырезов: {len(offsets)}, пропущено {skipped}")
    print(f"Качество совмещения (NCC): {describe(scores, '')}")
    print()
    print(f"Смещение разметки от совмещённого центра: {describe(offsets)}")
    for limit in REPORT_DISTANCES:
        share = sum(o > limit for o in offsets) / len(offsets)
        print(f"  доля вырезов дальше {limit:.0f} px: {share:.2f}")
    dxs = [d[0] for d in per_axis]
    dys = [d[1] for d in per_axis]
    print(
        f"  систематический сдвиг: dx {statistics.mean(dxs):+.2f} px, "
        f"dy {statistics.mean(dys):+.2f} px"
    )


if __name__ == "__main__":
    main()
