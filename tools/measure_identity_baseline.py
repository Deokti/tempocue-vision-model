"""Планка для ступени «кто»: сколько даёт пиксельное сравнение без обучения.

Ступень 5 учит сеть отличать чемпионов друг от друга. Прежде чем её строить,
нужно знать, что обученное сравнение вообще должно превзойти — иначе «лучше»
не с чем соотнести. Планка снимается тем же способом, каким личность
определяет нынешний конвейер приложения: вырез сравнивается с эталонами
по маскированной нормированной кросс-корреляции, побеждает наибольший балл.

Задача проще, чем у DeeperLeague, и это надо учитывать честно: состав из
десяти чемпионов известен из API, поэтому выбор идёт **из десяти**, а не из
ста семидесяти. Инструмент меряет оба случая — по составу матча и по всей
библиотеке — чтобы видеть цену незнания состава.

    .venv/Scripts/python tools/measure_identity_baseline.py

Меряется по кадрам корпуса с выправленной разметкой. Печатает долю верных
опознаний и распределение отрыва первого кандидата от второго: именно узость
этого отрыва (полоса 0,71-0,85 в плане) и есть болезнь пиксельного сравнения.
"""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path

import numpy as np

from tcvm.cdragon import base_circle_bgra, patch_of
from tcvm.formats import ReplayFrame, default_corpus_dir, load_corpus
from tcvm.matching import ICON_SIDE, INNER_RADIUS, circular_mask, find_best_match
from tcvm.render import RenderParams, render_icon

DEFAULT_CORPUS = None  # ищется при запуске: см. formats.default_corpus_dir
PATCH_VERSION = "16.16.1"
NAIVE = RenderParams(1.0, ICON_SIDE, "box", "srgb", 0.0, "bilinear", 0.0, 0.0)
INNER_MASK = circular_mask(ICON_SIDE, INNER_RADIUS)
# Подстройка положения при сравнении: конвейер тоже ищет в маленькой окрестности,
# и без неё замер мерил бы точность разметки, а не различение.
ALIGN_RADIUS = 2
# Отрыв, ниже которого решение считается шатким: полоса из плана, где шум
# сопоставим с разницей между чемпионами.
FRAGILE_MARGIN = 0.05


def icon_of(
    champion: str, patch: str, cache: dict[str, np.ndarray | None]
) -> np.ndarray | None:
    if champion not in cache:
        try:
            cache[champion] = render_icon(base_circle_bgra(champion, patch), NAIVE)
        except (OSError, ValueError):
            cache[champion] = None
    return cache[champion]


def rank_candidates(
    frame: ReplayFrame,
    center: tuple[float, float],
    candidates: list[str],
    icons: dict[str, np.ndarray | None],
    patch: str,
) -> list[tuple[float, str]]:
    """Балл каждого кандидата в данной точке, по убыванию."""
    scored: list[tuple[float, str]] = []
    for champion in candidates:
        template = icon_of(champion, patch, icons)
        if template is None:
            continue
        try:
            _, _, score = find_best_match(
                frame.pixels,
                template,
                INNER_MASK,
                (round(center[0]), round(center[1])),
                ALIGN_RADIUS,
            )
        except ValueError:
            continue
        scored.append((score, champion))
    scored.sort(reverse=True)
    return scored


def describe(values: list[float]) -> str:
    return (
        f"медиана {statistics.median(values):.3f}, "
        f"p25 {np.percentile(values, 25):.3f}, минимум {min(values):.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=None)
    args = parser.parse_args()
    corpus = args.corpus or default_corpus_dir()

    patch = patch_of(PATCH_VERSION)
    icons: dict[str, np.ndarray | None] = {}
    library = sorted(
        {
            path.name.split("_circle")[0]
            for path in Path("data/cdragon/16.16/circle").glob("*.png")
        }
    )

    results: dict[str, list[bool]] = {"состав матча": [], "вся библиотека": []}
    margins: list[float] = []
    winners: list[float] = []
    mistakes: list[tuple[str, str, str, float, float]] = []

    for frame in load_corpus(corpus):
        roster = [reference.champion_id for reference in frame.references]
        for region in frame.labels.champions if frame.labels else ():
            center = (region.x + region.width / 2, region.y + region.height / 2)
            truth = region.champion_id

            ranked = rank_candidates(frame, center, roster, icons, patch)
            if not ranked:
                continue
            results["состав матча"].append(ranked[0][1] == truth)
            winners.append(ranked[0][0])
            if len(ranked) > 1:
                margins.append(ranked[0][0] - ranked[1][0])
            if ranked[0][1] != truth:
                found = next((s for s, c in ranked if c == truth), float("nan"))
                mistakes.append((frame.name, truth, ranked[0][1], ranked[0][0], found))

            wide = rank_candidates(frame, center, library, icons, patch)
            if wide:
                results["вся библиотека"].append(wide[0][1].lower() == truth.lower())

    print(f"Чемпионов в замере: {len(results['состав матча'])}")
    print()
    for name, hits in results.items():
        if hits:
            print(
                f"  выбор по «{name}»: верно {sum(hits)} из {len(hits)} — {np.mean(hits):.3f}"
            )
    print()
    print(f"  балл победителя: {describe(winners)}")
    print(f"  отрыв от второго: {describe(margins)}")
    fragile = sum(margin < FRAGILE_MARGIN for margin in margins) / len(margins)
    print(f"  доля решений с отрывом меньше {FRAGILE_MARGIN}: {fragile:.2f}")
    print()
    print("  ошибки опознания (по составу матча):")
    for frame_name, truth, taken, taken_score, truth_score in mistakes:
        print(
            f"    {frame_name[:30]:30} {truth:12} принят за {taken:12} "
            f"{taken_score:.3f} против {truth_score:.3f}"
        )


if __name__ == "__main__":
    main()
