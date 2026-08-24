"""Проверка обученного опознания на настоящих кадрах по записанному критерию.

    .venv/Scripts/python tools/check_identity_on_corpus.py --weights out/identity/best.pt

Критерий записан в `docs/identity.md` **до** постройки сети, и инструмент
проверяет ровно его, а не то, что получилось:

1. превзойти планку пиксельного сравнения `0,788` при выборе из состава матча;
2. не потерять ни одного из тех, кого пиксельное сравнение берёт уверенно
   (отрыв от второго кандидата больше `0,2`);
3. уметь отказывать: вырез без чемпиона не должен получать имя.

Положения берутся из выправленной разметки корпуса — детектор в проверку не
входит, мерится чистое различение. Оговорка та же, что у детектора: двенадцать
кадров корпуса разбирались при разработке, итоговая приёмка — по слепому
набору инструментами приложения.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from tcvm.cdragon import base_circle_bgra, patch_of
from tcvm.formats import ReplayFrame, bgra_to_rgb, load_corpus
from tcvm.identity import (
    CANONICAL_CROP,
    NO_CHAMPION,
    IdentityNet,
    as_input,
    choose,
)
from tcvm.matching import ICON_SIDE, INNER_RADIUS, circular_mask, find_best_match
from tcvm.render import RenderParams, render_icon

DEFAULT_CORPUS = Path(
    r"C:\Users\deokn\.codex\worktrees\739a\PROJECT\tests\TempoCue.Vision.Tests\ReplayCorpus"
)
PATCH_VERSION = "16.16.1"
NAIVE = RenderParams(1.0, ICON_SIDE, "box", "srgb", 0.0, "bilinear", 0.0, 0.0)
INNER_MASK = circular_mask(ICON_SIDE, INNER_RADIUS)
ALIGN_RADIUS = 2
# Планка из docs/identity.md: пиксельное сравнение при выборе из состава матча.
BASELINE = 0.788
# Отрыв, выше которого случай считается лёгким для пиксельного сравнения.
EASY_MARGIN = 0.2
# Отступ от чемпионов для отрицательных вырезов на настоящих кадрах.
NEGATIVE_CLEARANCE = 20.0
NEGATIVES_PER_FRAME = 8


@dataclass
class Tally:
    """Копилка итогов проверки; поля читаются по имени, а не по индексу."""

    right: int = 0
    total: int = 0
    easy_right: int = 0
    easy_total: int = 0
    rejects_right: int = 0
    rejects_total: int = 0
    mistakes: list[tuple[str, str, str, float]] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return self.right / self.total if self.total else 0.0

    @property
    def easy(self) -> float:
        return self.easy_right / self.easy_total if self.easy_total else 0.0

    @property
    def rejects(self) -> float:
        return self.rejects_right / self.rejects_total if self.rejects_total else 0.0


def crop_at(pixels: np.ndarray, cx: float, cy: float) -> np.ndarray | None:
    half = CANONICAL_CROP // 2
    left, top = round(cx) - half, round(cy) - half
    if (
        left < 0
        or top < 0
        or left + CANONICAL_CROP > pixels.shape[1]
        or top + CANONICAL_CROP > pixels.shape[0]
    ):
        return None
    return pixels[top : top + CANONICAL_CROP, left : left + CANONICAL_CROP]


def baseline_margin(
    frame: ReplayFrame, center: tuple[float, float], roster: list[str], patch: str, cache: dict
) -> tuple[str, float]:
    """Кого выбирает пиксельное сравнение и с каким отрывом от второго."""
    scored = []
    for champion in roster:
        if champion not in cache:
            try:
                cache[champion] = render_icon(base_circle_bgra(champion, patch), NAIVE)
            except (OSError, ValueError):
                cache[champion] = None
        template = cache[champion]
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
    if not scored:
        return "", 0.0
    margin = scored[0][0] - scored[1][0] if len(scored) > 1 else scored[0][0]
    return scored[0][1], margin


def negative_centers(
    frame: ReplayFrame, taken: list[tuple[float, float]], rng: np.random.Generator
) -> list[tuple[int, int]]:
    """Точки настоящего кадра, где чемпиона заведомо нет."""
    spots: list[tuple[int, int]] = []
    for _ in range(NEGATIVES_PER_FRAME * 8):
        if len(spots) >= NEGATIVES_PER_FRAME:
            break
        x = int(rng.integers(CANONICAL_CROP, frame.pixels.shape[1] - CANONICAL_CROP))
        y = int(rng.integers(CANONICAL_CROP, frame.pixels.shape[0] - CANONICAL_CROP))
        if any(np.hypot(x - tx, y - ty) < NEGATIVE_CLEARANCE for tx, ty in taken):
            continue
        spots.append((x, y))
    return spots


def examine(
    frame: ReplayFrame,
    model: IdentityNet,
    vocabulary: list[str],
    context: tuple[str, dict, np.random.Generator],
    tally: Tally,
) -> None:
    """Прогоняет один кадр: размеченных чемпионов и отрицательные вырезы."""
    patch, icons, rng = context
    pixels = bgra_to_rgb(frame.pixels)
    roster = [reference.champion_id for reference in frame.references]
    centers: list[tuple[float, float]] = []

    def decide(piece: np.ndarray) -> str:
        with torch.no_grad():
            logits = model(as_input(torch.from_numpy(piece[None].copy())))
        return choose(logits, vocabulary, roster)[0]

    for region in frame.labels.champions if frame.labels else ():
        center = (region.x + region.width / 2, region.y + region.height / 2)
        centers.append(center)
        piece = crop_at(pixels, *center)
        if piece is None:
            continue
        taken = decide(piece)
        correct = taken.lower() == region.champion_id.lower()
        tally.total += 1
        tally.right += correct
        _, margin = baseline_margin(frame, center, roster, patch, icons)
        if margin > EASY_MARGIN:
            tally.easy_total += 1
            tally.easy_right += correct
        if not correct:
            tally.mistakes.append((frame.name, region.champion_id, taken or "отказ", margin))

    for x, y in negative_centers(frame, centers, rng):
        piece = crop_at(pixels, x, y)
        if piece is None:
            continue
        tally.rejects_total += 1
        tally.rejects_right += decide(piece) == NO_CHAMPION


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--vocabulary", type=Path, default=None)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    vocabulary_path = args.vocabulary or args.weights.parent / "vocabulary.json"
    vocabulary = json.loads(vocabulary_path.read_text(encoding="utf-8"))
    model = IdentityNet(len(vocabulary))
    model.load_state_dict(torch.load(args.weights, map_location="cpu"))
    model.eval()
    print(f"Веса: {args.weights}; классов {len(vocabulary)}")

    patch = patch_of(PATCH_VERSION)
    icons: dict[str, np.ndarray | None] = {}
    rng = np.random.default_rng(args.seed)

    tally = Tally()

    for frame in load_corpus(args.corpus):
        examine(frame, model, vocabulary, (patch, icons, rng), tally)

    print()
    print(
        f"  1. опознано верно: {tally.right} из {tally.total} — "
        f"{tally.accuracy:.3f} (планка {BASELINE})"
    )
    print(
        f"  2. на лёгких случаях: {tally.easy_right} из {tally.easy_total} — {tally.easy:.3f}"
    )
    print(f"  3. отказы: {tally.rejects_right} из {tally.rejects_total} — {tally.rejects:.3f}")
    print()
    verdicts = {
        "превзойти планку": tally.accuracy > BASELINE,
        "не терять лёгкие": tally.easy >= 1.0,
        "уметь отказывать": tally.rejects >= 1.0,
    }
    for name, passed in verdicts.items():
        print(f"  {'ДА ' if passed else 'НЕТ'}  {name}")
    print()
    print(
        f"Вердикт: {'критерий выполнен' if all(verdicts.values()) else 'критерий не выполнен'}"
    )
    if tally.mistakes:
        print()
        print("  ошибки опознания (отрыв — насколько лёгок случай для пикселей):")
        for frame_name, truth, taken, margin in tally.mistakes:
            print(
                f"    {frame_name[:30]:30} {truth:12} принят за {taken:12} отрыв {margin:.3f}"
            )


if __name__ == "__main__":
    main()
