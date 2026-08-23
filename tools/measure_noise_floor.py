"""Замер шумового пола: насколько значок отличается сам от себя между кадрами.

Этапы 2-3 сверки отрисовки (docs/render-verification.md). Два источника:

- записи ReplaySequences: значок ведётся от кадра к кадру локальным поиском
  от ручной стартовой позиции (annotations/sequence-seeds.json); расхождение
  считается между вырезами соседних кадров. Это основной шумовой пол;
- повторы в корпусе: один чемпион, размеченный в нескольких кадрах одного
  матча (матч определяется составом эталонов). Центры ручных рамок в разных
  кадрах гуляют на несколько пикселей, поэтому перед сравнением пара
  выравнивается поиском лучшего совмещения в окне +-3 px — сверка отрисовки
  будет подбирать сдвиг точно так же.

Расхождение = 1 - NCC, отдельно по внутреннему диску (радиус 10) и по
центральному квадрату 11x11. Запуск из корня репозитория:

    .venv/Scripts/python tools/measure_noise_floor.py

Картинки для проверки глазами кладутся в out/noise-floor/.
"""

from __future__ import annotations

import argparse
import json
import statistics
from itertools import pairwise
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from tcvm.formats import ReplayFrame, bgra_to_rgb, load_corpus, load_sequences
from tcvm.matching import (
    CENTER_SIDE,
    ICON_SIDE,
    INNER_RADIUS,
    center_square_mask,
    circular_mask,
    crop_centered,
    find_best_match,
    masked_ncc,
)

DEFAULT_DATA_ROOT = Path(
    r"C:\Users\deokn\.codex\worktrees\739a\PROJECT\tests\TempoCue.Vision.Tests"
)
SEEDS_PATH = Path(__file__).resolve().parents[1] / "annotations" / "sequence-seeds.json"
SEARCH_RADIUS = 6
ALIGN_RADIUS = 3
UPSCALE = 8

INNER_MASK = circular_mask(ICON_SIDE, INNER_RADIUS)
CENTER_MASK = center_square_mask(ICON_SIDE, CENTER_SIDE)


def upscaled(bgra: np.ndarray) -> Image.Image:
    image = Image.fromarray(bgra_to_rgb(bgra), "RGB")
    return image.resize((image.width * UPSCALE, image.height * UPSCALE), Image.NEAREST)


def describe(values: list[float]) -> str:
    return (
        f"медиана {statistics.median(values):.4f}, "
        f"p95 {np.percentile(values, 95):.4f}, максимум {max(values):.4f}"
    )


def track_sequences(data_root: Path, out_root: Path) -> dict[str, list[float]]:
    """Ведёт размеченные значки по записям; возвращает расхождения по группам."""
    seeds = json.loads(SEEDS_PATH.read_text(encoding="utf-8"))["sequences"]
    grouped: dict[str, list[float]] = {
        "static-inner": [],
        "static-center": [],
        "moving-inner": [],
        "moving-center": [],
    }

    for sequence in load_sequences(data_root / "ReplaySequences"):
        if sequence.name not in seeds:
            continue
        out = out_root / sequence.name
        out.mkdir(parents=True, exist_ok=True)
        overlay = Image.fromarray(bgra_to_rgb(sequence.frames[0]), "RGB").resize(
            (640, 640), Image.NEAREST
        )
        draw = ImageDraw.Draw(overlay)

        for seed in seeds[sequence.name]:
            template = crop_centered(sequence.frames[0], seed["x"], seed["y"])
            positions = [(seed["x"], seed["y"], 1.0)]
            crops = [template]
            for frame in sequence.frames[1:]:
                px, py, _ = positions[-1]
                x, y, score = find_best_match(
                    frame, template, INNER_MASK, (px, py), SEARCH_RADIUS
                )
                positions.append((x, y, score))
                crops.append(crop_centered(frame, x, y))

            kind = "moving" if seed["moving"] else "static"
            inner, center = [], []
            for previous, current in pairwise(crops):
                inner.append(1.0 - masked_ncc(previous, current, INNER_MASK))
                center.append(1.0 - masked_ncc(previous, current, CENTER_MASK))
            grouped[f"{kind}-inner"].extend(inner)
            grouped[f"{kind}-center"].extend(center)

            travel = max(abs(x - seed["x"]) + abs(y - seed["y"]) for x, y, _ in positions)
            worst_track = min(score for _, _, score in positions[1:])
            print(
                f"  {sequence.name} / {seed['championId']} ({kind}): "
                f"смещение до {travel} px, худший балл трекинга {worst_track:.3f}, "
                f"диск: {describe(inner)}"
            )

            strip = Image.new("RGB", (ICON_SIDE * UPSCALE * len(crops), ICON_SIDE * UPSCALE))
            for column, patch in enumerate(crops):
                strip.paste(upscaled(patch), (column * ICON_SIDE * UPSCALE, 0))
            strip.save(out / f"{seed['championId']}-track.png")
            draw.rectangle(
                [
                    (seed["x"] - 12) * 2,
                    (seed["y"] - 12) * 2,
                    (seed["x"] + 12) * 2,
                    (seed["y"] + 12) * 2,
                ],
                outline=(0, 255, 70),
                width=2,
            )
            draw.text(
                ((seed["x"] - 12) * 2, (seed["y"] - 12) * 2 - 12),
                seed["championId"],
                fill=(0, 255, 70),
            )
        overlay.save(out / "frame000-seeds.png")

    return grouped


def roster_signature(frame: ReplayFrame) -> tuple:
    return tuple(sorted((r.champion_id, r.affiliation) for r in frame.references))


def corpus_repeats(data_root: Path) -> list[float]:
    """Попарные расхождения одного чемпиона между кадрами одного матча.

    Пара сначала выравнивается: вырез второго кадра ищется в окне +-3 px
    вокруг центра его рамки так, чтобы совпадение с первым было максимально.
    Без этого измеряется не отрисовка, а разброс центров ручной разметки.
    """
    frames = load_corpus(data_root / "ReplayCorpus")
    by_match: dict[tuple, list[ReplayFrame]] = {}
    for frame in frames:
        by_match.setdefault(roster_signature(frame), []).append(frame)

    values = []
    print(f"  матчей в корпусе по составам эталонов: {len(by_match)}")
    for match_frames in by_match.values():
        sites: dict[str, list[tuple[ReplayFrame, int, int]]] = {}
        for frame in match_frames:
            for region in frame.labels.champions if frame.labels else ():
                cx = region.x + region.width // 2
                cy = region.y + region.height // 2
                sites.setdefault(region.champion_id, []).append((frame, cx, cy))
        for champion, items in sites.items():
            for i in range(len(items)):
                for j in range(i + 1, len(items)):
                    frame_a, ax, ay = items[i]
                    frame_b, bx, by = items[j]
                    try:
                        crop_a = crop_centered(frame_a.pixels, ax, ay)
                        _, _, score = find_best_match(
                            frame_b.pixels, crop_a, INNER_MASK, (bx, by), ALIGN_RADIUS
                        )
                    except ValueError:
                        continue  # значок у самого края кадра
                    value = 1.0 - score
                    values.append(value)
                    print(f"    {champion}: {value:.4f} ({frame_a.name} против {frame_b.name})")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out", type=Path, default=Path("out/noise-floor"))
    args = parser.parse_args()

    print("Записи (соседние кадры, один и тот же значок):")
    grouped = track_sequences(args.data, args.out)
    print()
    for name, values in grouped.items():
        if values:
            print(f"  {name}: n={len(values)}, {describe(values)}")

    print()
    print("Корпус (пары кадров одного матча, после выравнивания +-3 px):")
    repeats = corpus_repeats(args.data)
    if repeats:
        print(f"  пар: {len(repeats)}, диск: {describe(repeats)}")
    print()
    print(f"Картинки: {args.out.resolve()}")


if __name__ == "__main__":
    main()
