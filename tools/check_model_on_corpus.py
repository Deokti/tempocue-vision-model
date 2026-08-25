"""Вся модель целиком по настоящим кадрам: детектор плюс опознание.

    .venv/Scripts/python tools/check_model_on_corpus.py \\
        --detector out/detector-02/best.pt --identity out/identity-01/best.pt

До сих пор ступени мерились по отдельности: детектор — по центрам при готовой
разметке, опознание — по вырезам при готовых центрах. Здесь они соединены так,
как будут работать в приложении: кадр → центры → вырезы → имена. Это первое
число, сравнимое с конвейером приложения (45 чемпионов из 60 при нуле ложных).

Чемпион считается **взятым**, если модель нашла его центр не дальше допуска и
назвала верное имя. Всё остальное — либо пропуск, либо ложное срабатывание;
ложным считается срабатывание, получившее имя чемпиона вдали от любого
размеченного.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from tcvm.cdragon import base_circle_bgra, patch_of
from tcvm.detector import MATCH_DISTANCE, CenterDetector, decode_heatmap
from tcvm.formats import ReplayFrame, bgra_to_rgb, default_corpus_dir, load_corpus
from tcvm.identity import (
    CANONICAL_CROP,
    NO_CHAMPION,
    IdentityNet,
    as_input,
    check_setup,
    choose,
)
from tcvm.matching import ICON_SIDE, INNER_RADIUS, circular_mask, find_best_match
from tcvm.render import RenderParams, render_icon

DEFAULT_CORPUS = None  # ищется при запуске: см. formats.default_corpus_dir
# Числа конвейера приложения по тому же корпусу (docs/own-model-plan.md):
# 45 чемпионов из 60 при нуле ложных и 58 мс на кадр.
PIPELINE_TAKEN = 45
PIPELINE_TOTAL = 60
PIPELINE_FALSE = 0
# Один чемпион не может стоять на карте дважды: состав матча — десять разных
# героев. Значит два срабатывания с одним именем — это один чемпион, увиденный
# дважды, и лишнее надо убрать. Это не уловка, а правило самой игры.
ONE_PER_CHAMPION = True

# Уточнение положения по опознанному имени. Детектор отвечает «здесь кто-то
# есть» и в плотных группах мажет мимо центра на 3-12 px; но как только имя
# известно, положение можно досчитать совмещением значка именно этого чемпиона.
# Это не новая ступень, а завершение прежней: то же совмещение, каким конвейер
# приложения уже пользуется, только теперь ему подсказано, что искать.
REFINE_RADIUS = 8
PATCH_VERSION = "16.16.1"
NAIVE = RenderParams(1.0, ICON_SIDE, "box", "srgb", 0.0, "bilinear", 0.0, 0.0)
INNER_MASK = circular_mask(ICON_SIDE, INNER_RADIUS)


@dataclass
class Tally:
    taken: int = 0
    labelled: int = 0
    named_wrong: int = 0
    missed_place: int = 0
    false_alarms: int = 0
    per_frame: list[tuple[str, int, int, int]] = field(default_factory=list)


def frame_to_tensor(pixels_bgra: np.ndarray) -> torch.Tensor:
    rgb = bgra_to_rgb(pixels_bgra).astype(np.float32) / 255.0
    return torch.from_numpy(np.transpose(rgb, (2, 0, 1))).unsqueeze(0)


def crop_at(pixels: np.ndarray, cx: float, cy: float) -> np.ndarray | None:
    half = CANONICAL_CROP // 2
    left, top = round(cx) - half, round(cy) - half
    if left < 0 or top < 0:
        return None
    if left + CANONICAL_CROP > pixels.shape[1] or top + CANONICAL_CROP > pixels.shape[0]:
        return None
    return pixels[top : top + CANONICAL_CROP, left : left + CANONICAL_CROP]


def refine(
    frame: ReplayFrame,
    x: float,
    y: float,
    champion: str,
    icons: dict[str, np.ndarray | None],
) -> tuple[float, float]:
    """Точный центр значка названного чемпиона рядом с приблизительной точкой."""
    if champion not in icons:
        try:
            icons[champion] = render_icon(
                base_circle_bgra(champion, patch_of(PATCH_VERSION)), NAIVE
            )
        except (OSError, ValueError):
            icons[champion] = None
    template = icons[champion]
    if template is None:
        return x, y
    try:
        found_x, found_y, _ = find_best_match(
            frame.pixels, template, INNER_MASK, (round(x), round(y)), REFINE_RADIUS
        )
    except ValueError:
        return x, y
    return float(found_x), float(found_y)


def name_detections(
    pixels: np.ndarray,
    found: list[tuple[float, float, float]],
    identity: IdentityNet,
    vocabulary: list[str],
    roster: list[str],
) -> list[tuple[float, float, str]]:
    """Каждому найденному центру — имя чемпиона либо отказ.

    Если одно имя досталось нескольким срабатываниям, остаётся самое уверенное:
    чемпион не может стоять на карте дважды. Остальные не отбрасываются, а
    превращаются в отказ — они всё ещё где-то что-то нашли, просто это не
    отдельный герой.
    """
    scored: list[tuple[float, float, str, float]] = []
    for x, y, _ in found:
        piece = crop_at(pixels, x, y)
        if piece is None:
            continue
        with torch.no_grad():
            logits = identity(as_input(torch.from_numpy(piece[None].copy())))
        name = choose(logits, vocabulary, roster)[0]
        confidence = float(torch.softmax(logits, dim=1).max())
        scored.append((x, y, name, confidence))

    if not ONE_PER_CHAMPION:
        return [(x, y, name) for x, y, name, _ in scored]

    best_for: dict[str, int] = {}
    for index, (_, _, name, confidence) in enumerate(scored):
        if name == NO_CHAMPION:
            continue
        current = best_for.get(name)
        if current is None or confidence > scored[current][3]:
            best_for[name] = index
    keep = set(best_for.values())
    return [
        (x, y, name if index in keep or name == NO_CHAMPION else NO_CHAMPION)
        for index, (x, y, name, _) in enumerate(scored)
    ]


def score_frame(
    frame: ReplayFrame,
    named: list[tuple[float, float, str]],
    tally: Tally,
    tolerance: float = MATCH_DISTANCE,
) -> None:
    """Сводит найденное с разметкой: взятые, перепутанные, пропущенные, ложные."""
    truth = [
        (region.x + region.width / 2, region.y + region.height / 2, region.champion_id)
        for region in (frame.labels.champions if frame.labels else ())
    ]
    free = list(range(len(truth)))
    taken = wrong = false = 0

    for x, y, name in named:
        distances = [(np.hypot(x - truth[i][0], y - truth[i][1]), i) for i in free]
        best = min(distances) if distances else (999.0, -1)
        if best[0] <= tolerance:
            free.remove(best[1])
            if name.lower() == truth[best[1]][2].lower():
                taken += 1
            else:
                wrong += 1
        elif name != NO_CHAMPION:
            false += 1

    tally.taken += taken
    tally.named_wrong += wrong
    tally.labelled += len(truth)
    tally.missed_place += len(free)
    tally.false_alarms += false
    tally.per_frame.append((frame.name, taken, len(truth), false))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detector", type=Path, required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--vocabulary", type=Path, default=None)
    parser.add_argument("--corpus", type=Path, default=None)
    parser.add_argument(
        "--no-refine", action="store_true", help="не уточнять положение по опознанному имени"
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=MATCH_DISTANCE,
        help="допуск совпадения центра в пикселях",
    )
    args = parser.parse_args()
    corpus = args.corpus or default_corpus_dir()

    vocabulary_path = args.vocabulary or args.identity.parent / "vocabulary.json"
    vocabulary = json.loads(vocabulary_path.read_text(encoding="utf-8"))

    detector = CenterDetector()
    detector.load_state_dict(torch.load(args.detector, map_location="cpu"))
    detector.eval()
    check_setup(args.weights if hasattr(args, "weights") else args.identity)
    identity = IdentityNet(len(vocabulary))
    identity.load_state_dict(torch.load(args.identity, map_location="cpu"))
    identity.eval()
    print(f"Детектор: {args.detector}")
    print(f"Опознание: {args.identity} ({len(vocabulary)} классов)")
    print()

    tally = Tally()
    icons: dict[str, np.ndarray | None] = {}
    print("Уточнение положения по имени: " + ("выключено" if args.no_refine else "включено"))
    print()
    for frame in load_corpus(corpus):
        pixels = bgra_to_rgb(frame.pixels)
        with torch.no_grad():
            found = decode_heatmap(detector(frame_to_tensor(frame.pixels)))[0]
        roster = [reference.champion_id for reference in frame.references]
        named = name_detections(pixels, found, identity, vocabulary, roster)
        if not args.no_refine:
            named = [
                (*refine(frame, x, y, name, icons), name) if name else (x, y, name)
                for x, y, name in named
            ]
        score_frame(frame, named, tally, args.tolerance)

    print("  кадр                                      взято  из  ложных")
    for name, taken, total, false in tally.per_frame:
        print(f"  {name:40} {taken:5}  {total:3}  {false:6}")

    share = tally.taken / tally.labelled if tally.labelled else 0.0
    print()
    print(f"Взято чемпионов: {tally.taken} из {tally.labelled} — {share:.3f}")
    print(f"  найдено, но названо неверно: {tally.named_wrong}")
    print(f"  не найдено вовсе: {tally.missed_place}")
    print(f"  ложных срабатываний с именем: {tally.false_alarms}")
    print()
    print(
        f"Конвейер приложения по тому же корпусу: {PIPELINE_TAKEN} из {PIPELINE_TOTAL} "
        f"({PIPELINE_TAKEN / PIPELINE_TOTAL:.3f}) при {PIPELINE_FALSE} ложных"
    )
    print(
        "Сравнение неточное: у конвейера числа сняты по прежней разметке, "
        "где чемпионов было 60, а не 66."
    )


if __name__ == "__main__":
    main()
