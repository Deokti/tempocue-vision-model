"""Сколько деталей несёт кадр: какого размера была миникарта до приведения.

Приложение приводит миникарту любого размера к каноническим 320×320, и родной
размер в файле не сохраняется. Но он оставляет след: растянутый с 200 кадр
физически не содержит мелких деталей, а ужатый с 600 содержит их с избытком.
След читается по спектру — доле энергии в верхней трети частот.

    .venv/Scripts/python tools/measure_frame_detail.py --corpus replay-inbox

Зачем это нужно. Резкость значка — вероятно самая важная ось из всех: от неё
прямо зависит, сколько информации доступно опознанию. Проверить, покрывает ли
корпус весь диапазон, иначе нечем.

Оговорка о пределах способа. Величина зависит не только от размера, но и от
содержимого кадра — тумана, числа значков, белых линий интерфейса. Поэтому она
годится для сравнения матчей между собой и для отсева сильно растянутых
кадров, но не для точного восстановления родного размера в пикселях.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from tcvm.formats import bgra_to_rgb, default_corpus_dir, load_corpus
from tcvm.synthesis import MAP_PLACEMENT, map_rect

# Считаем только по области карты: рамка интерфейса и её резкие белые линии
# дают высокие частоты независимо от размера миникарты и смазали бы замер.
CANONICAL_FRAME = 320
# Граница «высоких» частот в долях от предельной различимой. Выше неё лежат
# детали, которых у растянутого кадра быть не может.
HIGH_BAND = 0.62


def detail_share(pixels_bgra: np.ndarray) -> float:
    """Доля энергии в верхней трети частот области карты."""
    left, top, right, bottom = map_rect(CANONICAL_FRAME, MAP_PLACEMENT)
    grey = bgra_to_rgb(pixels_bgra).astype(np.float64)[top:bottom, left:right].mean(axis=2)
    window = np.hanning(right - left)[None, :] * np.hanning(bottom - top)[:, None]
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2((grey - grey.mean()) * window))) ** 2

    side = spectrum.shape[0]
    rows, columns = np.mgrid[0:side, 0:side] - side / 2
    radius = np.hypot(rows, columns) / (side / 2)
    inside = radius <= 1.0
    total = spectrum[inside].sum()
    return float(spectrum[inside & (radius > HIGH_BAND)].sum() / total) if total else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=None)
    args = parser.parse_args()
    corpus = args.corpus or default_corpus_dir()

    groups: dict[tuple[str, ...], list[float]] = {}
    for frame in load_corpus(corpus):
        key = tuple(sorted(reference.champion_id for reference in frame.references))
        groups.setdefault(key, []).append(detail_share(frame.pixels))

    print(f"Кадров: {sum(len(v) for v in groups.values())}, матчей: {len(groups)}")
    print()
    print("  матч                     кадров  детали  разброс")
    for index, (roster, values) in enumerate(
        sorted(groups.items(), key=lambda item: -len(item[1])), 1
    ):
        share = np.array(values)
        name = roster[0] if roster else "?"
        print(f"  {index:2}. {name:20} {len(share):5}  {share.mean():.4f}  {share.std():.4f}")

    everything = np.concatenate([np.array(v) for v in groups.values()])
    print()
    print(f"  по всем кадрам: {everything.min():.4f}..{everything.max():.4f}")
    print("  для сравнения, синтетика, приведённая к 320: родные 200 дают ~0,007,")
    print("  родные 320 ~0,035, родные 600 ~0,015.")


if __name__ == "__main__":
    main()
