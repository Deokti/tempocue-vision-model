"""Весь конвейер на одном ONNX, без PyTorch: образец для стороны C#.

    .venv/Scripts/python tools/run_onnx.py --model out/onnx --corpus <корпус>

Зачем отдельный инструмент, когда есть `check_model_on_corpus.py`. Тот считает
через PyTorch и через библиотеку `tcvm`; в приложении не будет ни того, ни
другого. Здесь то же самое собрано из одного ONNX и numpy — ровно те действия,
которые предстоит повторить на C#, и ни одного лишнего.

Смысл в том, чтобы порт был **проверяемым**. Числа этого инструмента — цель,
в которую распознаватель приложения обязан попасть кадр в кадр. Не попал —
ошибка в порте, а не в модели, и искать её надо там.

Уточнение положения по арту сюда не входит намеренно: в приложении эту работу
делает уже существующее совмещение значков, переносить её незачем. Поэтому
числа сравниваются с `check_model_on_corpus.py --no-refine`.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime

from tcvm.formats import default_corpus_dir, load_corpus

# Провайдер вычислений: приёмка идёт на процессоре, как и в приложении.
PROVIDERS = ["CPUExecutionProvider"]


@dataclass(frozen=True)
class Model:
    """Две сети и описание из `model.json` — всё, что отдаёт репозиторий."""

    setup: dict
    detector: onnxruntime.InferenceSession
    identity: onnxruntime.InferenceSession
    detector_input: str
    identity_input: str

    @property
    def vocabulary(self) -> list[str]:
        return self.setup["vocabulary"]

    @classmethod
    def load(cls, directory: Path) -> Model:
        setup = json.loads((directory / "model.json").read_text(encoding="utf-8"))
        detector = onnxruntime.InferenceSession(
            str(directory / setup["detector"]), providers=PROVIDERS
        )
        identity = onnxruntime.InferenceSession(
            str(directory / setup["identity"]), providers=PROVIDERS
        )
        return cls(
            setup=setup,
            detector=detector,
            identity=identity,
            detector_input=detector.get_inputs()[0].name,
            identity_input=identity.get_inputs()[0].name,
        )


# Окно подавления соседей: клетка остаётся вершиной, только если равна
# максимуму окрестности 3x3. Так два близких значка дают две вершины.
NMS_KERNEL = 3
# Плоскую вершину параболой не уточнить: знаменатель обращается в ноль.
FLAT_PEAK_EPSILON = 1e-6


def parabolic_offset(plane: np.ndarray, gy: int, gx: int, axis: int) -> float:
    """Субпиксельная поправка к клетке по вершине параболы через три значения.

    Клетка карты центров — четыре канонических пикселя, поэтому округление до
    клетки само по себе даёт ошибку до 2 px. Соседние значения бугра несут долю
    пикселя: вершина параболы через них её восстанавливает.
    """
    height, width = plane.shape
    if axis == 0:
        if gy in (0, height - 1):
            return 0.0
        before, center, after = plane[gy - 1, gx], plane[gy, gx], plane[gy + 1, gx]
    else:
        if gx in (0, width - 1):
            return 0.0
        before, center, after = plane[gy, gx - 1], plane[gy, gx], plane[gy, gx + 1]

    denominator = float(before) - 2 * float(center) + float(after)
    if abs(denominator) < FLAT_PEAK_EPSILON:
        return 0.0
    offset = 0.5 * (float(before) - float(after)) / denominator
    # Вершина дальше половины клетки означала бы, что максимум не здесь.
    return max(-0.5, min(0.5, offset))


def local_maxima(plane: np.ndarray) -> np.ndarray:
    """Маска клеток, равных максимуму своей окрестности NMS_KERNEL.

    В PyTorch это max_pool2d с шагом 1. Здесь то же самое сдвигами: numpy без
    свёрточных слоёв, и на C# это будет такой же двойной цикл.
    """
    pad = NMS_KERNEL // 2
    padded = np.pad(plane, pad, mode="constant", constant_values=-np.inf)
    pooled = np.full_like(plane, -np.inf)
    for dy in range(NMS_KERNEL):
        for dx in range(NMS_KERNEL):
            window = padded[dy : dy + plane.shape[0], dx : dx + plane.shape[1]]
            pooled = np.maximum(pooled, window)
    return plane >= pooled


def decode_centers(
    probabilities: np.ndarray, threshold: float, stride: int
) -> list[tuple[float, float, float]]:
    """Карта центров → список (x, y, уверенность) в канонических пикселях."""
    plane = probabilities[0, 0]
    peaks = local_maxima(plane) & (plane >= threshold)
    found = []
    for gy, gx in zip(*np.nonzero(peaks), strict=True):
        found.append(
            (
                (gx + parabolic_offset(plane, gy, gx, axis=1)) * stride,
                (gy + parabolic_offset(plane, gy, gx, axis=0)) * stride,
                float(plane[gy, gx]),
            )
        )
    return sorted(found, key=lambda item: -item[2])


def crop_at(pixels_bgra: np.ndarray, cx: float, cy: float, side: int) -> np.ndarray | None:
    """Вырез стороной side вокруг точки; None, если вылезает за кадр."""
    half = side // 2
    left, top = round(cx) - half, round(cy) - half
    if (
        left < 0
        or top < 0
        or left + side > pixels_bgra.shape[1]
        or top + side > pixels_bgra.shape[0]
    ):
        return None
    return pixels_bgra[top : top + side, left : left + side]


def name_crops(
    scores: np.ndarray, vocabulary: list[str], roster: list[str], reject_index: int
) -> list[tuple[str, float]]:
    """Баллы по классам → имя и уверенность, выбор только из состава матча.

    Прочие классы не подавляются, а просто не рассматриваются: это ровно то,
    что даёт знание состава из API.
    """
    wanted = {name.lower() for name in roster}
    allowed = [
        index
        for index, name in enumerate(vocabulary)
        if index == reject_index or name.lower() in wanted
    ]
    chosen = []
    for row in scores:
        best = max(allowed, key=lambda index: row[index])
        chosen.append((vocabulary[best], float(row[best])))
    return chosen


def keep_one_per_champion(
    named: list[tuple[float, float, str, float]], reject: str
) -> list[tuple[float, float, str]]:
    """Одно имя — один чемпион: лишние срабатывания становятся отказом.

    Чемпион не может стоять на карте дважды, состав матча — десять разных
    героев. Это правило самой игры, а не уловка ради чисел.
    """
    best_for: dict[str, int] = {}
    for index, (_, _, name, confidence) in enumerate(named):
        if name == reject:
            continue
        current = best_for.get(name)
        if current is None or confidence > named[current][3]:
            best_for[name] = index
    keep = set(best_for.values())
    return [
        (x, y, name if index in keep or name == reject else reject)
        for index, (x, y, name, _) in enumerate(named)
    ]


@dataclass
class Tally:
    """Копилка итогов: те же четыре графы, что и у сводки на PyTorch."""

    taken: int = 0
    wrong: int = 0
    missed: int = 0
    false_alarms: int = 0
    labelled: int = 0


def recognise(
    frame_bgra: np.ndarray, roster: list[str], model: Model, threshold: float
) -> list[tuple[float, float, str]]:
    """Кадр и состав матча → чемпионы с координатами. Весь разбор целиком."""
    probabilities = model.detector.run(None, {model.detector_input: frame_bgra[None]})[0]
    crops, places = [], []
    for x, y, _ in decode_centers(probabilities, threshold, model.setup["stride"]):
        piece = crop_at(frame_bgra, x, y, model.setup["cropSide"])
        if piece is not None:
            crops.append(piece)
            places.append((x, y))
    if not crops:
        return []

    scores = model.identity.run(None, {model.identity_input: np.stack(crops)})[0]
    chosen = name_crops(scores, model.vocabulary, roster, model.setup["rejectIndex"])
    named = [(x, y, name, share) for (x, y), (name, share) in zip(places, chosen, strict=True)]
    return keep_one_per_champion(named, model.setup["rejectName"])


def score_frame(
    decided: list[tuple[float, float, str]],
    truth: list[tuple[float, float, str]],
    reject: str,
    tolerance: float,
    tally: Tally,
) -> tuple[int, list[str]]:
    """Сводит найденное с разметкой. Отказ в сопоставлении не участвует.

    Возвращает взятых и поимённо непойманных: сверять порт по суммам мало,
    расходиться он может при одинаковом итоге.
    """
    before = tally.taken
    free = list(range(len(truth)))
    for x, y, name in decided:
        if name == reject:
            continue
        distances = [(float(np.hypot(x - truth[i][0], y - truth[i][1])), i) for i in free]
        best = min(distances) if distances else (999.0, -1)
        if best[0] > tolerance:
            tally.false_alarms += 1
            continue
        free.remove(best[1])
        if name.lower() == truth[best[1]][2].lower():
            tally.taken += 1
        else:
            tally.wrong += 1
    tally.labelled += len(truth)
    tally.missed += len(free)
    return tally.taken - before, sorted(truth[index][2] for index in free)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("out/onnx"))
    parser.add_argument("--corpus", type=Path, default=None)
    parser.add_argument("--tolerance", type=float, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()
    corpus = args.corpus or default_corpus_dir()

    model = Model.load(args.model)
    setup = model.setup
    threshold = args.threshold if args.threshold is not None else setup["peakThreshold"]
    tolerance = args.tolerance if args.tolerance is not None else setup["matchDistance"]

    print(f"Модель: {args.model} ({len(model.vocabulary)} классов)")
    print(f"Корпус: {corpus}")
    print(f"Порог вершины {threshold}, допуск совпадения {tolerance} px")
    print()

    tally = Tally()
    for frame in load_corpus(corpus):
        roster = [reference.champion_id for reference in frame.references]
        decided = recognise(frame.pixels, roster, model, threshold)
        truth = [
            (region.x + region.width / 2, region.y + region.height / 2, region.champion_id)
            for region in (frame.labels.champions if frame.labels else ())
        ]
        taken_here, lost = score_frame(decided, truth, setup["rejectName"], tolerance, tally)
        print(
            f"  {frame.name[:44]:44} взято {taken_here:2d} из {len(truth):2d}"
            + (f"   не поймано: {', '.join(lost)}" if lost else "")
        )

    taken, wrong, missed, false, labelled = (
        tally.taken,
        tally.wrong,
        tally.missed,
        tally.false_alarms,
        tally.labelled,
    )
    share = taken / labelled if labelled else 0.0
    print(f"Взято чемпионов: {taken} из {labelled} — {share:.3f}")
    print(f"  найдено, но названо неверно: {wrong}")
    print(f"  не найдено вовсе: {missed}")
    print(f"  ложных срабатываний с именем: {false}")
    print()
    print("Сверять с: tools/check_model_on_corpus.py --no-refine на тех же весах.")


if __name__ == "__main__":
    main()
