"""Детектор «где значки»: карта центров с шагом 4.

Ступень 3 программы. Сеть получает канонический кадр 320×320 и выдаёт **карту
центров** (heatmap) 80×80: значение клетки — уверенность, что здесь центр
значка чемпиона. Размер значка не предсказывается вовсе: он известен (25 px в
каноническом кадре), и лишняя голова только тратила бы ёмкость.

Почему карта центров, а не рамки: два перекрытых значка дают одну связную
область, но две отдельные вершины на карте центров. Почему шаг 4: значок
диаметром 25 px занимает при нём ~6,3 клетки — минимум, при котором мелкий
объект надёжно локализуется; при шаге 8 остаётся 3,1 клетки, и близкие значки
сливаются (own-model-plan.md).

Цель обучения — не бинарная маска, а гауссовы бугры в центрах: соседние с
центром клетки почти правильны, и наказывать их как полностью неверные
вредно. Функция потерь — фокальная с ослаблением наказания вблизи центра
(подход CenterNet): положительных клеток на два порядка меньше отрицательных,
и обычная перекрёстная энтропия утонула бы в фоне.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as torch_functional

CANONICAL_SIDE = 320
STRIDE = 4
HEATMAP_SIDE = CANONICAL_SIDE // STRIDE

# Радиус гауссова бугра в клетках карты: значок 25 px при шаге 4 — это ~6,3
# клетки, треть радиуса даёт бугор, покрывающий ядро значка.
GAUSSIAN_SIGMA = 1.0
# Порог уверенности и радиус подавления соседей при чтении карты. Порог
# выбран по кривой «точность против полноты» на кадрах корпуса
# (tools/sweep_threshold.py): от 0,1 до 0,7 полнота не меняется вовсе — сеть
# отвечает почти всегда уверенно, — а точность растёт с 0,562 до 0,672.
# Выше 0,7 начинает теряться полнота, поэтому взята верхняя точка полки.
PEAK_THRESHOLD = 0.6
NMS_KERNEL = 3
# Допуск при сверке с разметкой: расстояние между центрами в канонических
# пикселях. IoU не годится — при значке 25 px ошибка в пару пикселей резко
# меняет IoU, не влияя на прикладную корректность (own-model-plan.md).
MATCH_DISTANCE = 3.0
# Порог «парабола вырождена»: делить на такой знаменатель бессмысленно.
FLAT_PEAK_EPSILON = 1e-6


@dataclass(frozen=True)
class Augmentation:
    """Разброс, добавляемый обучающим кадрам; проверочные не трогаются.

    Ни одна ось не выдумана. **Яркость**: настоящие кадры ярче синтетики —
    45,3 против 42,0 по среднему пикселю (замер по 12 кадрам корпуса против
    датасета), и весь настоящий разброс 43,3-47,3 лежит выше синтетического
    среднего; множитель подобран так, чтобы синтетика накрывала настоящий
    диапазон с запасом. **Контраст** — тот же довод при разнице СКО 45,5
    против 45,2, поэтому разброс узкий. **Сдвиг кадра** — не наблюдаемое
    явление, а защита: без него сеть вольна запомнить, в каких местах карты
    вообще встречаются значки, и на живом кадре это знание её подводит.
    """

    shift_px: int = 2
    brightness: tuple[float, float] = (0.95, 1.20)
    contrast: tuple[float, float] = (0.92, 1.08)


NO_AUGMENTATION = Augmentation(0, (1.0, 1.0), (1.0, 1.0))


def augment_frame(
    frame: np.ndarray,
    centers: list[tuple[float, float]],
    rng: np.random.Generator,
    setup: Augmentation,
) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """Кадр (3,320,320) и центры → сдвинутый и перекрашенный кадр с центрами.

    Сдвиг делается копированием края наружу: пустых полос не появляется, а
    рамка интерфейса у края кадра остаётся рамкой.
    """
    shifted = frame
    dx = dy = 0
    if setup.shift_px:
        dx = int(rng.integers(-setup.shift_px, setup.shift_px + 1))
        dy = int(rng.integers(-setup.shift_px, setup.shift_px + 1))
        pad = setup.shift_px
        padded = np.pad(frame, ((0, 0), (pad, pad), (pad, pad)), mode="edge")
        top, left = pad - dy, pad - dx
        shifted = padded[:, top : top + CANONICAL_SIDE, left : left + CANONICAL_SIDE]

    brightness = rng.uniform(*setup.brightness)
    contrast = rng.uniform(*setup.contrast)
    mean = float(shifted.mean())
    painted = (shifted - mean) * contrast + mean * brightness
    moved = [(x + dx, y + dy) for x, y in centers]
    return np.clip(painted, 0.0, 1.0, dtype=np.float32), moved


@dataclass(frozen=True)
class DetectionMetrics:
    """Точность, полнота и средняя ошибка положения на наборе кадров."""

    matched: int
    predicted: int
    labelled: int
    mean_offset: float

    @property
    def precision(self) -> float:
        return self.matched / self.predicted if self.predicted else 0.0

    @property
    def recall(self) -> float:
        return self.matched / self.labelled if self.labelled else 0.0


def make_heatmap(centers: list[tuple[float, float]]) -> np.ndarray:
    """Карта центров (1, 80, 80) с гауссовыми буграми в позициях значков.

    Ключевая деталь: ближайшая к центру клетка получает **ровно 1,0**. Центр
    почти никогда не попадает в середину клетки, и без этого максимум бугра
    оказывался около 0,9 — а положительной в фокальной потере считается только
    клетка со значением 1,0. Получалось обучение вовсе без положительных
    примеров: потери падали, полнота оставалась нулевой.
    """
    heatmap = np.zeros((1, HEATMAP_SIDE, HEATMAP_SIDE), dtype=np.float32)
    radius = int(3 * GAUSSIAN_SIGMA)
    for x, y in centers:
        cx, cy = x / STRIDE, y / STRIDE
        left, right = int(cx) - radius, int(cx) + radius + 1
        top, bottom = int(cy) - radius, int(cy) + radius + 1
        for gy in range(max(top, 0), min(bottom, HEATMAP_SIDE)):
            for gx in range(max(left, 0), min(right, HEATMAP_SIDE)):
                distance = (gx - cx) ** 2 + (gy - cy) ** 2
                value = np.exp(-distance / (2 * GAUSSIAN_SIGMA**2))
                heatmap[0, gy, gx] = max(heatmap[0, gy, gx], value)
        peak_x = min(max(round(cx), 0), HEATMAP_SIDE - 1)
        peak_y = min(max(round(cy), 0), HEATMAP_SIDE - 1)
        heatmap[0, peak_y, peak_x] = 1.0
    return heatmap


class CenterDetector(nn.Module):
    """Маленькая свёрточная сеть: кадр 320×320 → карта центров 80×80."""

    def __init__(self, width: int = 24):
        super().__init__()

        def block(in_channels: int, out_channels: int, stride: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

        self.trunk = nn.Sequential(
            block(3, width, 1),
            block(width, width * 2, 2),  # 160
            block(width * 2, width * 2, 1),
            block(width * 2, width * 4, 2),  # 80
            block(width * 4, width * 4, 1),
            block(width * 4, width * 4, 1),
        )
        self.head = nn.Conv2d(width * 4, 1, 1)
        # Смещение головы к «здесь ничего нет»: клеток фона на два порядка
        # больше, и без этого первые шаги обучения тратятся на очевидное.
        nn.init.constant_(self.head.bias, -4.0)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Возвращает логиты карты центров; сигмоида применяется в потерях."""
        return self.head(self.trunk(batch))


def focal_heatmap_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Фокальная потеря с ослаблением наказания вблизи центра (CenterNet).

    Клетка с целью 1 — положительная. Остальные отрицательные, но чем ближе
    цель к единице, тем слабее наказание: соседи центра почти правы.
    """
    probabilities = torch.sigmoid(logits).clamp(1e-4, 1 - 1e-4)
    positive = target.ge(1.0).float()
    negative = 1.0 - positive

    positive_loss = -((1 - probabilities) ** 2) * torch.log(probabilities) * positive
    negative_loss = (
        -((1 - target) ** 4) * (probabilities**2) * torch.log(1 - probabilities) * negative
    )
    count = positive.sum().clamp(min=1.0)
    return (positive_loss.sum() + negative_loss.sum()) / count


def _parabolic_offset(plane: torch.Tensor, gy: int, gx: int, axis: int) -> float:
    """Субпиксельная поправка к клетке по вершине параболы через три значения.

    Клетка карты — это 4 канонических пикселя, поэтому округление до клетки
    само по себе даёт ошибку до 2 px. Соседние значения бугра несут долю
    пикселя: вершина параболы, проведённой через них, восстанавливает её.
    """
    height, width = plane.shape
    if axis == 0:
        if gy == 0 or gy == height - 1:
            return 0.0
        before, center, after = plane[gy - 1, gx], plane[gy, gx], plane[gy + 1, gx]
    else:
        if gx == 0 or gx == width - 1:
            return 0.0
        before, center, after = plane[gy, gx - 1], plane[gy, gx], plane[gy, gx + 1]

    denominator = float(before - 2 * center + after)
    if abs(denominator) < FLAT_PEAK_EPSILON:
        return 0.0
    offset = 0.5 * float(before - after) / denominator
    # Вершина дальше половины клетки означала бы, что максимум не здесь.
    return max(-0.5, min(0.5, offset))


def decode_heatmap(
    logits: torch.Tensor, threshold: float = PEAK_THRESHOLD
) -> list[list[tuple[float, float, float]]]:
    """Логиты → список центров (x, y, уверенность) в канонических пикселях.

    Локальные максимумы находит максимум-пулинг: клетка остаётся, только если
    равна максимуму своей окрестности. Так близкие значки дают две вершины, а
    один значок — одну.

    Клетка g соответствует канонической координате `g * STRIDE`: цель строится
    округлением `x / STRIDE` до клетки. Первая версия прибавляла полклетки и
    давала систематический сдвиг +2 px — при допуске 3 px этого хватало, чтобы
    терять больше половины верных находок даже у идеальной модели. Остаток
    округления снимает субпиксельная поправка по соседним клеткам.
    """
    probabilities = torch.sigmoid(logits)
    pooled = torch_functional.max_pool2d(
        probabilities, NMS_KERNEL, stride=1, padding=NMS_KERNEL // 2
    )
    peaks = (probabilities == pooled) & (probabilities >= threshold)

    results: list[list[tuple[float, float, float]]] = []
    for index in range(probabilities.shape[0]):
        found = []
        plane = probabilities[index, 0]
        ys, xs = torch.nonzero(peaks[index, 0], as_tuple=True)
        for gy, gx in zip(ys.tolist(), xs.tolist(), strict=True):
            confidence = float(plane[gy, gx])
            offset_x = _parabolic_offset(plane, gy, gx, axis=1)
            offset_y = _parabolic_offset(plane, gy, gx, axis=0)
            found.append(((gx + offset_x) * STRIDE, (gy + offset_y) * STRIDE, confidence))
        results.append(sorted(found, key=lambda item: -item[2]))
    return results


def match_centers(
    predicted: list[tuple[float, float, float]],
    labelled: list[tuple[float, float]],
    max_distance: float = MATCH_DISTANCE,
) -> tuple[int, float]:
    """Сопоставление один-к-одному по близости; возвращает попадания и ошибку."""
    free = list(labelled)
    matched = 0
    offsets: list[float] = []
    for x, y, _ in predicted:
        if not free:
            break
        distances = [((x - lx) ** 2 + (y - ly) ** 2) ** 0.5 for lx, ly in free]
        best = int(np.argmin(distances))
        if distances[best] <= max_distance:
            matched += 1
            offsets.append(distances[best])
            free.pop(best)
    return matched, float(np.mean(offsets)) if offsets else 0.0
