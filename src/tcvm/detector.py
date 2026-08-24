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
# Порог уверенности и радиус подавления соседей при чтении карты.
PEAK_THRESHOLD = 0.3
NMS_KERNEL = 3
# Допуск при сверке с разметкой: расстояние между центрами в канонических
# пикселях. IoU не годится — при значке 25 px ошибка в пару пикселей резко
# меняет IoU, не влияя на прикладную корректность (own-model-plan.md).
MATCH_DISTANCE = 3.0


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
    """Карта центров (1, 80, 80) с гауссовыми буграми в позициях значков."""
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


def decode_heatmap(
    logits: torch.Tensor, threshold: float = PEAK_THRESHOLD
) -> list[list[tuple[float, float, float]]]:
    """Логиты → список центров (x, y, уверенность) в канонических пикселях.

    Локальные максимумы находит максимум-пулинг: клетка остаётся, только если
    равна максимуму своей окрестности. Так близкие значки дают две вершины, а
    один значок — одну.
    """
    probabilities = torch.sigmoid(logits)
    pooled = torch_functional.max_pool2d(
        probabilities, NMS_KERNEL, stride=1, padding=NMS_KERNEL // 2
    )
    peaks = (probabilities == pooled) & (probabilities >= threshold)

    results: list[list[tuple[float, float, float]]] = []
    for index in range(probabilities.shape[0]):
        found = []
        ys, xs = torch.nonzero(peaks[index, 0], as_tuple=True)
        for gy, gx in zip(ys.tolist(), xs.tolist(), strict=True):
            confidence = float(probabilities[index, 0, gy, gx])
            found.append(((gx + 0.5) * STRIDE, (gy + 0.5) * STRIDE, confidence))
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
