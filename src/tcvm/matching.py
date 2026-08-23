"""Сравнение вырезов значков: маскированная нормированная кросс-корреляция.

Метрика у сверки отрисовки и у замера шумового пола одна и та же: косинус
похожести двух вырезов после вычитания среднего, посчитанный только по
пикселям внутри маски. Кольцо команды и фон исключаются из сравнения:
кольцо — не признак личности, а фон под краем меняется при каждом сдвиге.
"""

from __future__ import annotations

import numpy as np

# Значок в каноническом кадре занимает около 25 пикселей; внутренний диск
# радиуса 10 отрезает командное кольцо (~2 px) и сглаженный край.
ICON_SIDE = 25
INNER_RADIUS = 10.0
# Центральный квадрат, которым конвейер фактически решает личность.
CENTER_SIDE = 11


def circular_mask(side: int, radius: float) -> np.ndarray:
    """Булева маска диска данного радиуса с центром в середине квадрата."""
    center = (side - 1) / 2
    yy, xx = np.mgrid[0:side, 0:side]
    return (yy - center) ** 2 + (xx - center) ** 2 <= radius**2


def center_square_mask(side: int, square: int) -> np.ndarray:
    """Булева маска центрального квадрата square x square."""
    mask = np.zeros((side, side), dtype=bool)
    start = (side - square) // 2
    mask[start : start + square, start : start + square] = True
    return mask


def masked_ncc(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    """Нормированная кросс-корреляция (zero-mean NCC) по пикселям маски.

    1,0 — совпадение с точностью до общей яркости и контраста; около нуля —
    связи нет. Считается по трём цветовым каналам сразу, альфа не участвует.
    """
    av = a[mask][:, :3].astype(np.float64).ravel()
    bv = b[mask][:, :3].astype(np.float64).ravel()
    av -= av.mean()
    bv -= bv.mean()
    denominator = np.linalg.norm(av) * np.linalg.norm(bv)
    if denominator == 0:
        return 0.0
    return float(av @ bv / denominator)


def crop_centered(frame: np.ndarray, cx: int, cy: int, side: int = ICON_SIDE) -> np.ndarray:
    """Вырез side x side с центром (cx, cy); у края кадра поднимает ошибку."""
    half = side // 2
    top, left = cy - half, cx - half
    if top < 0 or left < 0 or top + side > frame.shape[0] or left + side > frame.shape[1]:
        raise ValueError(f"Вырез ({cx}, {cy}) выходит за границы кадра")
    return frame[top : top + side, left : left + side]


def find_best_match(
    frame: np.ndarray,
    template: np.ndarray,
    mask: np.ndarray,
    cx: int,
    cy: int,
    search_radius: int,
) -> tuple[int, int, float]:
    """Ищет положение шаблона рядом с (cx, cy) перебором целых сдвигов.

    Возвращает центр лучшего совпадения и его балл NCC. Субпиксельные
    положения не перебираются: их остаток — часть измеряемого шума.
    """
    best = (cx, cy, -2.0)
    for dy in range(-search_radius, search_radius + 1):
        for dx in range(-search_radius, search_radius + 1):
            try:
                candidate = crop_centered(frame, cx + dx, cy + dy, template.shape[0])
            except ValueError:
                continue
            score = masked_ncc(candidate, template, mask)
            if score > best[2]:
                best = (cx + dx, cy + dy, score)
    return best
