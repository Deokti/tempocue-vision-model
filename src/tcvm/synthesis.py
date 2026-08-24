"""Сборка синтетического кадра миникарты: фон из ассетов игры плюс значки.

Первый шаг ступени 2 (композитор). Фон: тёмный «пол», поверх — слой карты
(2dlevelminimap_*, 512×512, прозрачное = пол), затем общее затемнение — в
матче большая часть карты лежит под туманом войны. Значки: circle-иконка
базового скина, обрезанная в круг, с командным кольцом, уменьшенная
измеренной цепочкой (bilinear, смешение в sRGB — docs/render-verification.md)
и наложенная по альфе.

Константы измерены по кадрам корпуса; происхождение у каждой в комментарии.
Пока не моделируются: зона видимости (затемнение однородное), свечение,
пинги, миньоны, постройки, рамка интерфейса по краю кадра.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

# Медианы по кадру static-tower-as-champion-01 (замер 24.08.2026):
# цвет пола — медиана пикселей кадра там, где слой карты прозрачен.
FLOOR_BGR = (27, 28, 16)
# Затемнение карты: отношение яркости стен настоящего кадра к яркости
# текстуры (40/112, 50/153, 40/144 ≈ один множитель по трём каналам).
MAP_DIM = 0.33

# Цвета колец: медиана верхней четверти по насыщенности кольцевой полосы
# (радиусы 10,6–12,4) кадра 08 — сглаженные с фоном пиксели отсеяны.
# Союзное — по Sett; вражеское — медиана Shen и Teemo (согласованы).
ALLY_RING_BGR = (208, 151, 79)
ENEMY_RING_BGR = (50, 59, 203)
# Геометрия кольца в долях стороны значка: при 25 px кольцо занимает
# радиусы ~10,5–12,4 — согласуется с маской сверки (портрет до радиуса 10).
RING_OUTER_FRAC = 0.496
RING_THICKNESS_FRAC = 0.076


def load_map_layer(map_dir: Path, variant: str) -> np.ndarray:
    """Слой карты `2dlevelminimap_<variant>.png` как BGRA-массив 512×512."""
    with Image.open(map_dir / f"2dlevelminimap_{variant}.png") as image:
        rgba = np.asarray(image.convert("RGBA"))
    return rgba[..., [2, 1, 0, 3]].copy()


def compose_background(
    layer_bgra: np.ndarray,
    side: int,
    floor_bgr: tuple[int, int, int] = FLOOR_BGR,
    dim: float = MAP_DIM,
) -> np.ndarray:
    """Фон миникарты стороной side: пол + слой карты + затемнение (BGR float)."""
    layer = Image.fromarray(layer_bgra[..., [2, 1, 0, 3]], "RGBA").resize(
        (side, side), Image.BILINEAR
    )
    layer_np = np.asarray(layer).astype(np.float64)
    alpha = layer_np[..., 3:4] / 255.0
    layer_bgr = layer_np[..., [2, 1, 0]]

    floor = np.full((side, side, 3), floor_bgr, dtype=np.float64)
    # Пол уже измерен в затемнённом виде, поэтому множитель — только на слой.
    return floor * (1.0 - alpha) + layer_bgr * dim * alpha


def ringed_icon(icon_bgra: np.ndarray, ring_bgr: tuple[int, int, int]) -> np.ndarray:
    """Circle-иконка с командным кольцом в родном разрешении (BGRA).

    Портрет обрезается в круг до внутреннего края кольца, кольцо рисуется
    заливкой кольцевой полосы; сглаживание краёв даст последующее уменьшение.
    """
    side = icon_bgra.shape[0]
    center = (side - 1) / 2
    yy, xx = np.mgrid[0:side, 0:side]
    radius = np.sqrt((yy - center) ** 2 + (xx - center) ** 2)
    outer = RING_OUTER_FRAC * side
    inner = outer - RING_THICKNESS_FRAC * side

    result = icon_bgra.copy()
    ring_band = (radius >= inner) & (radius <= outer)
    result[ring_band, :3] = ring_bgr
    result[..., 3] = np.where(radius <= outer, 255, 0)
    return result


def place_icon(
    canvas_bgr: np.ndarray,
    icon_bgra: np.ndarray,
    center_xy: tuple[int, int],
    icon_side: int,
) -> None:
    """Сажает значок на фон: уменьшение bilinear в sRGB, наложение по альфе."""
    icon = Image.fromarray(icon_bgra[..., [2, 1, 0, 3]], "RGBA").resize(
        (icon_side, icon_side), Image.BILINEAR
    )
    icon_np = np.asarray(icon).astype(np.float64)
    alpha = icon_np[..., 3:4] / 255.0
    icon_bgr = icon_np[..., [2, 1, 0]]

    half = icon_side // 2
    x0, y0 = center_xy[0] - half, center_xy[1] - half
    region = canvas_bgr[y0 : y0 + icon_side, x0 : x0 + icon_side]
    region[:] = icon_bgr * alpha + region * (1.0 - alpha)


def to_uint8_bgr(canvas: np.ndarray) -> np.ndarray:
    return np.clip(canvas + 0.5, 0, 255).astype(np.uint8)
