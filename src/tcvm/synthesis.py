"""Сборка синтетического кадра миникарты: фон из ассетов игры плюс значки.

Первый шаг ступени 2 (композитор). Фон: тёмный «пол», поверх — слой карты
(2dlevelminimap_*, 512×512, прозрачное = пол), затем общее затемнение — в
матче большая часть карты лежит под туманом войны. Значки: circle-иконка
базового скина, обрезанная в круг, с командным кольцом, уменьшенная
измеренной цепочкой (bilinear, смешение в sRGB — docs/render-verification.md)
и наложенная по альфе.

Фон собирается в двух версиях — открытая и под туманом войны — и смешивается
маской зоны видимости (круги обзора с мягким краем).

Постройки — тонируемые иконки интерфейса из ассетов (turret_*plate, tower,
inhibitor, nexus): тёмная заливка со светлым контуром, цвет задаёт команда.

Константы измерены по кадрам корпуса; происхождение у каждой в комментарии.
Пока не моделируются: миньоны, пинги, свечение отзыва/телепорта, рамка
интерфейса по краю кадра, обрезка обзора стенами.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .render import gaussian_blur

# Медианы по кадру static-tower-as-champion-01 (замер 24.08.2026):
# цвет пола — медиана пикселей кадра там, где слой карты прозрачен,
# отдельно под туманом и в зоне видимости (порог по яркости).
FLOOR_FOG_BGR = (27, 28, 16)
FLOOR_VISIBLE_BGR = (44, 56, 52)
# Затемнение слоя карты: отношение яркости стен настоящего кадра к яркости
# текстуры. Под туманом три канала дают один множитель ~0,33; в зоне
# видимости множитель ровно 1,0 — открытая карта рисуется без затемнения.
MAP_DIM_FOG = 0.33
MAP_DIM_VISIBLE = 1.0

# Радиус обзора в канонических пикселях: обзор чемпиона ~1200 игровых
# единиц при стороне карты ~15000 единиц и канонических 320 px.
SIGHT_RADIUS = 26.0
# Мягкость края зоны видимости, подобрана глазами по кадру 01.
SIGHT_EDGE_SIGMA = 4.0

# Командные цвета построек: медиана ярких насыщенных пикселей Tower-регионов
# разметки кадров 01, 02 и 07 (замер 24.08.2026). Иконки построек в ассетах
# серые с альфой — игра тонирует их цветом команды.
STRUCTURE_ALLY_BGR = (117, 124, 45)
STRUCTURE_ENEMY_BGR = (39, 40, 145)
# Сторона иконки башни в канонических пикселях — размер Tower-регионов разметки.
STRUCTURE_SIDE = 16
# Порог альфы «пиксель принадлежит иконке»: половина шкалы, край сглаживания.
VISIBLE_ALPHA = 128

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


def visibility_mask(
    side: int,
    sight_sources: list[tuple[int, int]],
    radius: float = SIGHT_RADIUS,
    edge_sigma: float = SIGHT_EDGE_SIGMA,
) -> np.ndarray:
    """Маска зоны видимости [0..1]: круги обзора с мягким краем.

    Игровая логика обзора (стены, кусты) не воспроизводится — для синтетики
    важна правдоподобная текстура пятен света, а не честная симуляция.
    """
    yy, xx = np.mgrid[0:side, 0:side].astype(np.float64)
    mask = np.zeros((side, side))
    for cx, cy in sight_sources:
        distance = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        mask = np.maximum(mask, distance <= radius)
    if edge_sigma > 0:
        mask = gaussian_blur(mask[..., np.newaxis], edge_sigma)[..., 0]
    return np.clip(mask, 0.0, 1.0)


def compose_background(
    layer_bgra: np.ndarray,
    side: int,
    visibility: np.ndarray | None = None,
) -> np.ndarray:
    """Фон миникарты стороной side (BGR float).

    Две версии — открытая (светлый пол, текстура как есть) и под туманом
    (тёмный пол, текстура на треть яркости) — смешиваются маской видимости.
    """
    layer = Image.fromarray(layer_bgra[..., [2, 1, 0, 3]], "RGBA").resize(
        (side, side), Image.BILINEAR
    )
    layer_np = np.asarray(layer).astype(np.float64)
    alpha = layer_np[..., 3:4] / 255.0
    layer_bgr = layer_np[..., [2, 1, 0]]

    def blend(floor_bgr, dim):
        floor = np.full((side, side, 3), floor_bgr, dtype=np.float64)
        return floor * (1.0 - alpha) + layer_bgr * dim * alpha

    fogged = blend(FLOOR_FOG_BGR, MAP_DIM_FOG)
    if visibility is None:
        return fogged
    lit = blend(FLOOR_VISIBLE_BGR, MAP_DIM_VISIBLE)
    v = visibility[..., np.newaxis]
    return lit * v + fogged * (1.0 - v)


def load_minimap_icon(icons_dir: Path, name: str) -> np.ndarray:
    """Иконка интерфейса миникарты (`assets/ux/minimap/icons`) как BGRA."""
    with Image.open(icons_dir / f"{name}.png") as image:
        rgba = np.asarray(image.convert("RGBA"))
    return rgba[..., [2, 1, 0, 3]].copy()


def tinted_icon(icon_bgra: np.ndarray, tint_bgr: tuple[int, int, int]) -> np.ndarray:
    """Тонирует серую иконку командным цветом, сохраняя её светотень.

    Иконки построек — тёмная заливка со светлым контуром; нормируем яркость
    на светлую часть (90-й перцентиль видимых пикселей), чтобы контур получил
    измеренный командный цвет, а сердцевина осталась тёмной.
    """
    luminance = icon_bgra[..., :3].astype(np.float64).mean(axis=2)
    visible = icon_bgra[..., 3] > VISIBLE_ALPHA
    reference = np.percentile(luminance[visible], 90) if visible.any() else 255.0
    scale = luminance[..., np.newaxis] / max(reference, 1.0)

    result = icon_bgra.copy()
    result[..., :3] = np.clip(np.array(tint_bgr) * scale, 0, 255).astype(np.uint8)
    return result


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
