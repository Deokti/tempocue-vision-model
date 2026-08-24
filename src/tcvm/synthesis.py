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

Миньоны — колонны тонированных точек minionmapcircle вдоль линий.

Мелкие постоянные объекты (лагеря джунглей, растения) — простые фигуры
измеренного цвета и размера: на канонической карте они занимают 3-5 px, и
рисовать их иконками интерфейса бессмысленно. Их отсутствие было главной
причиной, по которой судья доменов отличал синтетику (docs/generator.md).

Константы измерены по кадрам корпуса; происхождение у каждой в комментарии.
Пока не моделируются: пинги, свечение отзыва/телепорта, рамка интерфейса по
краю кадра, обрезка обзора стенами, канонические позиции построек (в превью
постройки стоят только там, где размечены).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .render import gaussian_blur

# Устройство фона измерено по кадрам корпуса (24.08.2026). Земля рисуется
# самой текстурой слоя: под туманом войны с множителем 0,29, в зоне видимости
# — ровно 1,0 (открытая карта не затемняется вообще).
GROUND_DIM_FOG = 0.29
GROUND_DIM_VISIBLE = 1.0

# Чёрные области карты (стены, скалы, часть кустов) — самостоятельные данные:
# ни альфа текстуры, ни grasstint, ни накладка тумана их не описывают
# (проверено). Маска снята с 17 настоящих кадров как «пиксель тёмный больше
# чем в 60 % кадров» и лежит в annotations/map-darkness.png. Именно контраст
# чёрных областей со светлой землёй даёт настоящему кадру его резкость.
DARKNESS_BGR = (6, 6, 4)
# Порог чтения чёрно-белой маски из PNG: середина шкалы.
MASK_THRESHOLD = 127

# Радиус обзора в канонических пикселях: обзор чемпиона ~1200 игровых
# единиц при стороне карты ~15000 единиц и канонических 320 px.
SIGHT_RADIUS = 26.0
# Мягкость края зоны видимости, подобрана глазами по кадру 01.
SIGHT_EDGE_SIGMA = 4.0

# Командные цвета построек: те же, что у колец чемпионов, — сравнение
# бок-о-бок подтвердило, что игра красит команды одним цветом, а замер по
# Tower-регионам занижал яркость из-за тёмной заливки щита в выборке.
STRUCTURE_ALLY_BGR = (208, 151, 79)
STRUCTURE_ENEMY_BGR = (50, 59, 203)
# Сторона иконки башни в канонических пикселях: подбор NCC по настоящей
# башне кадра 01 (16-24 px, максимум на 23). Tower-регионы разметки 16x16 —
# рамка меньше видимого значка.
STRUCTURE_SIDE = 23
# Порог альфы «пиксель принадлежит иконке»: половина шкалы, край сглаживания.
VISIBLE_ALPHA = 128

# Миньоны: белая точка minionmapcircle из ассетов, тонируется цветом команды.
# Цвета — медианы насыщенных пикселей волн: союзные из кадра 03, вражеские из
# lag-01 (замер 24.08.2026); диаметр точки ~5 px по зуму кадра 03.
MINION_ALLY_BGR = (201, 142, 71)
MINION_ENEMY_BGR = (35, 35, 126)
MINION_SIDE = 5
# Шаг точек в колонне волны: точки соприкасаются (зум кадра 03).
MINION_SPACING = 6

# Мелкие объекты карты: цвет — медиана пикселей соответствующего цветового
# кластера по кадрам корпуса, сторона — медиана ширины найденных пятен
# (замер 24.08.2026, tools/extract_map_objects.py).
MAP_OBJECT_STYLE = {
    "camp": {"bgr": (50, 103, 162), "side": 5, "shape": "diamond"},
    "plant_green": {"bgr": (60, 154, 63), "side": 3, "shape": "square"},
    "plant_yellow": {"bgr": (58, 149, 198), "side": 3, "shape": "square"},
}

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


def load_darkness_mask(path: Path, side: int) -> np.ndarray:
    """Маска чёрных областей карты (annotations/map-darkness.png)."""
    with Image.open(path) as image:
        grey = np.asarray(image.convert("L").resize((side, side), Image.NEAREST))
    return grey > MASK_THRESHOLD


def compose_background(
    layer_bgra: np.ndarray,
    side: int,
    visibility: np.ndarray | None = None,
    darkness: np.ndarray | None = None,
) -> np.ndarray:
    """Фон миникарты стороной side (BGR float).

    Земля рисуется текстурой: приглушённо под туманом, в полную силу в зоне
    видимости. Поверх накладываются чёрные области карты по маске.
    """
    layer = Image.fromarray(layer_bgra[..., [2, 1, 0, 3]], "RGBA").resize(
        (side, side), Image.BILINEAR
    )
    layer_np = np.asarray(layer).astype(np.float64)
    texture = layer_np[..., [2, 1, 0]]

    fogged = texture * GROUND_DIM_FOG
    if visibility is None:
        canvas = fogged
    else:
        lit = texture * GROUND_DIM_VISIBLE
        v = visibility[..., np.newaxis]
        canvas = lit * v + fogged * (1.0 - v)

    if darkness is not None:
        canvas = np.where(darkness[..., np.newaxis], np.array(DARKNESS_BGR, float), canvas)
    return canvas


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


def draw_minion_column(
    canvas_bgr: np.ndarray,
    dot_bgra: np.ndarray,
    start_xy: tuple[int, int],
    direction_xy: tuple[float, float],
    count: int,
) -> None:
    """Колонна миньонов: точки с шагом MINION_SPACING вдоль направления."""
    length = max((direction_xy[0] ** 2 + direction_xy[1] ** 2) ** 0.5, 1e-6)
    step = (
        direction_xy[0] / length * MINION_SPACING,
        direction_xy[1] / length * MINION_SPACING,
    )
    for index in range(count):
        center = (
            round(start_xy[0] + step[0] * index),
            round(start_xy[1] + step[1] * index),
        )
        place_icon(canvas_bgr, dot_bgra, center, MINION_SIDE)


def draw_map_object(canvas_bgr: np.ndarray, kind: str, x: int, y: int) -> None:
    """Рисует мелкий объект карты: квадрат растения или ромб лагеря."""
    style = MAP_OBJECT_STYLE[kind]
    side = style["side"]
    half = side // 2
    for dy in range(-half, half + 1):
        for dx in range(-half, half + 1):
            if style["shape"] == "diamond" and abs(dx) + abs(dy) > half:
                continue
            py, px = y + dy, x + dx
            if 0 <= py < canvas_bgr.shape[0] and 0 <= px < canvas_bgr.shape[1]:
                canvas_bgr[py, px] = style["bgr"]


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
