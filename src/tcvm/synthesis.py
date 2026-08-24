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

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .render import gaussian_blur

# Устройство фона измерено по кадрам корпуса (24.08.2026). Земля рисуется
# самой текстурой слоя: под туманом войны приглушённо, в зоне видимости —
# ровно 1,0 (открытая карта не затемняется вообще).
#
# Множитель тумана уточнён по распределению отношения «кадр к текстуре» на
# 605 537 пикселях земли двенадцати кадров: распределение двугорбое, и горбы
# стоят ровно там, где им положено — 1,01 для освещённой земли и 0,36 для
# тумана. Прежнее значение 0,29 делало туман заметно темнее настоящего;
# ошибка пряталась за тем, что средняя яркость кадра всё равно сходилась —
# её вытягивала избыточная чернота, которой не было в игре.
GROUND_DIM_FOG = 0.36
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
    # Варды: тот же зелёный, что у растений (замер по кадрам корпуса — там
    # большая часть зелёных пятен оказалась именно вардами), розовые реже.
    "ward_green": {"bgr": (60, 154, 63), "side": 3, "shape": "square"},
    "ward_pink": {"bgr": (150, 70, 190), "side": 3, "shape": "square"},
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


@dataclass(frozen=True)
class MapPlacement:
    """Куда и в каком размере ложится текстура карты в каноническом кадре."""

    side: int  # сторона карты в канонических пикселях
    dx: int  # сдвиг относительно центра кадра
    dy: int


# Карта занимает в каноническом кадре **не всю сторону**: вокруг неё рамка
# интерфейса. Замер — совмещение текстуры слоя с 12 кадрами корпуса по NCC
# (внутренняя область 230x230, перебор стороны 286-302 и сдвигов ±6):
# 10 кадров из 12 дали сторону 294 при сдвиге вниз на 2 пикселя; совпадение
# при ней 0,64 против 0,28 при прежних 320 — разница не оставляет места
# сомнению. Оставшиеся два кадра дали 290: это погрешность совмещения на
# плотных кадрах, а не второй захват, потому что кайма во всех 12 кадрах
# совпадает побитово.
MAP_PLACEMENT = MapPlacement(294, 0, 2)


def map_rect(side: int, placement: MapPlacement) -> tuple[int, int, int, int]:
    """Границы области карты в кадре: (левая, верхняя, правая, нижняя)."""
    offset = (side - placement.side) // 2
    left, top = offset + placement.dx, offset + placement.dy
    return left, top, left + placement.side, top + placement.side


def place_map_texture(layer_bgra: np.ndarray, side: int, placement: MapPlacement) -> np.ndarray:
    """Текстура карты, вписанная в кадр стороной side; вне карты — нули."""
    scaled = Image.fromarray(layer_bgra[..., [2, 1, 0, 3]], "RGBA").resize(
        (placement.side, placement.side), Image.BILINEAR
    )
    texture = np.asarray(scaled).astype(np.float64)[..., [2, 1, 0]]

    canvas = np.zeros((side, side, 3), dtype=np.float64)
    x0, y0, x1, y1 = map_rect(side, placement)
    left, top = max(0, x0), max(0, y0)
    right, bottom = min(side, x1), min(side, y1)
    canvas[top:bottom, left:right] = texture[top - y0 : bottom - y0, left - x0 : right - x0]
    return canvas


def compose_background(
    layer_bgra: np.ndarray,
    side: int,
    visibility: np.ndarray | None = None,
    darkness: np.ndarray | None = None,
    placement: MapPlacement = MAP_PLACEMENT,
) -> np.ndarray:
    """Фон миникарты стороной side (BGR float).

    Земля рисуется текстурой: приглушённо под туманом, в полную силу в зоне
    видимости. Поверх накладываются чёрные области карты по маске.
    """
    texture = place_map_texture(layer_bgra, side, placement)

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


def load_map_border(path: Path) -> np.ndarray:
    """Рамка интерфейса миникарты (annotations/map-border.png) как BGRA."""
    with Image.open(path) as image:
        rgba = np.asarray(image.convert("RGBA"))
    return rgba[..., [2, 1, 0, 3]].copy()


def draw_border(canvas_bgr: np.ndarray, border_bgra: np.ndarray) -> None:
    """Кладёт рамку интерфейса поверх всего: в игре она рисуется последней."""
    alpha = border_bgra[..., 3:4].astype(np.float64) / 255.0
    canvas_bgr[:] = border_bgra[..., :3] * alpha + canvas_bgr * (1.0 - alpha)


# Рамка обзора камеры: белый прямоугольник в один пиксель, который игрок
# двигает по карте. Размер замерен по кадрам корпуса, где рамка видна целиком
# (6 кадров из 12): ширина 107-109, высота 45-46 — берутся медианы. В
# остальных шести рамка обрезана краем карты, и это тоже воспроизводится.
# Линия чисто белая (255,255,255) толщиной в пиксель; соседний пиксель светлее
# фона — это сглаживание дробного положения, поэтому линия рисуется с
# субпиксельным весом, а не по целым координатам.
CAMERA_RECT_SIZE = (108, 45)
CAMERA_LINE_BGR = (255, 255, 255)


def _blend_line(canvas_bgr: np.ndarray, position: float, axis: int, span: slice) -> None:
    """Линия белого цвета с дробным положением: вес делится между соседями."""
    low = int(np.floor(position))
    weight_high = position - low
    for index, weight in ((low, 1.0 - weight_high), (low + 1, weight_high)):
        if weight <= 0.0 or not 0 <= index < canvas_bgr.shape[1 - axis]:
            continue
        strip = canvas_bgr[index, span] if axis == 0 else canvas_bgr[span, index]
        strip[:] = np.array(CAMERA_LINE_BGR, float) * weight + strip * (1.0 - weight)


def draw_camera_rect(
    canvas_bgr: np.ndarray, center: tuple[float, float], bounds: tuple[int, int, int, int]
) -> None:
    """Рамка обзора камеры вокруг center, обрезанная границами карты bounds."""
    width, height = CAMERA_RECT_SIZE
    left, top = center[0] - width / 2, center[1] - height / 2
    map_left, map_top, map_right, map_bottom = bounds

    inside_x = slice(max(map_left, int(left)), min(map_right, int(left + width) + 1))
    inside_y = slice(max(map_top, int(top)), min(map_bottom, int(top + height) + 1))
    if inside_x.start >= inside_x.stop or inside_y.start >= inside_y.stop:
        return
    for edge in (top, top + height):
        if map_top <= edge < map_bottom - 1:
            _blend_line(canvas_bgr, edge, 0, inside_x)
    for edge in (left, left + width):
        if map_left <= edge < map_right - 1:
            _blend_line(canvas_bgr, edge, 1, inside_y)


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
