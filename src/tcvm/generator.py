"""Случайные синтетические кадры миникарты с бесплатной разметкой.

Сцена собирается композитором (`synthesis`) по случайным осям, каждая из
которых обоснована замером или находкой программы:

- вариант рельефа — один из 16 (души драконов перерисовывают карту);
- сторона игрока — союзная база southwest или northeast (замер lag-01);
- состав — случайные чемпионы, базовые circle-иконки (сверка отрисовки);
- позиции — по проходимой области (пол и река слоя карты), с намеренными
  кластерами для перекрытий — главного источника пропусков конвейера;
- субпиксельный сдвиг и размытие значка — главный источник изменчивости
  (шумовой пол: движение даёт остаток до 0,34, покой — ровно 0);
- постройки — канонические позиции, состояние зависит от стадии игры
  (пластины до ~14 минут, снос к поздней игре);
- зона видимости — от союзных построек, чемпионов и случайных вардов;
- мелкие объекты карты (лагеря, растения) — из канонических позиций, часть
  случайно скрыта: лагерь убит, растение сорвано;
- миньоны — колонны, ложащиеся вдоль проходимых коридоров.

Кадры канонические 320×320: значок чемпиона всегда ~25 px, как после
нормализации приложения. Родные размеры 16-65 px — для будущего генератора
вырезов вложения, не для кадров детектора.

Пока не рандомизируются: пинги, свечение отзыва, смерть чемпиона,
направленное размытие движения (пока изотропное гауссово).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .cdragon import base_circle_bgra
from .matching import ICON_SIDE
from .render import gaussian_blur
from .synthesis import (
    ALLY_RING_BGR,
    ENEMY_RING_BGR,
    MINION_ALLY_BGR,
    MINION_ENEMY_BGR,
    MINION_SPACING,
    STRUCTURE_ALLY_BGR,
    STRUCTURE_ENEMY_BGR,
    STRUCTURE_SIDE,
    compose_background,
    draw_map_object,
    draw_minion_column,
    load_darkness_mask,
    load_map_layer,
    load_minimap_icon,
    place_icon,
    ringed_icon,
    tinted_icon,
    to_uint8_bgr,
    visibility_mask,
)

CANONICAL_SIDE = 320

MAP_VARIANTS = (
    "base_baron1",
    "base_baron2",
    "base_baron3",
    "cloud_baron1",
    "cloud_baron2",
    "cloud_baron3",
    "hextech_baron1",
    "hextech_baron2",
    "hextech_baron3",
    "infernal_baron1",
    "mountain_baron1",
    "mountain_baron2",
    "mountain_baron3",
    "ocean_baron1",
    "ocean_baron2",
    "ocean_baron3",
)

# Проходимость выводится из маски чёрных областей карты: тёмное — стены и
# скалы, светлое — проходимая земля. Альфа текстуры для этого не годится:
# сравнение показало, что она почти не связана с темнотой настоящих кадров.
# Отступ значка от края кадра: половина значка плюс запас на кольцо.
EDGE_MARGIN = 14

# Кластеры: доля чемпионов, сажаемых рядом с уже стоящим, и радиусы посадки.
# Перекрытия — главный источник пропусков конвейера (10 из 16), их в данных
# должно быть много.
CLUSTER_PROBABILITY = 0.35
CLUSTER_RADIUS = (5, 18)

# Размытие значка: изотропное приближение размытия движения. Верх подобран
# по шумовому полу: сигма ~1,2 даёт расхождение с резким значком порядка
# наблюдаемого у движущегося Дариуса.
BLUR_SIGMA_MAX = 1.2
MOVING_PROBABILITY = 0.45

# Стадия игры: до этой доли lateness башни носят пластины (в игре пластины
# падают к 14-й минуте), дальше башни начинают сноситься.
PLATES_UNTIL = 0.3
DEMOLISH_CHANCE_AT_LATE = 0.6

# Направления колонн миньонов: восемь румбов. Само направление выбирается
# не случайно, а по проходимости — колонна ложится вдоль коридора. Задавать
# линии координатами оказалось хуже: ломаная мимо коридора обрезалась в ноль.
COLUMN_DIRECTIONS = (
    (1.0, 0.0),
    (0.0, 1.0),
    (1.0, 1.0),
    (1.0, -1.0),
    (-1.0, 0.0),
    (0.0, -1.0),
    (-1.0, -1.0),
    (-1.0, 1.0),
)
MINION_COUNT = (4, 8)
# Равные шансы двух исходов: сторона игрока, принадлежность колонны, половина
# состава — всё это честная монета.
EVEN_ODDS = 0.5

WARD_SIGHT_MAX = 4

# Доля видимых мелких объектов карты: лагерь убит или растение сорвано —
# обычное дело, но большая часть карты в любой момент цела.
MAP_OBJECT_VISIBLE_SHARE = (0.55, 0.95)
# Варды: живые точки обзора на карте, число случайно. Розовые (контрольные)
# ставят реже обычных зелёных.
WARD_COUNT_MAX = 8
PINK_WARD_SHARE = 0.25


@dataclass(frozen=True)
class PlacedChampion:
    champion_id: str
    x: float
    y: float
    ally: bool
    moving: bool
    blur_sigma: float


@dataclass(frozen=True)
class MinionColumn:
    x: int
    y: int
    direction: tuple[float, float]
    ally: bool
    count: int


@dataclass(frozen=True)
class Scene:
    """Полное описание случайного кадра; разметка следует из него бесплатно."""

    variant: str
    ally_side: str  # "southwest" | "northeast"
    lateness: float  # 0 — ранняя игра, 1 — поздняя
    champions: tuple[PlacedChampion, ...]
    minion_columns: tuple[MinionColumn, ...]
    ward_sights: tuple[tuple[int, int, str], ...]  # x, y, тип иконки варда
    turret_states: tuple[str, ...]  # имя иконки или "destroyed", по map-structures
    visible_map_objects: tuple[bool, ...]  # по объекту из map-objects.json


class AssetLibrary:
    """Ленивый доступ к ассетам с кэшем в памяти; сеть только при промахе кэша."""

    def __init__(
        self,
        map_dir: Path,
        patch: str,
        structures: list[dict],
        map_objects: list[dict] | None = None,
        darkness_path: Path | None = None,
    ):
        self.map_dir = map_dir
        self.patch = patch
        self.structures = structures
        self.map_objects = map_objects or []
        self.darkness = (
            load_darkness_mask(darkness_path, CANONICAL_SIDE) if darkness_path else None
        )
        self._layers: dict[str, np.ndarray] = {}
        self._icons: dict[str, np.ndarray] = {}
        self._circles: dict[str, np.ndarray | None] = {}

    def layer(self, variant: str) -> np.ndarray:
        if variant not in self._layers:
            self._layers[variant] = load_map_layer(self.map_dir / self.patch / "map11", variant)
        return self._layers[variant]

    def icon(self, name: str) -> np.ndarray:
        if name not in self._icons:
            self._icons[name] = load_minimap_icon(
                self.map_dir / self.patch / "minimap-icons", name
            )
        return self._icons[name]

    def circle(self, champion_id: str) -> np.ndarray | None:
        if champion_id not in self._circles:
            try:
                self._circles[champion_id] = base_circle_bgra(champion_id, self.patch)
            except OSError:
                self._circles[champion_id] = None
        return self._circles[champion_id]


def walkable_mask(darkness: np.ndarray) -> np.ndarray:
    """Проходимая область: всё, что не чёрная зона карты, кроме полей кадра."""
    mask = ~darkness.copy()
    mask[:EDGE_MARGIN] = mask[-EDGE_MARGIN:] = False
    mask[:, :EDGE_MARGIN] = mask[:, -EDGE_MARGIN:] = False
    return mask


def _sample_position(
    rng: np.random.Generator,
    walkable: np.ndarray,
    placed: list[PlacedChampion],
) -> tuple[float, float]:
    if placed and rng.random() < CLUSTER_PROBABILITY:
        anchor = placed[rng.integers(len(placed))]
        for _ in range(20):
            radius = rng.uniform(*CLUSTER_RADIUS)
            angle = rng.uniform(0, 2 * np.pi)
            x = anchor.x + radius * np.cos(angle)
            y = anchor.y + radius * np.sin(angle)
            if (
                EDGE_MARGIN <= x < CANONICAL_SIDE - EDGE_MARGIN
                and EDGE_MARGIN <= y < CANONICAL_SIDE - EDGE_MARGIN
                and walkable[int(y), int(x)]
            ):
                return float(x), float(y)
    while True:
        x = rng.integers(EDGE_MARGIN, CANONICAL_SIDE - EDGE_MARGIN)
        y = rng.integers(EDGE_MARGIN, CANONICAL_SIDE - EDGE_MARGIN)
        if walkable[y, x]:
            return float(x) + rng.random(), float(y) + rng.random()


def _column_direction(
    rng: np.random.Generator, start: tuple[int, int], walkable: np.ndarray, count: int
) -> tuple[float, float] | None:
    """Направление, вдоль которого колонна укладывается в коридор целиком."""
    fitting = []
    for dx, dy in COLUMN_DIRECTIONS:
        length = (dx * dx + dy * dy) ** 0.5
        step = (dx / length * MINION_SPACING, dy / length * MINION_SPACING)
        if all(
            0 <= round(start[0] + step[0] * i) < CANONICAL_SIDE
            and 0 <= round(start[1] + step[1] * i) < CANONICAL_SIDE
            and walkable[round(start[1] + step[1] * i), round(start[0] + step[0] * i)]
            for i in range(count)
        ):
            fitting.append((dx, dy))
    if not fitting:
        return None
    return fitting[rng.integers(len(fitting))]


def random_scene(
    rng: np.random.Generator,
    roster: list[str],
    structures: list[dict],
    map_objects: list[dict] | None = None,
) -> Scene:
    """Случайная сцена; все оси рандомизации описаны в докстринге модуля."""
    variant = MAP_VARIANTS[rng.integers(len(MAP_VARIANTS))]
    ally_side = "southwest" if rng.random() < EVEN_ODDS else "northeast"
    lateness = float(rng.random())

    turret_states = []
    for structure in structures:
        if structure["type"] != "turret":
            turret_states.append("")
            continue
        if lateness > PLATES_UNTIL and rng.random() < lateness * DEMOLISH_CHANCE_AT_LATE:
            turret_states.append("destroyed")
        elif lateness <= PLATES_UNTIL:
            turret_states.append(f"turret_{rng.integers(1, 6)}plate")
        else:
            turret_states.append("tower")

    picked = list(rng.choice(len(roster), size=min(10, len(roster)), replace=False))
    visible_count = int(rng.integers(2, 11))
    champions: list[PlacedChampion] = []
    for order, roster_index in enumerate(picked[:visible_count]):
        moving = rng.random() < MOVING_PROBABILITY
        champions.append(
            PlacedChampion(
                champion_id=roster[roster_index],
                x=0.0,
                y=0.0,
                ally=order < visible_count / 2,
                moving=moving,
                blur_sigma=float(rng.uniform(0.4, BLUR_SIGMA_MAX)) if moving else 0.0,
            )
        )

    # Колонны без позиций: их досаживает place_entities, где есть проходимость.
    columns = [
        MinionColumn(
            x=0,
            y=0,
            direction=(1.0, 0.0),
            ally=bool(rng.random() < EVEN_ODDS),
            count=int(rng.integers(*MINION_COUNT)),
        )
        for _ in range(rng.integers(0, 7))
    ]
    wards = tuple(
        (
            int(rng.integers(EDGE_MARGIN, CANONICAL_SIDE - EDGE_MARGIN)),
            int(rng.integers(EDGE_MARGIN, CANONICAL_SIDE - EDGE_MARGIN)),
            "ward_pink" if rng.random() < PINK_WARD_SHARE else "ward_green",
        )
        for _ in range(rng.integers(0, WARD_COUNT_MAX + 1))
    )
    share = rng.uniform(*MAP_OBJECT_VISIBLE_SHARE)
    visible_objects = tuple(bool(rng.random() < share) for _ in range(len(map_objects or [])))

    return Scene(
        variant=variant,
        ally_side=ally_side,
        lateness=lateness,
        champions=tuple(champions),
        minion_columns=tuple(columns),
        ward_sights=wards,
        turret_states=tuple(turret_states),
        visible_map_objects=visible_objects,
    )


def _place_column(
    rng: np.random.Generator, column: MinionColumn, walkable: np.ndarray
) -> MinionColumn | None:
    """Сажает колонну целиком в проходимый коридор; иначе отбрасывает её."""
    for _ in range(30):
        start = (
            int(rng.integers(EDGE_MARGIN, CANONICAL_SIDE - EDGE_MARGIN)),
            int(rng.integers(EDGE_MARGIN, CANONICAL_SIDE - EDGE_MARGIN)),
        )
        if not walkable[start[1], start[0]]:
            continue
        direction = _column_direction(rng, start, walkable, column.count)
        if direction is not None:
            return MinionColumn(start[0], start[1], direction, column.ally, column.count)
    return None


def place_entities(rng: np.random.Generator, scene: Scene, walkable: np.ndarray) -> Scene:
    """Досаживает чемпионов и колонны миньонов по проходимой области."""
    placed: list[PlacedChampion] = []
    for champion in scene.champions:
        x, y = _sample_position(rng, walkable, placed)
        placed.append(
            PlacedChampion(
                champion.champion_id, x, y, champion.ally, champion.moving, champion.blur_sigma
            )
        )
    columns = tuple(
        seated
        for seated in (_place_column(rng, column, walkable) for column in scene.minion_columns)
        if seated is not None
    )
    return Scene(
        scene.variant,
        scene.ally_side,
        scene.lateness,
        tuple(placed),
        columns,
        scene.ward_sights,
        scene.turret_states,
        scene.visible_map_objects,
    )


STRUCTURE_ICON_SIZE = {
    "turret": STRUCTURE_SIDE,
    "nexus_turret": 16,
    "inhibitor": 14,
    "nexus": 18,
}


def _place_subpixel(
    canvas: np.ndarray, icon_bgra: np.ndarray, x: float, y: float, side: int
) -> None:
    """Посадка значка с дробным сдвигом: аффинное уменьшение в окно side+1."""
    window = side + 1
    fx, fy = x - int(x), y - int(y)
    scale = icon_bgra.shape[0] / side
    coefficients = (scale, 0.0, -fx * scale, 0.0, scale, -fy * scale)
    shifted = Image.fromarray(icon_bgra[..., [2, 1, 0, 3]], "RGBA").transform(
        (window, window), Image.AFFINE, coefficients, resample=Image.BILINEAR
    )
    icon_np = np.asarray(shifted).astype(np.float64)
    alpha = icon_np[..., 3:4] / 255.0
    bgr = icon_np[..., [2, 1, 0]]

    half = side // 2
    x0, y0 = int(x) - half, int(y) - half
    if x0 < 0 or y0 < 0 or x0 + window > canvas.shape[1] or y0 + window > canvas.shape[0]:
        return
    region = canvas[y0 : y0 + window, x0 : x0 + window]
    region[:] = bgr * alpha + region * (1.0 - alpha)


def render_scene(scene: Scene, assets: AssetLibrary) -> tuple[np.ndarray, dict]:
    """Сцена → (кадр BGR uint8, разметка). Разметка бесплатна по построению."""
    ally_structures = [s for s in assets.structures if s["side"] == scene.ally_side]
    living = [
        s
        for s, state in zip(assets.structures, scene.turret_states, strict=True)
        if state != "destroyed"
    ]

    sight = [(s["x"], s["y"]) for s in ally_structures if s in living]
    sight += [(int(c.x), int(c.y)) for c in scene.champions if c.ally]
    sight += [(x, y) for x, y, _ in scene.ward_sights]
    canvas = compose_background(
        assets.layer(scene.variant),
        CANONICAL_SIDE,
        visibility_mask(CANONICAL_SIDE, sight),
        assets.darkness,
    )

    for structure, state in zip(assets.structures, scene.turret_states, strict=True):
        if state == "destroyed":
            continue
        icon_name = (
            state
            if structure["type"] == "turret"
            else {
                "nexus_turret": "tower",
                "inhibitor": "inhibitor",
                "nexus": "nexus",
            }[structure["type"]]
        )
        tint = (
            STRUCTURE_ALLY_BGR if structure["side"] == scene.ally_side else STRUCTURE_ENEMY_BGR
        )
        place_icon(
            canvas,
            tinted_icon(assets.icon(icon_name), tint),
            (structure["x"], structure["y"]),
            STRUCTURE_ICON_SIZE[structure["type"]],
        )

    for map_object, visible in zip(assets.map_objects, scene.visible_map_objects, strict=True):
        if visible:
            draw_map_object(canvas, map_object["type"], map_object["x"], map_object["y"])

    for x, y, kind in scene.ward_sights:
        draw_map_object(canvas, kind, x, y)

    dot_ally = tinted_icon(assets.icon("minionmapcircle"), MINION_ALLY_BGR)
    dot_enemy = tinted_icon(assets.icon("minionmapcircle"), MINION_ENEMY_BGR)
    for column in scene.minion_columns:
        draw_minion_column(
            canvas,
            dot_ally if column.ally else dot_enemy,
            (column.x, column.y),
            column.direction,
            column.count,
        )

    labels = []
    for champion in scene.champions:
        circle = assets.circle(champion.champion_id)
        if circle is None:
            continue
        ring = ALLY_RING_BGR if champion.ally else ENEMY_RING_BGR
        icon = ringed_icon(circle, ring)
        if champion.blur_sigma > 0:
            blurred = gaussian_blur(icon[..., :3].astype(np.float64), champion.blur_sigma)
            icon = icon.copy()
            icon[..., :3] = np.clip(blurred + 0.5, 0, 255).astype(np.uint8)
        _place_subpixel(canvas, icon, champion.x, champion.y, ICON_SIDE)
        labels.append(
            {
                "championId": champion.champion_id,
                "x": round(champion.x, 2),
                "y": round(champion.y, 2),
                "affiliation": "Ally" if champion.ally else "Enemy",
                "moving": champion.moving,
            }
        )

    metadata = {
        "variant": scene.variant,
        "allySide": scene.ally_side,
        "lateness": round(scene.lateness, 3),
        "champions": labels,
    }
    return to_uint8_bgr(canvas), metadata
