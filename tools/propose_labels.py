"""Предложения по разметке корпуса: сплошной поиск значков по кадру.

Разметка корпуса неполна и неточна. В каждом кадре записаны эталоны всех
десяти чемпионов матча, а размечено от двух до восьми; сам формат это
признаёт полем `allowedMissedChampions`. Вдобавок четверть меток стоит дальше
трёх пикселей от настоящего центра значка (`tools/measure_label_jitter.py`).
Мерить модель такой лентой нельзя: три подряд «потолка» ступени 3 оказались
именно в измерении, а не в модели.

**Модель здесь не участвует намеренно.** Строить метки детектором, а потом им
же его судить — замкнутый круг. Поиск идёт сплошным сравнением: значок
рисуется из circle-арта игры (источник проверен на ступени 1: медиана остатка
0,033 на чистых вырезах) и сравнивается с каждым положением кадра по
маскированной нормированной кросс-корреляции. Свёртка вместо перебора делает
это за секунды.

    .venv/Scripts/python tools/propose_labels.py

Инструмент **ничего не переписывает**. Он кладёт в out/label-proposals
предложения и картинки для проверки глазами: `gallery.png` — вырезы вокруг
каждого предложения, `frames/` — кадры целиком с пометками. Применять
предложения к корпусу — отдельный шаг и отдельное решение человека.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.nn import functional as torch_functional

from tcvm.cdragon import base_circle_bgra, patch_of
from tcvm.formats import (
    LOOKS_LIKE_MAP,
    ReplayFrame,
    bgra_to_rgb,
    default_corpus_dir,
    load_corpus,
    map_likeness,
)
from tcvm.matching import (
    ICON_SIDE,
    INNER_RADIUS,
    MIN_VISIBLE_SHARE,
    circular_mask,
    crop_centered,
    masked_ncc,
    visible_mask,
)
from tcvm.render import RenderParams, render_icon
from tcvm.synthesis import MAP_PLACEMENT, load_map_layer, map_rect, place_map_texture

DEFAULT_CORPUS = None  # ищется при запуске: см. formats.default_corpus_dir
PATCH_VERSION = "16.16.1"
# Сторона канонического кадра приложения.
CANONICAL_FRAME = 320

# Наивная отрисовка: поиску хватает формы значка, подбор параметров не нужен
# (тот же набор, с которого начинает tools/fit_render.py).
NAIVE = RenderParams(1.0, ICON_SIDE, "box", "srgb", 0.0, "bilinear", 0.0, 0.0)
INNER_MASK = circular_mask(ICON_SIDE, INNER_RADIUS)

# Пороги совпадения. Замер по 120 парам «кадр — чемпион матча»: у размеченных
# чемпионов максимум по кадру имеет медиану 0,926, у неразмеченных — 0,629.
# Выше ADD_SCORE неразмеченных почти нет, и те, что есть, — настоящие
# пропуски разметчика; ниже REVIEW_SCORE смотреть нечего.
ADD_SCORE = 0.80
REVIEW_SCORE = 0.55
# Сдвигать метку можно только при уверенном совпадении. Контрольный лист
# «до и после» показал прямо: ниже этого значка на месте нет (закрыт другим
# значком, размыт движением, срезан краем), и уточнённый центр уезжает в
# случайное место. Такую метку лучше оставить как есть, чем испортить.
MOVE_SCORE = 0.80
# Ближе этого значки считаются перекрывающимися и сравниваются по видимой части.
CROWD_DISTANCE = 30.0
# Насколько далеко от существующей метки искать уточнённый центр. Первая
# версия искала в пяти пикселях, и этого не хватило: разбор потерь детектора
# показал метки, уехавшие на 7-12 px (Ziggs в трёх кадрах, Lulu на 12,4).
# Инструмент их просто не видел, а модель за них наказывалась.
ALIGN_RADIUS = 12
# Сдвиг метки, ниже которого её не трогаем: формат хранит угол коробки 25x25
# целыми числами, поэтому представимые центры отстоят друг от друга на пиксель,
# и половина пикселя — предел точности самой записи, а не разметки.
WORTH_MOVING = 1.5
# Порог вырожденной параболы при субпиксельном уточнении вершины.
FLAT_PEAK = 1e-9
# Два значка ближе этого расстояния — одно и то же место: в корпусе самые
# тесные чемпионы стоят в 5,4 px друг от друга, поэтому радиус меньше.
SAME_SPOT = 5.0
# Числовая мелочь: делить на нулевую дисперсию нельзя.
VARIANCE_FLOOR = 1e-9
# Наименьшая доля диска, при которой срезанному краем значку ещё можно верить.
MIN_VISIBLE = 0.5
# Вырез вокруг предложения в контрольном листе и его увеличение.
CROP_SIDE = 33
CROP_ZOOM = 4
GALLERY_COLUMNS = 8


@dataclass(frozen=True)
class Proposal:
    """Одно предложение по одному чемпиону в одном кадре."""

    frame: str
    champion: str
    affiliation: str
    x: float  # центр значка
    y: float
    score: float
    kind: str  # "добавить" | "сдвинуть" | "отвергнуто"
    shift: float = 0.0  # для сдвига: на сколько уехала метка
    shift_x: float = 0.0  # и в какую сторону — нужно для листа «до и после»
    shift_y: float = 0.0


@dataclass
class FrameProposals:
    """Предложения по одному кадру и всё, что нужно для их отрисовки."""

    frame: ReplayFrame
    added: list[Proposal] = field(default_factory=list)
    moved: list[Proposal] = field(default_factory=list)
    rejected: list[Proposal] = field(default_factory=list)


def ncc_map(frame_bgra: np.ndarray, template_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Совпадение шаблона с каждым положением кадра, посчитанное свёрткой.

    Та же величина, что даёт `matching.masked_ncc`, но сразу для всех
    положений: числитель и обе суммы под маской — три свёртки. Сверено с
    `masked_ncc` на трёхстах случайных положениях, расхождение 9e-16.
    """
    frame = torch.from_numpy(frame_bgra[..., :3].astype(np.float64).transpose(2, 0, 1))
    template = np.ascontiguousarray(template_bgr.astype(np.float64).transpose(2, 0, 1))
    plane = mask.astype(np.float64)
    count = 3 * plane.sum()

    masked = template * plane
    centered = (masked - masked.sum() / count * plane) * plane
    norm = np.sqrt((centered**2).sum())

    kernel = torch.from_numpy(centered).unsqueeze(0)
    ones = torch.from_numpy(np.repeat(plane[None], 3, 0)).unsqueeze(0)
    batch = frame.unsqueeze(0)

    product = torch_functional.conv2d(batch, kernel)[0, 0]
    total = torch_functional.conv2d(batch, ones)[0, 0]
    squares = torch_functional.conv2d(batch**2, ones)[0, 0]
    variance = torch.clamp(squares - total**2 / count, min=VARIANCE_FLOOR)
    return (product / (torch.sqrt(variance) * norm)).numpy()


def best_near(
    scores: np.ndarray, center: tuple[float, float], radius: int
) -> tuple[float, float, float]:
    """Лучшее совпадение в окрестности точки; координаты — центр значка."""
    half = ICON_SIDE // 2
    cx, cy = round(center[0]) - half, round(center[1]) - half
    left, top = max(cx - radius, 0), max(cy - radius, 0)
    right = min(cx + radius + 1, scores.shape[1])
    bottom = min(cy + radius + 1, scores.shape[0])
    if left >= right or top >= bottom:
        return center[0], center[1], -1.0
    window = scores[top:bottom, left:right]
    index = np.unravel_index(window.argmax(), window.shape)
    row, column = int(top + index[0]), int(left + index[1])
    dx, dy = subpixel_peak(scores, row, column)
    return float(column + half + dx), float(row + half + dy), float(window.max())


def subpixel_peak(scores: np.ndarray, row: int, column: int) -> tuple[float, float]:
    """Уточнение вершины по параболе через соседей — как при чтении карты центров."""
    offsets = []
    for index, limit, along_row in (
        (column, scores.shape[1], True),
        (row, scores.shape[0], False),
    ):
        if index <= 0 or index >= limit - 1:
            offsets.append(0.0)
            continue
        low = scores[row, index - 1] if along_row else scores[row - 1, column]
        middle = scores[row, column]
        high = scores[row, index + 1] if along_row else scores[row + 1, column]
        denominator = low - 2 * middle + high
        shift = 0.0 if abs(denominator) < FLAT_PEAK else 0.5 * (low - high) / denominator
        offsets.append(float(np.clip(shift, -0.5, 0.5)))
    return offsets[0], offsets[1]


def global_best(scores: np.ndarray) -> tuple[float, float, float]:
    half = ICON_SIDE // 2
    row, column = np.unravel_index(scores.argmax(), scores.shape)
    dx, dy = subpixel_peak(scores, int(row), int(column))
    return float(column + half + dx), float(row + half + dy), float(scores.max())


def clipped_best(
    frame_bgra: np.ndarray,
    template_bgr: np.ndarray,
    bounds: tuple[int, int, int, int],
    near: tuple[float, float] | None = None,
) -> tuple[float, float, float, float]:
    """Лучшее совпадение у края карты, где значок срезан её границей.

    Сравнение по полному кругу там проваливается: часть круга приходится на
    рамку интерфейса, а не на значок. У фонтана так стоят чемпионы после
    смерти и возврата — целый класс, который иначе выпадает из разметки.
    Возвращает совпадение, центр и долю видимого диска. Параметр near
    ограничивает поиск окрестностью существующей метки.
    """
    left, top, right, bottom = bounds
    half = ICON_SIDE // 2
    if near is not None:
        left = max(left, round(near[0]) - ALIGN_RADIUS)
        right = min(right, round(near[0]) + ALIGN_RADIUS + 1)
        top = max(top, round(near[1]) - ALIGN_RADIUS)
        bottom = min(bottom, round(near[1]) + ALIGN_RADIUS + 1)
    rows, columns = np.mgrid[0:ICON_SIDE, 0:ICON_SIDE]
    full = float(INNER_MASK.sum())
    masks: dict[tuple[int, int, int, int], tuple[np.ndarray, np.ndarray, float]] = {}
    best = (-2.0, 0.0, 0.0, 0.0)

    for cy in range(top, bottom):
        for cx in range(left, right):
            cuts = (
                max(0, left - (cx - half)),
                max(0, (cx + half) - (right - 1)),
                max(0, top - (cy - half)),
                max(0, (cy + half) - (bottom - 1)),
            )
            if not any(cuts):
                continue  # круг целиком внутри карты — это дело быстрого прохода
            if cuts not in masks:
                inside = (
                    (columns >= cuts[0])
                    & (columns < ICON_SIDE - cuts[1])
                    & (rows >= cuts[2])
                    & (rows < ICON_SIDE - cuts[3])
                )
                mask = INNER_MASK & inside
                share = mask.sum() / full
                if share < MIN_VISIBLE:
                    masks[cuts] = (mask, np.zeros(0), 0.0)
                else:
                    values = template_bgr[mask][:, :3].astype(np.float64).ravel()
                    masks[cuts] = (mask, values - values.mean(), share)
            mask, wanted, share = masks[cuts]
            if share < MIN_VISIBLE:
                continue
            window = frame_bgra[cy - half : cy + half + 1, cx - half : cx + half + 1]
            if window.shape[:2] != (ICON_SIDE, ICON_SIDE):
                continue
            seen = window[mask][:, :3].astype(np.float64).ravel()
            seen = seen - seen.mean()
            denominator = np.linalg.norm(seen) * np.linalg.norm(wanted)
            score = float(seen @ wanted / denominator) if denominator else 0.0
            if score > best[0]:
                best = (score, float(cx), float(cy), share)
    return best


def icon_of(
    champion: str, patch: str, cache: dict[str, np.ndarray | None]
) -> np.ndarray | None:
    if champion not in cache:
        try:
            cache[champion] = render_icon(base_circle_bgra(champion, patch), NAIVE)
        except (OSError, ValueError) as error:
            print(f"  нет circle-иконки у {champion}: {error}")
            cache[champion] = None
    return cache[champion]


def best_unoccluded(
    frame_bgra: np.ndarray,
    template: np.ndarray,
    center: tuple[float, float],
    neighbours: list[tuple[float, float]],
) -> tuple[float, float, float]:
    """Уточнение центра с учётом того, что значок закрыт соседями."""
    full = float(circular_mask(ICON_SIDE, INNER_RADIUS).sum())
    best = (center[0], center[1], -2.0)
    for dy in range(-ALIGN_RADIUS, ALIGN_RADIUS + 1):
        for dx in range(-ALIGN_RADIUS, ALIGN_RADIUS + 1):
            x, y = round(center[0]) + dx, round(center[1]) + dy
            mask = visible_mask((x, y), neighbours)
            if mask.sum() / full < MIN_VISIBLE_SHARE:
                continue
            try:
                score = masked_ncc(crop_centered(frame_bgra, x, y), template, mask)
            except ValueError:
                continue
            if score > best[2]:
                best = (float(x), float(y), score)
    return best


def refine_label(
    frame: ReplayFrame,
    template: np.ndarray,
    labelled: dict[str, tuple[float, float]],
    who: tuple[str, str, tuple[int, int, int, int]],
    scores: np.ndarray,
) -> Proposal | None:
    """Уточнение одной метки тремя способами по возрастанию цены.

    Быстрый проход свёрткой; если значок в толпе — сравнение по видимой части;
    если у края карты — по той части диска, что внутри карты. Побеждает
    наибольшее совпадение, и метка двигается только при уверенном.
    """
    champion, side, bounds = who
    center = labelled[champion]
    x, y, score = best_near(scores, center, ALIGN_RADIUS)

    neighbours = [
        point
        for name, point in labelled.items()
        if name != champion
        and np.hypot(point[0] - center[0], point[1] - center[1]) < CROWD_DISTANCE
    ]
    if neighbours:
        crowded = best_unoccluded(frame.pixels, template, center, neighbours)
        if crowded[2] > score:
            x, y, score = crowded[0], crowded[1], crowded[2]
    if score < MOVE_SCORE:
        edge = clipped_best(frame.pixels, template, bounds, center)
        if edge[0] > score:
            score, x, y = edge[0], edge[1], edge[2]

    dx, dy = x - center[0], y - center[1]
    if float(np.hypot(dx, dy)) < WORTH_MOVING or score < MOVE_SCORE:
        return None
    if any(
        np.hypot(x - point[0], y - point[1]) < SAME_SPOT
        for name, point in labelled.items()
        if name != champion
    ):
        return None  # уточнение уехало на чужой значок
    return Proposal(
        frame.name, champion, side, x, y, score, "сдвинуть", float(np.hypot(dx, dy)), dx, dy
    )


def second_pass(
    frame: ReplayFrame,
    unsure: list[tuple[str, str, np.ndarray, float, float, float]],
    taken: list[tuple[float, float]],
    result: FrameProposals,
) -> None:
    """Второй заход по тем, кого не взял первый: значок закрыт соседом.

    Первый проход находит открытые значки. Закрытые он пропускает, потому что
    сравнение по полному диску считает чужие пиксели. Но после первого прохода
    соседи уже известны, и закрытый значок можно сравнить по видимой части —
    тем же приёмом, каким выправлялись метки корпуса.

    Порог для второго захода тот же: доверять уточнению ниже него нельзя,
    проверка глазами показала, что там оно уезжает на чужой значок.
    """
    for champion, side, template, x, y, first in unsure:
        neighbours = [
            point for point in taken if np.hypot(point[0] - x, point[1] - y) < CROWD_DISTANCE
        ]
        score, best_x, best_y = first, x, y
        if neighbours:
            found_x, found_y, found = best_unoccluded(
                frame.pixels, template, (x, y), neighbours
            )
            if found > score:
                score, best_x, best_y = found, found_x, found_y
        if score >= ADD_SCORE and not any(
            np.hypot(best_x - tx, best_y - ty) < SAME_SPOT for tx, ty in taken
        ):
            result.added.append(
                Proposal(frame.name, champion, side, best_x, best_y, score, "добавить")
            )
            taken.append((best_x, best_y))
        elif score >= REVIEW_SCORE:
            result.rejected.append(
                Proposal(frame.name, champion, side, best_x, best_y, score, "отвергнуто")
            )


def examine_frame(
    frame: ReplayFrame, patch: str, icons: dict[str, np.ndarray | None]
) -> FrameProposals:
    """Ищет каждого чемпиона матча по всему кадру и раскладывает по решениям."""
    result = FrameProposals(frame=frame)
    regions = frame.labels.champions if frame.labels else ()
    labelled = {
        region.champion_id: (region.x + region.width / 2, region.y + region.height / 2)
        for region in regions
    }
    taken = list(labelled.values())
    unsure: list[tuple[str, str, np.ndarray, float, float, float]] = []
    bounds = map_rect(CANONICAL_FRAME, MAP_PLACEMENT)

    for reference in frame.references:
        champion = reference.champion_id
        template = icon_of(champion, patch, icons)
        if template is None:
            continue
        scores = ncc_map(frame.pixels, template, INNER_MASK)
        side = "Ally" if reference.affiliation == 0 else "Enemy"

        if champion in labelled:
            move = refine_label(frame, template, labelled, (champion, side, bounds), scores)
            if move is not None:
                result.moved.append(move)
            continue

        x, y, score = global_best(scores)
        if score < ADD_SCORE:
            edge_score, edge_x, edge_y, _ = clipped_best(frame.pixels, template, bounds)
            if edge_score > score:
                x, y, score = edge_x, edge_y, edge_score
        if any(np.hypot(x - tx, y - ty) < SAME_SPOT for tx, ty in taken):
            continue
        if score >= ADD_SCORE:
            result.added.append(Proposal(frame.name, champion, side, x, y, score, "добавить"))
            taken.append((x, y))
        else:
            unsure.append((champion, side, template, x, y, score))

    second_pass(frame, unsure, taken, result)
    return result


def draw_frame_sheet(found: FrameProposals, path: Path) -> None:
    """Кадр целиком: зелёное — есть в разметке, жёлтое — добавить, голубое — сдвинуть.

    Отвергнутые кандидаты здесь не рисуются намеренно. Их «положение» — это
    просто лучшее место, какое нашлось при заведомо низком совпадении; рисовать
    его с именем чемпиона означало бы показывать чемпиона там, где его нет.
    """
    zoom = 2
    image = Image.fromarray(bgra_to_rgb(found.frame.pixels), "RGB").resize(
        (320 * zoom, 320 * zoom), Image.NEAREST
    )
    draw = ImageDraw.Draw(image)
    for region in found.frame.labels.champions if found.frame.labels else ():
        x = (region.x + region.width / 2) * zoom
        y = (region.y + region.height / 2) * zoom
        draw.rectangle([x - 13, y - 13, x + 13, y + 13], outline=(0, 255, 70))
    for proposal in found.added:
        x, y = round(proposal.x * zoom), round(proposal.y * zoom)
        draw.ellipse([x - 14, y - 14, x + 14, y + 14], outline=(255, 230, 60), width=2)
        draw.text(
            (x + 15, y - 16), f"{proposal.champion} {proposal.score:.2f}", fill=(255, 230, 60)
        )
    for proposal in found.moved:
        x, y = round(proposal.x * zoom), round(proposal.y * zoom)
        was_x = round((proposal.x - proposal.shift_x) * zoom)
        was_y = round((proposal.y - proposal.shift_y) * zoom)
        draw.line([was_x, was_y, x, y], fill=(120, 200, 255), width=2)
        draw.ellipse([x - 4, y - 4, x + 4, y + 4], outline=(120, 200, 255), width=2)
    image.save(path)


def draw_gallery(proposals: list[Proposal], frames: dict[str, np.ndarray], path: Path) -> None:
    """Вырезы вокруг каждого предложения — то, что человек смотрит глазами."""
    if not proposals:
        return
    rows = (len(proposals) + GALLERY_COLUMNS - 1) // GALLERY_COLUMNS
    cell = CROP_SIDE * CROP_ZOOM
    sheet = Image.new("RGB", (GALLERY_COLUMNS * (cell + 8), rows * (cell + 30)), (24, 24, 24))
    draw = ImageDraw.Draw(sheet)
    half = CROP_SIDE // 2
    for index, proposal in enumerate(proposals):
        pixels = frames[proposal.frame]
        left = max(0, round(proposal.x) - half)
        top = max(0, round(proposal.y) - half)
        crop = pixels[top : top + CROP_SIDE, left : left + CROP_SIDE]
        if crop.shape[0] < CROP_SIDE or crop.shape[1] < CROP_SIDE:
            padded = np.zeros((CROP_SIDE, CROP_SIDE, 3), np.uint8)
            padded[: crop.shape[0], : crop.shape[1]] = crop
            crop = padded
        x0 = (index % GALLERY_COLUMNS) * (cell + 8)
        y0 = (index // GALLERY_COLUMNS) * (cell + 30)
        sheet.paste(Image.fromarray(crop).resize((cell, cell), Image.NEAREST), (x0, y0 + 26))
        colour = (255, 230, 60) if proposal.kind == "добавить" else (150, 150, 150)
        draw.text((x0, y0), f"{proposal.champion} {proposal.score:.2f}", fill=colour)
        draw.text((x0, y0 + 13), f"{proposal.kind} {proposal.frame[:18]}", fill=(170, 170, 170))
        middle = cell // 2
        draw.line(
            [x0 + middle - 5, y0 + 26 + middle, x0 + middle + 5, y0 + 26 + middle], fill=colour
        )
        draw.line(
            [x0 + middle, y0 + 26 + middle - 5, x0 + middle, y0 + 26 + middle + 5], fill=colour
        )
    sheet.save(path)


def draw_moves(proposals: list[Proposal], frames: dict[str, np.ndarray], path: Path) -> None:
    """Сдвиги центров «до и после»: слева вырез по метке, справа по предложению.

    Смотреть надо на то, стоит ли значок ровно посередине выреза. Если справа
    ровнее, чем слева, — предложение верное.
    """
    if not proposals:
        return
    pairs = 4  # пар «до-после» в строке
    cell = CROP_SIDE * CROP_ZOOM
    rows = (len(proposals) + pairs - 1) // pairs
    sheet = Image.new("RGB", (pairs * (cell * 2 + 14), rows * (cell + 30)), (24, 24, 24))
    draw = ImageDraw.Draw(sheet)
    half = CROP_SIDE // 2

    def crop_at(pixels: np.ndarray, cx: float, cy: float) -> Image.Image:
        left, top = max(0, round(cx) - half), max(0, round(cy) - half)
        piece = pixels[top : top + CROP_SIDE, left : left + CROP_SIDE]
        if piece.shape[0] < CROP_SIDE or piece.shape[1] < CROP_SIDE:
            padded = np.zeros((CROP_SIDE, CROP_SIDE, 3), np.uint8)
            padded[: piece.shape[0], : piece.shape[1]] = piece
            piece = padded
        return Image.fromarray(piece).resize((cell, cell), Image.NEAREST)

    for index, proposal in enumerate(proposals):
        pixels = frames[proposal.frame]
        x0 = (index % pairs) * (cell * 2 + 14)
        y0 = (index // pairs) * (cell + 30)
        old_x = proposal.x - proposal.shift_x
        old_y = proposal.y - proposal.shift_y
        sheet.paste(crop_at(pixels, old_x, old_y), (x0, y0 + 26))
        sheet.paste(crop_at(pixels, proposal.x, proposal.y), (x0 + cell + 6, y0 + 26))
        draw.text((x0, y0), f"{proposal.champion} {proposal.score:.2f}", fill=(120, 200, 255))
        draw.text(
            (x0, y0 + 13),
            f"сдвиг {proposal.shift:.1f} px   было | стало",
            fill=(170, 170, 170),
        )
        for offset, colour in ((0, (255, 90, 90)), (cell + 6, (0, 255, 120))):
            middle = cell // 2
            cx, cy = x0 + offset + middle, y0 + 26 + middle
            draw.line([cx - 6, cy, cx + 6, cy], fill=colour)
            draw.line([cx, cy - 6, cx, cy + 6], fill=colour)
    sheet.save(path)


def write_proposals(all_found: list[FrameProposals], path: Path) -> None:
    payload = {
        "schemaVersion": 1,
        "source": "tools/propose_labels.py",
        "method": "сплошной поиск circle-арта по кадру, маскированная NCC",
        "addScore": ADD_SCORE,
        "reviewScore": REVIEW_SCORE,
        "frames": [
            {
                "frame": found.frame.name,
                "added": [_as_dict(p) for p in found.added],
                "moved": [_as_dict(p) for p in found.moved],
                "rejected": [_as_dict(p) for p in found.rejected],
            }
            for found in all_found
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _as_dict(proposal: Proposal) -> dict:
    return {
        "championId": proposal.champion,
        "affiliation": proposal.affiliation,
        "centerX": round(proposal.x, 2),
        "centerY": round(proposal.y, 2),
        "boxX": round(proposal.x - ICON_SIDE / 2),
        "boxY": round(proposal.y - ICON_SIDE / 2),
        "score": round(proposal.score, 3),
        "shift": round(proposal.shift, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("out/label-proposals"))
    parser.add_argument("--map-dir", type=Path, default=Path("data/cdragon"))
    args = parser.parse_args()
    corpus = args.corpus or default_corpus_dir()
    (args.out / "frames").mkdir(parents=True, exist_ok=True)

    patch = patch_of(PATCH_VERSION)
    icons: dict[str, np.ndarray | None] = {}
    all_found: list[FrameProposals] = []
    pixels_by_frame: dict[str, np.ndarray] = {}

    print("  кадр                                      есть  добавить  сдвинуть  отвергнуто")
    texture = place_map_texture(
        load_map_layer(args.map_dir / patch / "map11", "base_baron1"),
        CANONICAL_FRAME,
        MAP_PLACEMENT,
    ).mean(axis=2)

    for frame in load_corpus(corpus):
        likeness = map_likeness(frame.pixels, texture)
        if likeness < LOOKS_LIKE_MAP:
            print(f"  {frame.name}: не похож на миникарту ({likeness:.2f}), пропущен")
            continue
        found = examine_frame(frame, patch, icons)
        all_found.append(found)
        pixels_by_frame[frame.name] = bgra_to_rgb(frame.pixels)
        draw_frame_sheet(found, args.out / "frames" / f"{frame.name}.png")
        existing = len(frame.labels.champions) if frame.labels else 0
        print(
            f"  {frame.name:40} {existing:5} {len(found.added):9} "
            f"{len(found.moved):9} {len(found.rejected):11}"
        )

    added = [p for f in all_found for p in f.added]
    moved = [p for f in all_found for p in f.moved]
    rejected = [p for f in all_found for p in f.rejected]
    draw_gallery(added, pixels_by_frame, args.out / "gallery.png")
    draw_gallery(
        sorted(rejected, key=lambda item: -item.score),
        pixels_by_frame,
        args.out / "rejected.png",
    )
    draw_moves(sorted(moved, key=lambda p: -p.shift), pixels_by_frame, args.out / "moves.png")
    write_proposals(all_found, args.out / "proposals.json")

    print()
    print(
        f"Добавить чемпионов: {len(added)}; отвергнуто по низкому совпадению: {len(rejected)}"
    )
    if moved:
        shifts = [p.shift for p in moved]
        print(
            f"Сдвинуть центров: {len(moved)}; сдвиг медиана {np.median(shifts):.1f} px, "
            f"максимум {max(shifts):.1f} px"
        )
    print(f"Предложения: {(args.out / 'proposals.json').resolve()}")
    print(f"Что добавить: {(args.out / 'gallery.png').resolve()}")
    print(f"Что отвергнуто и почему: {(args.out / 'rejected.png').resolve()}")
    print(f"Сдвиги «до и после»: {(args.out / 'moves.png').resolve()}")
    print("Корпус не изменён: применение — отдельный шаг после проверки глазами.")


if __name__ == "__main__":
    main()
