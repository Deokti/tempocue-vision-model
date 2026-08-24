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
from tcvm.formats import ReplayFrame, bgra_to_rgb, load_corpus
from tcvm.matching import ICON_SIDE, INNER_RADIUS, circular_mask
from tcvm.render import RenderParams, render_icon

DEFAULT_CORPUS = Path(
    r"C:\Users\deokn\.codex\worktrees\739a\PROJECT\tests\TempoCue.Vision.Tests\ReplayCorpus"
)
PATCH_VERSION = "16.16.1"

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
# Насколько далеко от существующей метки искать уточнённый центр.
ALIGN_RADIUS = 5
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
    kind: str  # "добавить" | "сдвинуть" | "проверить"
    shift: float = 0.0  # для сдвига: на сколько уехала метка
    shift_x: float = 0.0  # и в какую сторону — нужно для листа «до и после»
    shift_y: float = 0.0


@dataclass
class FrameProposals:
    """Предложения по одному кадру и всё, что нужно для их отрисовки."""

    frame: ReplayFrame
    added: list[Proposal] = field(default_factory=list)
    moved: list[Proposal] = field(default_factory=list)
    review: list[Proposal] = field(default_factory=list)


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

    for reference in frame.references:
        champion = reference.champion_id
        template = icon_of(champion, patch, icons)
        if template is None:
            continue
        scores = ncc_map(frame.pixels, template, INNER_MASK)
        side = "Ally" if reference.affiliation == 0 else "Enemy"

        if champion in labelled:
            x, y, score = best_near(scores, labelled[champion], ALIGN_RADIUS)
            dx = x - labelled[champion][0]
            dy = y - labelled[champion][1]
            shift = float(np.hypot(dx, dy))
            if shift >= WORTH_MOVING and score >= MOVE_SCORE:
                result.moved.append(
                    Proposal(frame.name, champion, side, x, y, score, "сдвинуть", shift, dx, dy)
                )
            continue

        x, y, score = global_best(scores)
        if any(np.hypot(x - tx, y - ty) < SAME_SPOT for tx, ty in taken):
            continue
        if score >= ADD_SCORE:
            result.added.append(Proposal(frame.name, champion, side, x, y, score, "добавить"))
            taken.append((x, y))
        elif score >= REVIEW_SCORE:
            result.review.append(Proposal(frame.name, champion, side, x, y, score, "проверить"))
    return result


def draw_frame_sheet(found: FrameProposals, path: Path) -> None:
    """Кадр целиком: зелёное — есть в разметке, жёлтое — добавить, оранжевое — проверить."""
    zoom = 2
    image = Image.fromarray(bgra_to_rgb(found.frame.pixels), "RGB").resize(
        (320 * zoom, 320 * zoom), Image.NEAREST
    )
    draw = ImageDraw.Draw(image)
    for region in found.frame.labels.champions if found.frame.labels else ():
        x = (region.x + region.width / 2) * zoom
        y = (region.y + region.height / 2) * zoom
        draw.rectangle([x - 13, y - 13, x + 13, y + 13], outline=(0, 255, 70))
    for proposal, colour in (
        *((p, (255, 230, 60)) for p in found.added),
        *((p, (255, 140, 40)) for p in found.review),
    ):
        x, y = round(proposal.x * zoom), round(proposal.y * zoom)
        draw.ellipse([x - 14, y - 14, x + 14, y + 14], outline=colour, width=2)
        draw.text((x + 15, y - 16), f"{proposal.champion} {proposal.score:.2f}", fill=colour)
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
        colour = (255, 230, 60) if proposal.kind == "добавить" else (255, 140, 40)
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
                "review": [_as_dict(p) for p in found.review],
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
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--out", type=Path, default=Path("out/label-proposals"))
    args = parser.parse_args()
    (args.out / "frames").mkdir(parents=True, exist_ok=True)

    patch = patch_of(PATCH_VERSION)
    icons: dict[str, np.ndarray | None] = {}
    all_found: list[FrameProposals] = []
    pixels_by_frame: dict[str, np.ndarray] = {}

    print("  кадр                                      есть  добавить  сдвинуть  проверить")
    for frame in load_corpus(args.corpus):
        found = examine_frame(frame, patch, icons)
        all_found.append(found)
        pixels_by_frame[frame.name] = bgra_to_rgb(frame.pixels)
        draw_frame_sheet(found, args.out / "frames" / f"{frame.name}.png")
        existing = len(frame.labels.champions) if frame.labels else 0
        print(
            f"  {frame.name:40} {existing:5} {len(found.added):9} "
            f"{len(found.moved):9} {len(found.review):10}"
        )

    added = [p for f in all_found for p in f.added]
    moved = [p for f in all_found for p in f.moved]
    review = [p for f in all_found for p in f.review]
    draw_gallery(added + review, pixels_by_frame, args.out / "gallery.png")
    draw_moves(sorted(moved, key=lambda p: -p.shift), pixels_by_frame, args.out / "moves.png")
    write_proposals(all_found, args.out / "proposals.json")

    print()
    print(f"Добавить чемпионов: {len(added)}; отвергнуто по низкому совпадению: {len(review)}")
    if moved:
        shifts = [p.shift for p in moved]
        print(
            f"Сдвинуть центров: {len(moved)}; сдвиг медиана {np.median(shifts):.1f} px, "
            f"максимум {max(shifts):.1f} px"
        )
    print(f"Предложения: {(args.out / 'proposals.json').resolve()}")
    print(f"Контрольный лист: {(args.out / 'gallery.png').resolve()}")
    print(f"Сдвиги «до и после»: {(args.out / 'moves.png').resolve()}")
    print("Корпус не изменён: применение — отдельный шаг после проверки глазами.")


if __name__ == "__main__":
    main()
