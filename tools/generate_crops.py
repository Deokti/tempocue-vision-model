"""Генератор вырезов для ступени «кто»: значки чемпионов и отрицательные примеры.

Вырезы берутся **из собранных кадров синтетики**, а не рисуются отдельно. Так
фон под значком, перекрытия соседними значками, туман и размытие движения
достаются даром и в точности совпадают с тем доменом, на котором работает
детектор. Отдельный рисовальщик вырезов пришлось бы согласовывать с
композитором вручную, и он бы разошёлся с ним при первой же правке.

    .venv/Scripts/python tools/generate_crops.py --frames 8000 --out out/crops-8k

Кладёт `crops.npy` (N, 64, 64, 3) uint8 и `labels.jsonl`. Один файл вместо
сотен тысяч PNG: так датасет читается за секунды и занимает втрое меньше места.

Отрицательные примеры — половина в трудных местах (постройки, волны миньонов,
лагеря джунглей), половина где придётся по проходимой карте. Без них сеть не
научится **отказывать**, а отказ — главное, чего не хватает нашей связке
против конвейера с его нулём ложных срабатываний.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from tcvm.cdragon import patch_of
from tcvm.ddragon import champion_ids, latest_version
from tcvm.generator import (
    CANONICAL_SIDE,
    Annotations,
    AssetLibrary,
    place_entities,
    random_scene,
    render_scene,
    walkable_mask,
)
from tcvm.identity import CANONICAL_CROP, INPUT_SIDE
from tcvm.synthesis import Geometry

ANNOTATIONS = Path(__file__).resolve().parents[1] / "annotations"
STRUCTURES_PATH = ANNOTATIONS / "map-structures.json"
OBJECTS_PATH = ANNOTATIONS / "map-objects.json"
DARKNESS_PATH = ANNOTATIONS / "map-darkness.png"
BORDER_PATH = ANNOTATIONS / "map-border.png"

# Размер миникарты выбирает игрок, и от него зависит вся резкость: значок
# несёт 16 настоящих пикселей при карте 200 и 47 при 600. Сеть обязана видеть
# весь диапазон, иначе на непривычной резкости она обвалится, а не ухудшится
# плавно. Границы взяты из таблицы размеров в own-model-plan.md.
NATIVE_SIDE_RANGE = (200, 600)
# Дрожание центра при вырезании: детектор промахивается примерно на пиксель,
# в плотных группах больше. Сеть должна видеть значок не только идеально
# посаженным, иначе на живом кадре её собьёт собственный детектор.
CENTER_JITTER = 2
# Насколько далеко от любого чемпиона должен стоять отрицательный пример.
NEGATIVE_CLEARANCE = 16.0
# Доля отрицательных примеров, взятых у построек и лагерей, а не наугад.
HARD_NEGATIVE_SHARE = 0.5
# Сколько отрицательных примеров приходится на одного чемпиона.
NEGATIVES_PER_CHAMPION = 0.5
# Метка отрицательного примера: пусто, потому что чемпиона в вырезе нет.
NO_CHAMPION = ""
# Как часто печатать ход работы.
PROGRESS_EVERY = 250


def cut(frame: np.ndarray, cx: int, cy: int, side: int) -> np.ndarray | None:
    """Вырез стороной side вокруг точки, приведённый ко входу сети.

    Вырез берётся в пикселях кадра — тем крупнее, чем крупнее карта у игрока, —
    и растягивается до постоянного входа сети. Резкость при этом сохраняется
    той, какую дала игра: в этом весь смысл.
    """
    half = side // 2
    left, top = cx - half, cy - half
    if left < 0 or top < 0 or left + side > frame.shape[1] or top + side > frame.shape[0]:
        return None
    piece = frame[top : top + side, left : left + side]
    if side == INPUT_SIDE:
        return piece.copy()
    return np.asarray(
        Image.fromarray(piece[..., ::-1].astype(np.uint8)).resize(
            (INPUT_SIDE, INPUT_SIDE), Image.BILINEAR
        )
    )[..., ::-1]


@dataclass(frozen=True)
class FrameLayout:
    """Во что превращается канонический кадр при данном размере карты."""

    scale: float
    walkable: np.ndarray


def negative_spots(
    rng: np.random.Generator,
    assets: AssetLibrary,
    layout: FrameLayout,
    taken: list[tuple[float, float]],
    count: int,
) -> list[tuple[int, int]]:
    """Точки без чемпионов: половина у построек и лагерей, половина наугад.

    Координаты возвращаются в пикселях кадра, а не канонических: разметка
    аннотаций каноническая, и её надо умножить на масштаб.
    """
    scale, walkable = layout.scale, layout.walkable
    hard = [(item["x"] * scale, item["y"] * scale) for item in assets.structures]
    hard += [(item["x"] * scale, item["y"] * scale) for item in assets.map_objects]
    spots: list[tuple[int, int]] = []
    for _ in range(count * 4):
        if len(spots) >= count:
            break
        if hard and rng.random() < HARD_NEGATIVE_SHARE:
            base = hard[rng.integers(len(hard))]
            x = int(base[0] + rng.integers(-3, 4))
            y = int(base[1] + rng.integers(-3, 4))
        else:
            x = int(rng.integers(0, round(CANONICAL_SIDE * scale)))
            y = int(rng.integers(0, round(CANONICAL_SIDE * scale)))
            if not walkable[min(round(y / scale), CANONICAL_SIDE - 1)][
                min(round(x / scale), CANONICAL_SIDE - 1)
            ]:
                continue
        if any(np.hypot(x - tx, y - ty) < NEGATIVE_CLEARANCE * scale for tx, ty in taken):
            continue
        spots.append((x, y))
    return spots


def harvest(
    frame: np.ndarray,
    metadata: dict,
    rng: np.random.Generator,
    assets: AssetLibrary,
    walkable: np.ndarray,
) -> tuple[list[np.ndarray], list[dict]]:
    """Все вырезы одного кадра: значки чемпионов и отрицательные примеры."""
    crops: list[np.ndarray] = []
    records: list[dict] = []
    native_side = metadata["frameSide"]
    scale = native_side / CANONICAL_SIDE
    crop_side = max(INPUT_SIDE // 4, round(CANONICAL_CROP * scale))
    jitter = max(1, round(CENTER_JITTER * scale))
    centers = [(item["frameX"], item["frameY"]) for item in metadata["champions"]]

    for item in metadata["champions"]:
        offset_x = int(rng.integers(-jitter, jitter + 1))
        offset_y = int(rng.integers(-jitter, jitter + 1))
        piece = cut(
            frame, round(item["frameX"]) + offset_x, round(item["frameY"]) + offset_y, crop_side
        )
        if piece is None:
            continue
        crops.append(piece[..., ::-1].copy())  # BGR кадра → RGB выреза
        records.append(
            {
                "championId": item["championId"],
                "affiliation": item["affiliation"],
                "moving": item["moving"],
                "nativeSide": native_side,
                "offset": [offset_x, offset_y],
            }
        )

    wanted = round(len(metadata["champions"]) * NEGATIVES_PER_CHAMPION)
    for x, y in negative_spots(rng, assets, FrameLayout(scale, walkable), centers, wanted):
        piece = cut(frame, x, y, crop_side)
        if piece is None:
            continue
        crops.append(piece[..., ::-1].copy())
        records.append(
            {
                "championId": NO_CHAMPION,
                "affiliation": "",
                "moving": False,
                "nativeSide": native_side,
            }
        )
    return crops, records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=2000, help="сколько кадров нарезать")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--version",
        default=None,
        help="версия Data Dragon; по умолчанию последняя. Указывать, когда"
        " локальные ассеты отстают от живого патча",
    )
    parser.add_argument("--out", type=Path, default=Path("out/crops"))
    parser.add_argument("--map-dir", type=Path, default=Path("data/cdragon"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    # Живой патч уходит вперёд раньше, чем скачиваются ассеты, и тогда
    # генерация падает на первом же кадре. Ключ позволяет остаться на той
    # версии, которая есть на диске.
    version = args.version or latest_version()
    patch = patch_of(version)
    structures = json.loads(STRUCTURES_PATH.read_text(encoding="utf-8"))["structures"]
    map_objects = json.loads(OBJECTS_PATH.read_text(encoding="utf-8"))["objects"]
    assets = AssetLibrary(
        args.map_dir,
        patch,
        structures,
        map_objects,
        Annotations(DARKNESS_PATH, BORDER_PATH),
    )
    champions = champion_ids(version)
    print(f"Data Dragon {version}, чемпионов {len(champions)}, seed {args.seed}")

    rng = np.random.default_rng(args.seed)
    walkable = walkable_mask(assets.darkness)
    crops: list[np.ndarray] = []
    records: list[dict] = []

    for index in range(args.frames):
        scene = place_entities(
            rng, random_scene(rng, champions, structures, map_objects), walkable
        )
        native_side = int(rng.integers(NATIVE_SIDE_RANGE[0], NATIVE_SIDE_RANGE[1] + 1))
        frame, metadata = render_scene(scene, assets, Geometry(native_side))
        pieces, notes = harvest(frame, metadata, rng, assets, walkable)
        crops.extend(pieces)
        records.extend(notes)
        if (index + 1) % PROGRESS_EVERY == 0:
            print(f"  {index + 1}/{args.frames}: вырезов {len(crops)}")

    stacked = np.stack(crops)
    np.save(args.out / "crops.npy", stacked)
    (args.out / "labels.jsonl").write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )

    positives = sum(1 for record in records if record["championId"])
    distinct = len({record["championId"] for record in records if record["championId"]})
    print()
    print(f"Вырезов: {len(crops)} ({positives} со значком, {len(crops) - positives} без)")
    print(f"Разных чемпионов: {distinct}")
    print(f"Записано: {(args.out / 'crops.npy').resolve()} ({stacked.nbytes / 1e6:.0f} МБ)")

    sides = [record["nativeSide"] for record in records]
    print(f"Размеры карты: от {min(sides)} до {max(sides)}, медиана {int(np.median(sides))}")

    preview = Image.new("RGB", (INPUT_SIDE * 16, INPUT_SIDE * 8), (24, 24, 24))
    chosen = rng.choice(len(crops), size=min(128, len(crops)), replace=False)
    for position, source in enumerate(chosen):
        tile = Image.fromarray(crops[source])
        preview.paste(tile, ((position % 16) * INPUT_SIDE, (position // 16) * INPUT_SIDE))
    preview.save(args.out / "contact-sheet.png")
    print(f"Посмотреть глазами: {(args.out / 'contact-sheet.png').resolve()}")


if __name__ == "__main__":
    main()
