"""Генератор вырезов для ступени «кто»: значки чемпионов и отрицательные примеры.

Вырезы берутся **из собранных кадров синтетики**, а не рисуются отдельно. Так
фон под значком, перекрытия соседними значками, туман и размытие движения
достаются даром и в точности совпадают с тем доменом, на котором работает
детектор. Отдельный рисовальщик вырезов пришлось бы согласовывать с
композитором вручную, и он бы разошёлся с ним при первой же правке.

    .venv/Scripts/python tools/generate_crops.py --frames 8000 --out out/crops-8k

Кладёт `crops.npy` (N, 32, 32, 3) uint8 и `labels.jsonl`. Один файл вместо
сотен тысяч PNG: так датасет читается за секунды и занимает втрое меньше места.

Отрицательные примеры — половина в трудных местах (постройки, волны миньонов,
лагеря джунглей), половина где придётся по проходимой карте. Без них сеть не
научится **отказывать**, а отказ — главное, чего не хватает нашей связке
против конвейера с его нулём ложных срабатываний.
"""

from __future__ import annotations

import argparse
import json
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

ANNOTATIONS = Path(__file__).resolve().parents[1] / "annotations"
STRUCTURES_PATH = ANNOTATIONS / "map-structures.json"
OBJECTS_PATH = ANNOTATIONS / "map-objects.json"
DARKNESS_PATH = ANNOTATIONS / "map-darkness.png"
BORDER_PATH = ANNOTATIONS / "map-border.png"

# Сторона выреза: значок занимает 25 px, оставшиеся 3-4 пикселя по краю дают
# сети немного окружения и терпимость к неточному центру от детектора.
CROP_SIDE = 32
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


def cut(frame: np.ndarray, cx: int, cy: int) -> np.ndarray | None:
    """Вырез CROP_SIDE вокруг точки; None, если не помещается в кадр."""
    half = CROP_SIDE // 2
    left, top = cx - half, cy - half
    if (
        left < 0
        or top < 0
        or left + CROP_SIDE > frame.shape[1]
        or top + CROP_SIDE > frame.shape[0]
    ):
        return None
    return frame[top : top + CROP_SIDE, left : left + CROP_SIDE]


def negative_spots(
    rng: np.random.Generator,
    assets: AssetLibrary,
    walkable: np.ndarray,
    taken: list[tuple[float, float]],
    count: int,
) -> list[tuple[int, int]]:
    """Точки без чемпионов: половина у построек и лагерей, половина наугад."""
    hard = [(item["x"], item["y"]) for item in assets.structures]
    hard += [(item["x"], item["y"]) for item in assets.map_objects]
    spots: list[tuple[int, int]] = []
    for _ in range(count * 4):
        if len(spots) >= count:
            break
        if hard and rng.random() < HARD_NEGATIVE_SHARE:
            base = hard[rng.integers(len(hard))]
            x = int(base[0] + rng.integers(-3, 4))
            y = int(base[1] + rng.integers(-3, 4))
        else:
            x = int(rng.integers(0, CANONICAL_SIDE))
            y = int(rng.integers(0, CANONICAL_SIDE))
            if not walkable[y, x]:
                continue
        if any(np.hypot(x - tx, y - ty) < NEGATIVE_CLEARANCE for tx, ty in taken):
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
    centers = [(item["x"], item["y"]) for item in metadata["champions"]]

    for item in metadata["champions"]:
        jitter_x = int(rng.integers(-CENTER_JITTER, CENTER_JITTER + 1))
        jitter_y = int(rng.integers(-CENTER_JITTER, CENTER_JITTER + 1))
        piece = cut(frame, round(item["x"]) + jitter_x, round(item["y"]) + jitter_y)
        if piece is None:
            continue
        crops.append(piece[..., ::-1].copy())  # BGR кадра → RGB выреза
        records.append(
            {
                "championId": item["championId"],
                "affiliation": item["affiliation"],
                "moving": item["moving"],
                "offset": [jitter_x, jitter_y],
            }
        )

    wanted = round(len(metadata["champions"]) * NEGATIVES_PER_CHAMPION)
    for x, y in negative_spots(rng, assets, walkable, centers, wanted):
        piece = cut(frame, x, y)
        if piece is None:
            continue
        crops.append(piece[..., ::-1].copy())
        records.append({"championId": NO_CHAMPION, "affiliation": "", "moving": False})
    return crops, records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=2000, help="сколько кадров нарезать")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("out/crops"))
    parser.add_argument("--map-dir", type=Path, default=Path("data/cdragon"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    version = latest_version()
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
        frame, metadata = render_scene(scene, assets)
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

    preview = Image.new("RGB", (CROP_SIDE * 16 * 2, CROP_SIDE * 8 * 2), (24, 24, 24))
    chosen = rng.choice(len(crops), size=min(128, len(crops)), replace=False)
    for position, source in enumerate(chosen):
        tile = Image.fromarray(crops[source]).resize(
            (CROP_SIDE * 2, CROP_SIDE * 2), Image.NEAREST
        )
        preview.paste(tile, ((position % 16) * CROP_SIDE * 2, (position // 16) * CROP_SIDE * 2))
    preview.save(args.out / "contact-sheet.png")
    print(f"Посмотреть глазами: {(args.out / 'contact-sheet.png').resolve()}")


if __name__ == "__main__":
    main()
