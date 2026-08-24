"""Генерация синтетического датасета кадров миникарты с разметкой.

Финал ступени 2. Кадры сохраняются PNG (без потерь), разметка — одним
файлом labels.jsonl (по строке JSON на кадр). Запуск детерминирован seed:
одинаковый seed даёт одинаковый датасет.

    .venv/Scripts/python tools/generate_dataset.py --count 200 --out out/dataset

Лист-контрольку из первых кадров кладёт рядом: результат проверяется глазами.
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
    AssetLibrary,
    place_entities,
    random_scene,
    render_scene,
    walkable_mask,
)

DEFAULT_MAP_DIR = Path(__file__).resolve().parents[1] / "data" / "cdragon"
STRUCTURES_PATH = Path(__file__).resolve().parents[1] / "annotations" / "map-structures.json"
CONTACT_SHEET_COLUMNS = 4


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--roster", type=int, default=0, help="0 — все чемпионы патча")
    parser.add_argument("--map-dir", type=Path, default=DEFAULT_MAP_DIR)
    parser.add_argument("--out", type=Path, default=Path("out/dataset"))
    args = parser.parse_args()
    (args.out / "frames").mkdir(parents=True, exist_ok=True)

    version = latest_version()
    patch = patch_of(version)
    roster = champion_ids(version)
    if args.roster:
        roster = roster[: args.roster]
    structures = json.loads(STRUCTURES_PATH.read_text(encoding="utf-8"))["structures"]
    assets = AssetLibrary(args.map_dir, patch, structures)
    rng = np.random.default_rng(args.seed)
    print(f"Data Dragon {version}, чемпионов в ростере {len(roster)}, seed {args.seed}")

    walkable_cache: dict[str, np.ndarray] = {}
    labels_path = args.out / "labels.jsonl"
    previews = []
    with labels_path.open("w", encoding="utf-8", newline="\n") as labels_file:
        for index in range(args.count):
            scene = random_scene(rng, roster, structures)
            if scene.variant not in walkable_cache:
                walkable_cache[scene.variant] = walkable_mask(assets.layer(scene.variant))
            scene = place_entities(rng, scene, walkable_cache[scene.variant])
            frame, metadata = render_scene(scene, assets)

            name = f"{index:05d}.png"
            Image.fromarray(frame[..., ::-1], "RGB").save(args.out / "frames" / name)
            labels_file.write(
                json.dumps({"frame": name, **metadata}, ensure_ascii=False) + "\n"
            )
            if len(previews) < CONTACT_SHEET_COLUMNS**2:
                previews.append(frame)
            if (index + 1) % 25 == 0:
                print(f"  {index + 1}/{args.count}")

    columns = min(CONTACT_SHEET_COLUMNS, len(previews))
    rows = (len(previews) + columns - 1) // columns
    side = previews[0].shape[0]
    sheet = Image.new("RGB", (side * columns, side * rows), (24, 24, 24))
    for position, frame in enumerate(previews):
        sheet.paste(
            Image.fromarray(frame[..., ::-1], "RGB"),
            (side * (position % columns), side * (position // columns)),
        )
    sheet.save(args.out / "contact-sheet.png")

    print(f"Кадров: {args.count}, разметка: {labels_path.resolve()}")
    print(f"Контролька: {(args.out / 'contact-sheet.png').resolve()}")


if __name__ == "__main__":
    main()
