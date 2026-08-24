"""Проверка домена: насколько легко отличить нашу синтетику от живых кадров.

Собирает настоящие кадры (корпус, записи, replay-inbox) и столько же
синтетических, режет их на патчи и обучает маленького судью. Разделение на
обучающую и проверочную выборки — по кадрам, чтобы патчи одного кадра не
попали в обе.

    .venv/Scripts/python tools/judge_domain.py

Печатает точность судьи по эпохам и кладёт в out/domain-judge/ кривые и лист
патчей обоих доменов для проверки глазами.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from tcvm.cdragon import patch_of
from tcvm.ddragon import champion_ids, latest_version
from tcvm.domain_judge import (
    PATCH_SIDE,
    Split,
    TrainingSetup,
    build_dataset,
    cut_patches,
    train_judge,
)
from tcvm.formats import load_corpus, load_frame, load_sequences
from tcvm.generator import (
    AssetLibrary,
    place_entities,
    random_scene,
    render_scene,
    walkable_mask,
)

DEFAULT_DATA_ROOT = Path(
    r"C:\Users\deokn\.codex\worktrees\739a\PROJECT\tests\TempoCue.Vision.Tests"
)
DEFAULT_INBOX = Path(r"C:\Users\deokn\.codex\worktrees\739a\PROJECT\replay-inbox")
DEFAULT_MAP_DIR = Path(__file__).resolve().parents[1] / "data" / "cdragon"
ANNOTATIONS = Path(__file__).resolve().parents[1] / "annotations"
STRUCTURES_PATH = ANNOTATIONS / "map-structures.json"
OBJECTS_PATH = ANNOTATIONS / "map-objects.json"
DARKNESS_PATH = ANNOTATIONS / "map-darkness.png"
VALIDATION_SHARE = 0.3
# Пороги вердикта: ниже CLOSE домены считаем неразличимыми, выше DIVERGED —
# разошедшимися. Взяты как «монетка с запасом» и «уверенное узнавание».
ACCURACY_CLOSE = 0.65
ACCURACY_DIVERGED = 0.85


def collect_real_frames(data_root: Path, inbox: Path) -> list[np.ndarray]:
    """Настоящие кадры: корпус, по одному кадру из каждой записи, replay-inbox."""
    frames = [frame.pixels for frame in load_corpus(data_root / "ReplayCorpus")]
    for sequence in load_sequences(data_root / "ReplaySequences"):
        frames.append(sequence.frames[0])
    if inbox.exists():
        for path in sorted(inbox.glob("*.tempocue-vision")):
            frames.append(load_frame(path).pixels)
    return frames


def synthesize_frames(count: int, assets: AssetLibrary, roster: list[str], seed: int):
    rng = np.random.default_rng(seed)
    walkable_cache: dict[str, np.ndarray] = {}
    frames = []
    for _ in range(count):
        scene = random_scene(rng, roster, assets.structures, assets.map_objects)
        if scene.variant not in walkable_cache:
            walkable_cache[scene.variant] = walkable_mask(assets.darkness)
        scene = place_entities(rng, scene, walkable_cache[scene.variant])
        frames.append(render_scene(scene, assets)[0])
    return frames


def save_patch_sheet(real: np.ndarray, synthetic: np.ndarray, path: Path) -> None:
    """Лист патчей: верхний ряд — настоящие, нижний — синтетические."""
    columns = 8
    scale = 3
    cell = PATCH_SIDE * scale
    sheet = Image.new("RGB", (cell * columns, cell * 2 + 20), (24, 24, 24))
    draw = ImageDraw.Draw(sheet)
    for row, (source, caption) in enumerate(((real, "настоящие"), (synthetic, "синтез"))):
        draw.text((4, row * (cell + 10)), caption, fill=(220, 220, 220))
        for column in range(min(columns, len(source))):
            rgb = (np.transpose(source[column], (1, 2, 0)) * 255).astype(np.uint8)
            image = Image.fromarray(rgb, "RGB").resize((cell, cell), Image.NEAREST)
            sheet.paste(image, (column * cell, row * (cell + 10) + 10))
    sheet.save(path)


def save_curves(losses: list[float], accuracies: list[float], path: Path) -> None:
    """Кривые обучения без внешних библиотек: потери и точность по эпохам."""
    width, height, pad = 640, 260, 30
    image = Image.new("RGB", (width, height), (24, 24, 24))
    draw = ImageDraw.Draw(image)
    plot_width = width - 2 * pad
    plot_height = height - 2 * pad
    draw.rectangle([pad, pad, width - pad, height - pad], outline=(90, 90, 90))
    # Линия «монетка»: точность 0,5 — цель, а не провал.
    coin_y = height - pad - int(0.5 * plot_height)
    draw.line([pad, coin_y, width - pad, coin_y], fill=(90, 90, 90))
    draw.text((pad + 4, coin_y - 14), "0,5 — домены неразличимы", fill=(150, 150, 150))

    top_loss = max(losses) or 1.0
    for series, color, scale_max in (
        (losses, (255, 160, 0), top_loss),
        (accuracies, (0, 255, 120), 1.0),
    ):
        points = [
            (
                pad + int(plot_width * index / max(len(series) - 1, 1)),
                height - pad - int(plot_height * value / scale_max),
            )
            for index, value in enumerate(series)
        ]
        draw.line(points, fill=color, width=2)
    draw.text(
        (pad + 4, 8), "оранжевый — потери, зелёный — точность судьи", fill=(200, 200, 200)
    )
    image.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--inbox", type=Path, default=DEFAULT_INBOX)
    parser.add_argument("--map-dir", type=Path, default=DEFAULT_MAP_DIR)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("out/domain-judge"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    version = latest_version()
    patch = patch_of(version)
    structures = json.loads(STRUCTURES_PATH.read_text(encoding="utf-8"))["structures"]
    map_objects = json.loads(OBJECTS_PATH.read_text(encoding="utf-8"))["objects"]
    assets = AssetLibrary(args.map_dir, patch, structures, map_objects, DARKNESS_PATH)

    real = collect_real_frames(args.data, args.inbox)
    synthetic = synthesize_frames(len(real), assets, champion_ids(version), args.seed)
    print(f"Кадров: настоящих {len(real)}, синтетических {len(synthetic)}")

    rng = np.random.default_rng(args.seed)
    split = int(len(real) * (1 - VALIDATION_SHARE))
    order_real = rng.permutation(len(real))
    order_synthetic = rng.permutation(len(synthetic))
    train_x, train_y = build_dataset(
        [real[i] for i in order_real[:split]],
        [synthetic[i] for i in order_synthetic[:split]],
        rng,
    )
    val_x, val_y = build_dataset(
        [real[i] for i in order_real[split:]],
        [synthetic[i] for i in order_synthetic[split:]],
        rng,
    )
    print(f"Патчей: обучение {len(train_x)}, проверка {len(val_x)}")

    _, losses, accuracies = train_judge(
        Split(train_x, train_y, val_x, val_y),
        TrainingSetup(epochs=args.epochs, seed=args.seed),
    )
    for epoch, (loss, accuracy) in enumerate(zip(losses, accuracies, strict=True), start=1):
        print(f"  эпоха {epoch:2d}: потери {loss:.4f}, точность судьи {accuracy:.3f}")

    save_curves(losses, accuracies, args.out / "curves.png")
    save_patch_sheet(
        cut_patches(real[0], np.random.default_rng(1)),
        cut_patches(synthetic[0], np.random.default_rng(1)),
        args.out / "patches.png",
    )

    best = max(accuracies)
    print()
    print(f"Лучшая точность судьи: {best:.3f}")
    if best < ACCURACY_CLOSE:
        print("Домены близки: судья почти не отличает синтетику от реальности.")
    elif best < ACCURACY_DIVERGED:
        print("Домены различимы: разницу стоит найти и объяснить.")
    else:
        print("Домены разошлись: генератор нужно чинить до обучения распознаванию.")
    print(f"Картинки: {args.out.resolve()}")


if __name__ == "__main__":
    main()
