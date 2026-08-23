"""Визуальная проверка читалки форматов: выгружает корпус и записи в PNG.

Запуск из корня репозитория:

    .venv\\Scripts\\python tools\\inspect_corpus.py

Результат в out/inspect/: по каталогу на кадр (кадр целиком, кадр с разметкой,
вырезы чемпионов, эталоны) и по каталогу на запись (полоса кадров).
Смысл инструмента — принцип программы «результат проверяется глазами»:
прежде чем считать по пикселям что-либо, надо увидеть, что мы читаем их верно.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from tcvm.formats import (
    ReplayFrame,
    ReplaySequence,
    bgra_to_rgb,
    load_corpus,
    load_sequences,
)

DEFAULT_DATA_ROOT = Path(
    r"C:\Users\deokn\.codex\worktrees\739a\PROJECT\tests\TempoCue.Vision.Tests"
)

UPSCALE = 8


def to_image(bgra: np.ndarray) -> Image.Image:
    return Image.fromarray(bgra_to_rgb(bgra), mode="RGB")


def upscaled(bgra: np.ndarray, factor: int = UPSCALE) -> Image.Image:
    image = to_image(bgra)
    return image.resize((image.width * factor, image.height * factor), Image.NEAREST)


def annotate(frame: ReplayFrame) -> Image.Image:
    image = to_image(frame.pixels).convert("RGB")
    scaled = image.resize((image.width * 2, image.height * 2), Image.NEAREST)
    draw = ImageDraw.Draw(scaled)
    if frame.labels is None:
        return scaled
    for region in frame.labels.regions:
        is_champion = region.kind == "Champion"
        color = (0, 255, 70) if is_champion else (255, 60, 60)
        box = [
            region.x * 2,
            region.y * 2,
            (region.x + region.width) * 2,
            (region.y + region.height) * 2,
        ]
        draw.rectangle(box, outline=color, width=2)
        caption = region.champion_id if is_champion else region.kind
        if caption:
            draw.text((box[0], max(0, box[1] - 12)), caption, fill=color)
    return scaled


def dump_frame(frame: ReplayFrame, out_root: Path) -> None:
    out = out_root / frame.name
    (out / "crops").mkdir(parents=True, exist_ok=True)
    (out / "refs").mkdir(parents=True, exist_ok=True)

    to_image(frame.pixels).save(out / "frame.png")
    annotate(frame).save(out / "frame-annotated.png")

    for region in frame.labels.champions if frame.labels else ():
        upscaled(frame.crop(region)).save(
            out / "crops" / f"{region.champion_id}-{region.affiliation}.png"
        )
    for reference in frame.references:
        side = "ally" if reference.affiliation == 0 else "enemy"
        upscaled(reference.pixels).save(out / "refs" / f"{reference.champion_id}-{side}.png")


def dump_sequence(sequence: ReplaySequence, out_root: Path, step: int = 4) -> None:
    out = out_root / sequence.name
    out.mkdir(parents=True, exist_ok=True)
    picked = list(range(0, len(sequence.frames), step))
    if picked[-1] != len(sequence.frames) - 1:
        picked.append(len(sequence.frames) - 1)

    height = sequence.frames[0].shape[0]
    width = sequence.frames[0].shape[1]
    strip = Image.new("RGB", (width * len(picked), height))
    for column, index in enumerate(picked):
        strip.paste(to_image(sequence.frames[index]), (column * width, 0))
    strip.save(out / f"strip-every-{step}.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="каталог с ReplayCorpus/ и ReplaySequences/ основного репозитория",
    )
    parser.add_argument("--out", type=Path, default=Path("out/inspect"))
    args = parser.parse_args()

    frames = load_corpus(args.data / "ReplayCorpus")
    print(f"Кадров: {len(frames)}")
    for frame in frames:
        champions = len(frame.labels.champions) if frame.labels else 0
        negatives = len(frame.labels.hard_negatives) if frame.labels else 0
        print(
            f"  {frame.name}: чемпионов {champions}, "
            f"hard negatives {negatives}, эталонов {len(frame.references)}"
        )
        dump_frame(frame, args.out / "frames")

    sequences = load_sequences(args.data / "ReplaySequences")
    print(f"Записей: {len(sequences)}")
    for sequence in sequences:
        print(
            f"  {sequence.name}: кадров {len(sequence.frames)} "
            f"за {sequence.duration_seconds:.1f} с, "
            f"видимы: {', '.join(sequence.visible_throughout) or '—'}"
        )
        dump_sequence(sequence, args.out / "sequences")

    print(f"Картинки: {args.out.resolve()}")


if __name__ == "__main__":
    main()
