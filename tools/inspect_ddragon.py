"""Проверка глазами: портреты Data Dragon против настоящих вырезов корпуса.

Этап 4 сверки отрисовки. Скачивает (с кэшем) портреты всех чемпионов,
встречающихся в разметке корпуса, и для каждого собирает полосу:

    настоящие вырезы 25x25 из кадров | портрет 120x120 | наивное уменьшение до 25

Наивное уменьшение — площадной ресемплер без всяких параметров; это НЕ синтез
сверки, а только проверка, что личности совпадают и версия патча пригодна.
Запуск: .venv/Scripts/python tools/inspect_ddragon.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tcvm.ddragon import champion_ids, latest_version, portrait_bgra  # noqa: E402
from tcvm.formats import bgra_to_rgb, load_corpus  # noqa: E402
from tcvm.matching import ICON_SIDE, crop_centered  # noqa: E402

DEFAULT_DATA_ROOT = Path(
    r"C:\Users\deokn\.codex\worktrees\739a\PROJECT\tests\TempoCue.Vision.Tests")
UPSCALE = 8


def upscaled(bgra: np.ndarray, factor: int = UPSCALE) -> Image.Image:
    image = Image.fromarray(bgra_to_rgb(bgra), "RGB")
    return image.resize((image.width * factor, image.height * factor), Image.NEAREST)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out", type=Path, default=Path("out/ddragon-check"))
    args = parser.parse_args()

    version = latest_version()
    known = set(champion_ids(version))
    print(f"Версия Data Dragon: {version}, чемпионов в реестре: {len(known)}")

    crops: dict[str, list[np.ndarray]] = {}
    for frame in load_corpus(args.data / "ReplayCorpus"):
        for region in frame.labels.champions if frame.labels else ():
            cx = region.x + region.width // 2
            cy = region.y + region.height // 2
            try:
                crops.setdefault(region.champion_id, []).append(
                    crop_centered(frame.pixels, cx, cy))
            except ValueError:
                continue  # значок у самого края кадра

    args.out.mkdir(parents=True, exist_ok=True)
    missing = sorted(set(crops) - known)
    for champion in sorted(crops):
        if champion in missing:
            continue
        portrait = portrait_bgra(champion, version)
        naive = np.asarray(
            Image.fromarray(portrait, "RGBA").resize(
                (ICON_SIDE, ICON_SIDE), Image.BOX))

        cell = ICON_SIDE * UPSCALE
        sheet = Image.new("RGB", (cell * (len(crops[champion]) + 2) + 16, cell),
                          (24, 24, 24))
        for column, crop in enumerate(crops[champion]):
            sheet.paste(upscaled(crop), (column * cell, 0))
        sheet.paste(
            Image.fromarray(portrait[..., [2, 1, 0]], "RGB").resize((cell, cell), Image.NEAREST),
            (len(crops[champion]) * cell + 8, 0))
        sheet.paste(upscaled(naive[..., :4]), ((len(crops[champion]) + 1) * cell + 16, 0))
        sheet.save(args.out / f"{champion}.png")

    print(f"Чемпионов в разметке корпуса: {len(crops)}, "
          f"листов сохранено: {len(crops) - len(missing)}")
    if missing:
        print(f"НЕТ в Data Dragon {version}: {', '.join(missing)}")
    print(f"Картинки: {args.out.resolve()}")


if __name__ == "__main__":
    main()
