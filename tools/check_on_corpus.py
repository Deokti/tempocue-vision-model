"""Проверка обученного детектора на настоящих кадрах корпуса.

Синтетическая проверка отвечает только на вопрос «выучилась ли задача».
Переносится ли выученное на живые кадры — отдельный вопрос, и этот
инструмент отвечает на него самым прямым способом: прогоняет веса по
корпусу реальных кадров и сравнивает с ручной разметкой.

    .venv/Scripts/python tools/check_on_corpus.py --weights out/detector-01/best.pt

Печатает точность и полноту по каждому кадру и в целом, кладёт в
out/corpus-check/ кадры с разметкой: зелёные крестики — истина, красные
круги — предсказание.

Важно про честность: нынешние 12 кадров корпуса — набор **для разработки**,
их разбор уже влиял на решения программы. Итоговая приёмка — по слепому
корпусу и инструментом `ReplayInspect` основного репозитория. Здесь мы
смотрим на них, чтобы понять, куда двигаться, а не чтобы объявить результат.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from tcvm.detector import (
    CenterDetector,
    decode_heatmap,
    match_centers,
)
from tcvm.formats import bgra_to_rgb, load_corpus

DEFAULT_CORPUS = Path(
    r"C:\Users\deokn\.codex\worktrees\739a\PROJECT\tests\TempoCue.Vision.Tests\ReplayCorpus"
)


def frame_to_tensor(pixels_bgra: np.ndarray) -> torch.Tensor:
    """Кадр BGRA 320×320 → тензор (1, 3, 320, 320) в том же виде, что при обучении."""
    rgb = bgra_to_rgb(pixels_bgra).astype(np.float32) / 255.0
    return torch.from_numpy(np.transpose(rgb, (2, 0, 1))).unsqueeze(0)


def draw_result(
    pixels_bgra: np.ndarray,
    truth: list[tuple[float, float]],
    found: list[tuple[float, float, float]],
    path: Path,
) -> None:
    scale = 2
    image = Image.fromarray(bgra_to_rgb(pixels_bgra), "RGB").resize(
        (320 * scale, 320 * scale), Image.NEAREST
    )
    draw = ImageDraw.Draw(image)
    for x, y in truth:
        cx, cy = x * scale, y * scale
        draw.line([cx - 9, cy, cx + 9, cy], fill=(0, 255, 70), width=2)
        draw.line([cx, cy - 9, cx, cy + 9], fill=(0, 255, 70), width=2)
    for x, y, confidence in found:
        cx, cy = x * scale, y * scale
        draw.ellipse([cx - 12, cy - 12, cx + 12, cy + 12], outline=(255, 80, 80), width=2)
        draw.text((cx + 13, cy - 13), f"{confidence:.2f}", fill=(255, 80, 80))
    image.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--out", type=Path, default=Path("out/corpus-check"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    model = CenterDetector()
    model.load_state_dict(torch.load(args.weights, map_location="cpu"))
    model.eval()
    print(f"Веса: {args.weights}")

    matched_total = predicted_total = labelled_total = 0
    offsets: list[float] = []
    for frame in load_corpus(args.corpus):
        truth = [
            (region.x + region.width / 2, region.y + region.height / 2)
            for region in (frame.labels.champions if frame.labels else ())
        ]
        with torch.no_grad():
            logits = model(frame_to_tensor(frame.pixels))
        found = decode_heatmap(logits, threshold=args.threshold)[0]
        hits, mean_offset = match_centers(found, truth)

        matched_total += hits
        predicted_total += len(found)
        labelled_total += len(truth)
        if hits:
            offsets.append(mean_offset)

        draw_result(frame.pixels, truth, found, args.out / f"{frame.name}.png")
        print(
            f"  {frame.name}: нашлось {hits} из {len(truth)}, всего предсказаний {len(found)}"
        )

    precision = matched_total / predicted_total if predicted_total else 0.0
    recall = matched_total / labelled_total if labelled_total else 0.0
    print()
    print(
        f"Итого по корпусу: полнота {recall:.3f} ({matched_total} из {labelled_total}), "
        f"точность {precision:.3f}, ошибка центра "
        f"{np.mean(offsets) if offsets else 0:.2f} px"
    )
    print(f"Картинки: {args.out.resolve()}")


if __name__ == "__main__":
    main()
