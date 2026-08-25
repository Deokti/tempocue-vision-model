"""Подбор порога уверенности детектора по кривой «точность против полноты».

Порог решает, какие вершины карты центров считать находками. Он не участвует
в обучении: одни и те же веса при разных порогах дают разные точность и
полноту, и выбор порога — отдельное решение, а не свойство модели.

    .venv/Scripts/python tools/sweep_threshold.py --weights out/detector-01/best.pt

Считает метрики на настоящих кадрах корпуса для набора порогов, печатает
таблицу и рисует кривую. Порядок целей программы лексикографический: сначала
не превышать бюджет ложных срабатываний, и только потом максимизировать
полноту — поэтому «лучший» порог выбирается не по одному числу, а глазами по
таблице.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from tcvm.detector import CenterDetector, decode_heatmap, match_centers
from tcvm.formats import bgra_to_rgb, default_corpus_dir, load_corpus

DEFAULT_CORPUS = None  # ищется при запуске: см. formats.default_corpus_dir
THRESHOLDS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def frame_to_tensor(pixels_bgra: np.ndarray) -> torch.Tensor:
    rgb = bgra_to_rgb(pixels_bgra).astype(np.float32) / 255.0
    return torch.from_numpy(np.transpose(rgb, (2, 0, 1))).unsqueeze(0)


def save_curve(rows: list[tuple[float, float, float]], path: Path) -> None:
    """Кривая: по горизонтали полнота, по вертикали точность."""
    size, pad = 420, 50
    image = Image.new("RGB", (size, size), (24, 24, 24))
    draw = ImageDraw.Draw(image)
    draw.rectangle([pad, pad, size - pad, size - pad], outline=(90, 90, 90))
    draw.text((pad, 8), "точность (вверх) против полноты (вправо)", fill=(200, 200, 200))

    plot = size - 2 * pad
    points = []
    for threshold, precision, recall in rows:
        x = pad + int(plot * recall)
        y = size - pad - int(plot * precision)
        points.append((x, y))
        draw.ellipse([x - 3, y - 3, x + 3, y + 3], fill=(0, 255, 120))
        draw.text((x + 5, y - 6), f"{threshold:.1f}", fill=(150, 150, 150))
    if len(points) > 1:
        draw.line(points, fill=(0, 180, 90))
    image.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("out/threshold-sweep"))
    args = parser.parse_args()
    corpus = args.corpus or default_corpus_dir()
    args.out.mkdir(parents=True, exist_ok=True)

    model = CenterDetector()
    model.load_state_dict(torch.load(args.weights, map_location="cpu"))
    model.eval()

    frames = load_corpus(corpus)
    logits_by_frame = []
    truths = []
    with torch.no_grad():
        for frame in frames:
            logits_by_frame.append(model(frame_to_tensor(frame.pixels)))
            truths.append(
                [
                    (region.x + region.width / 2, region.y + region.height / 2)
                    for region in (frame.labels.champions if frame.labels else ())
                ]
            )

    print(
        f"Веса: {args.weights}; кадров {len(frames)}, чемпионов {sum(len(t) for t in truths)}"
    )
    print()
    print("  порог  точность  полнота  ложных  ошибка центра")
    rows = []
    for threshold in THRESHOLDS:
        matched = predicted = labelled = 0
        offsets: list[float] = []
        for logits, truth in zip(logits_by_frame, truths, strict=True):
            found = decode_heatmap(logits, threshold=threshold)[0]
            hits, mean_offset = match_centers(found, truth)
            matched += hits
            predicted += len(found)
            labelled += len(truth)
            if hits:
                offsets.append(mean_offset)
        precision = matched / predicted if predicted else 0.0
        recall = matched / labelled if labelled else 0.0
        rows.append((threshold, precision, recall))
        print(
            f"  {threshold:5.1f}  {precision:8.3f}  {recall:7.3f}  "
            f"{predicted - matched:6d}  {np.mean(offsets) if offsets else 0:11.2f} px"
        )

    save_curve(rows, args.out / "precision-recall.png")
    print()
    print(f"Кривая: {(args.out / 'precision-recall.png').resolve()}")


if __name__ == "__main__":
    main()
