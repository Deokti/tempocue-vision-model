"""Проверка целей обучения глазами: карта центров поверх кадра.

Итерация «цели для детектора». Прежде чем обучать сеть, надо увидеть, что
цель совпадает со значками: ошибка в целях не проявится как падение потерь —
сеть послушно выучит неверное.

    .venv/Scripts/python tools/inspect_targets.py --dataset out/dataset

Результат — out/targets/<кадр>.png: кадр, карта центров, кадр с крестиками в
позициях цели.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from tcvm.detector import CANONICAL_SIDE, HEATMAP_SIDE, STRIDE, make_heatmap


def load_labels(dataset: Path) -> list[dict]:
    lines = (dataset / "labels.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


def sheet_for(dataset: Path, record: dict) -> Image.Image:
    frame = Image.open(dataset / "frames" / record["frame"]).convert("RGB")
    centers = [(c["x"], c["y"]) for c in record["champions"]]
    heatmap = make_heatmap(centers)[0]

    scale = 2
    side = CANONICAL_SIDE * scale
    sheet = Image.new("RGB", (side * 3 + 16, side), (24, 24, 24))
    sheet.paste(frame.resize((side, side), Image.NEAREST), (0, 0))

    hot = Image.fromarray((heatmap * 255).astype(np.uint8), "L").resize(
        (side, side), Image.NEAREST
    )
    sheet.paste(hot.convert("RGB"), (side + 8, 0))

    marked = frame.resize((side, side), Image.NEAREST)
    draw = ImageDraw.Draw(marked)
    for x, y in centers:
        cx, cy = x * scale, y * scale
        draw.line([cx - 8, cy, cx + 8, cy], fill=(0, 255, 70), width=2)
        draw.line([cx, cy - 8, cx, cy + 8], fill=(0, 255, 70), width=2)
    sheet.paste(marked, (side * 2 + 16, 0))
    return sheet


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("out/dataset"))
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--out", type=Path, default=Path("out/targets"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    records = load_labels(args.dataset)
    print(
        f"Кадров в датасете: {len(records)}, карта центров {HEATMAP_SIDE}×{HEATMAP_SIDE}, "
        f"шаг {STRIDE}"
    )
    total = 0
    for record in records[: args.count]:
        sheet_for(args.dataset, record).save(args.out / record["frame"])
        total += len(record["champions"])
        print(f"  {record['frame']}: чемпионов {len(record['champions'])}")

    champions = [len(r["champions"]) for r in records]
    print(
        f"Чемпионов на кадр: медиана {int(np.median(champions))}, "
        f"всего в датасете {sum(champions)}"
    )
    print(
        f"Положительных клеток на кадр: {total / max(args.count, 1):.1f} "
        f"из {HEATMAP_SIDE * HEATMAP_SIDE} — отсюда фокальная потеря"
    )
    print(f"Картинки: {args.out.resolve()}")


if __name__ == "__main__":
    main()
