"""Обучение сети опознания «кто это» на вырезах синтетики.

    .venv/Scripts/python tools/train_identity.py --dataset out/crops-8k --epochs 20

Учит по вырезам из `tools/generate_crops.py`. Проверочная выборка отделяется
**по кадрам не получится — вырезы уже перемешаны**, поэтому делится случайно;
это допустимо, потому что кадры синтетики независимы, а один и тот же значок
в двух вырезах не встречается.

Печатает две доли отдельно: сколько чемпионов опознано верно и сколько
отказов сделано верно. Мешать их в одно число нельзя — они лечат разные
болезни, и одна может расти за счёт другой.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import nn

from tcvm.identity import (
    NO_CHAMPION,
    IdentityNet,
    as_input,
    augment_crops,
    build_vocabulary,
    choose,
    evaluate_choices,
)

VALIDATION_SHARE = 0.15
# Разброс яркости — из измеренного разрыва синтетики с настоящими кадрами
# (docs/detector.md): настоящие ярче, множитель подобран с запасом.
BRIGHTNESS = (0.95, 1.20)
PREVIEW_COLUMNS = 12
PREVIEW_ROWS = 6
PREVIEW_ZOOM = 3


@dataclass(frozen=True)
class Fold:
    crops: torch.Tensor
    labels: torch.Tensor
    names: list[str]

    def __len__(self) -> int:
        return len(self.crops)


@dataclass(frozen=True)
class EpochReport:
    epoch: int
    train_loss: float
    val_loss: float
    champion_accuracy: float
    reject_accuracy: float
    seconds: float


def load_dataset(dataset: Path) -> tuple[np.ndarray, list[str]]:
    crops = np.load(dataset / "crops.npy")
    names = [
        json.loads(line)["championId"]
        for line in (dataset / "labels.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    return crops, names


def evaluate(
    model: IdentityNet, fold: Fold, vocabulary: list[str], device: torch.device, batch: int
) -> tuple[float, object]:
    model.eval()
    loss_sum = 0.0
    chosen: list[str] = []
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for start in range(0, len(fold), batch):
            crops = as_input(fold.crops[start : start + batch]).to(device)
            labels = fold.labels[start : start + batch].to(device)
            logits = model(crops)
            loss_sum += float(criterion(logits, labels)) * len(labels)
            chosen.extend(choose(logits.cpu(), vocabulary))
    return loss_sum / len(fold), evaluate_choices(chosen, fold.names)


def run_epoch(
    model: IdentityNet,
    fold: Fold,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    setup: tuple[int, np.random.Generator],
) -> float:
    batch_size, rng = setup
    model.train()
    criterion = nn.CrossEntropyLoss()
    order = torch.randperm(len(fold))
    total = 0.0
    for start in range(0, len(order), batch_size):
        index = order[start : start + batch_size]
        painted = augment_crops(fold.crops[index].numpy(), rng, BRIGHTNESS)
        crops = as_input(torch.from_numpy(painted.astype(np.uint8))).to(device)
        labels = fold.labels[index].to(device)
        optimizer.zero_grad()
        loss = criterion(model(crops), labels)
        loss.backward()
        optimizer.step()
        total += float(loss.detach()) * len(index)
    return total / len(order)


def save_preview(
    model: IdentityNet, fold: Fold, vocabulary: list[str], device: torch.device, path: Path
) -> None:
    """Вырезы проверки с ответом сети: зелёный — верно, красный — мимо."""
    count = min(PREVIEW_COLUMNS * PREVIEW_ROWS, len(fold))
    model.eval()
    with torch.no_grad():
        logits = model(as_input(fold.crops[:count]).to(device)).cpu()
    chosen = choose(logits, vocabulary)

    side = fold.crops.shape[1] * PREVIEW_ZOOM
    sheet = Image.new(
        "RGB", (PREVIEW_COLUMNS * (side + 6), PREVIEW_ROWS * (side + 16)), (24, 24, 24)
    )
    draw = ImageDraw.Draw(sheet)
    for position in range(count):
        x = (position % PREVIEW_COLUMNS) * (side + 6)
        y = (position // PREVIEW_COLUMNS) * (side + 16)
        tile = Image.fromarray(fold.crops[position].numpy()).resize((side, side), Image.NEAREST)
        sheet.paste(tile, (x, y + 14))
        right = chosen[position].lower() == fold.names[position].lower()
        label = chosen[position] or "отказ"
        draw.text((x, y), label[:14], fill=(0, 255, 120) if right else (255, 90, 90))
    sheet.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("out/crops"))
    parser.add_argument("--out", type=Path, default=Path("out/identity"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else args.device
    )
    torch.manual_seed(args.seed)
    print(
        f"Устройство: {device}"
        + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else "")
    )

    crops, names = load_dataset(args.dataset)
    vocabulary = build_vocabulary(names)
    index_of = {name: index for index, name in enumerate(vocabulary)}
    labels = np.array([index_of[name] for name in names], dtype=np.int64)

    order = np.random.default_rng(args.seed).permutation(len(crops))
    split = int(len(crops) * (1 - VALIDATION_SHARE))
    parts = {"обучение": order[:split], "проверка": order[split:]}
    folds = {
        title: Fold(
            crops=torch.from_numpy(crops[part]),
            labels=torch.from_numpy(labels[part]),
            names=[names[i] for i in part],
        )
        for title, part in parts.items()
    }
    positives = sum(1 for name in names if name != NO_CHAMPION)
    print(
        f"Вырезов: {len(crops)} ({positives} со значком, {len(crops) - positives} без); "
        f"классов {len(vocabulary)}"
    )
    print(f"Обучение {len(folds['обучение'])}, проверка {len(folds['проверка'])}")

    model = IdentityNet(len(vocabulary)).to(device)
    print(f"Параметров в модели: {sum(p.numel() for p in model.parameters()) / 1000:.0f} тысяч")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    rng = np.random.default_rng(args.seed + 1)

    reports: list[EpochReport] = []
    best = -1.0
    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        train_loss = run_epoch(
            model, folds["обучение"], optimizer, device, (args.batch_size, rng)
        )
        val_loss, metrics = evaluate(
            model, folds["проверка"], vocabulary, device, args.batch_size
        )
        report = EpochReport(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            champion_accuracy=metrics.champion_accuracy,
            reject_accuracy=metrics.reject_accuracy,
            seconds=time.perf_counter() - started,
        )
        reports.append(report)
        print(
            f"  эпоха {epoch:3d}: потери {train_loss:.3f}/{val_loss:.3f}, "
            f"чемпионы {metrics.champion_accuracy:.3f}, "
            f"отказы {metrics.reject_accuracy:.3f}, {report.seconds:.0f} с"
        )
        torch.save(model.state_dict(), args.out / "last.pt")
        if metrics.champion_accuracy > best:
            best = metrics.champion_accuracy
            torch.save(model.state_dict(), args.out / "best.pt")

    save_preview(model, folds["проверка"], vocabulary, device, args.out / "predictions.png")
    (args.out / "vocabulary.json").write_text(
        json.dumps(vocabulary, ensure_ascii=False), encoding="utf-8"
    )
    (args.out / "report.json").write_text(
        json.dumps(
            {
                "dataset": str(args.dataset),
                "crops": len(crops),
                "classes": len(vocabulary),
                "device": device.type,
                "epochs": [report.__dict__ for report in reports],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print()
    print(f"Лучшая доля по чемпионам: {best:.3f}")
    print(f"Результаты: {args.out.resolve()}")


if __name__ == "__main__":
    main()
