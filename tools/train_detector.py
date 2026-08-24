"""Обучение детектора центров на синтетическом датасете.

Вторая половина ступени 3. Цель ступени — не качество, а обкатка полного
цикла: данные → обучение → метрики → ONNX → приёмка по корпусу в основном
репозитории.

    # быстрая проверка на слабой машине (минуты)
    .venv/Scripts/python tools/train_detector.py --dataset out/dataset --epochs 3

    # настоящий прогон (вторая машина, видеокарта)
    .venv/Scripts/python tools/train_detector.py --dataset out/dataset-big --epochs 40

Что кладётся в каталог запуска (--out):

- `best.pt` — веса с лучшей полнотой на проверочной выборке;
- `last.pt` — веса последней эпохи (для продолжения после обрыва);
- `report.json` — метрики по эпохам, одним файлом для чтения ассистентом;
- `curves.png` — кривые потерь и метрик;
- `predictions.png` — предсказания на проверочных кадрах для проверки глазами.

Разделение выборок — по кадрам; кадры независимы (каждый — своя случайная
сцена), поэтому случайное деление корректно. Валидация синтетическая и
говорит только о том, выучилась ли задача: настоящая приёмка — по корпусу
реальных кадров в основном репозитории.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from tcvm.detector import (
    CANONICAL_SIDE,
    HEATMAP_SIDE,
    NO_AUGMENTATION,
    Augmentation,
    CenterDetector,
    DetectionMetrics,
    augment_frame,
    decode_heatmap,
    focal_heatmap_loss,
    make_heatmap,
    match_centers,
)

VALIDATION_SHARE = 0.2
# Шкала перевода байтов изображения в доли единицы.
BYTE_SCALE = 255.0
PREVIEW_FRAMES = 3


@dataclass(frozen=True)
class Fold:
    """Одна выборка: кадры, цели и центры для метрик."""

    frames: torch.Tensor  # uint8, перевод в доли единицы — на пачке
    targets: torch.Tensor
    centers: list[list[tuple]]

    def __len__(self) -> int:
        return len(self.frames)


@dataclass(frozen=True)
class EpochReport:
    epoch: int
    train_loss: float
    val_loss: float
    precision: float
    recall: float
    mean_offset: float
    seconds: float


def load_dataset(dataset: Path) -> tuple[torch.Tensor, torch.Tensor, list[list[tuple]]]:
    """Кадры (N,3,320,320) uint8, цели (N,1,80,80) и центры для метрик.

    Кадры держатся в памяти байтами, а не вещественными числами: 12 000
    кадров float32 — это 15 ГБ, те же кадры uint8 — 3,7 ГБ. Перевод в доли
    единицы делается на пачке, перед подачей в сеть.
    """
    records = [
        json.loads(line)
        for line in (dataset / "labels.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    frames = np.empty((len(records), 3, CANONICAL_SIDE, CANONICAL_SIDE), dtype=np.uint8)
    targets = np.empty((len(records), 1, 80, 80), dtype=np.float32)
    centers: list[list[tuple]] = []
    for index, record in enumerate(records):
        with Image.open(dataset / "frames" / record["frame"]) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        frames[index] = np.transpose(rgb, (2, 0, 1))
        points = [(c["x"], c["y"]) for c in record["champions"]]
        targets[index] = make_heatmap(points)
        centers.append(points)
    return torch.from_numpy(frames), torch.from_numpy(targets), centers


def as_input(frames: torch.Tensor) -> torch.Tensor:
    """Байты кадра → доли единицы, в том же виде, что при обучении."""
    return frames.float() / BYTE_SCALE


def evaluate(
    model: CenterDetector, fold: Fold, device: torch.device, batch_size: int
) -> tuple[float, DetectionMetrics]:
    """Потери и метрики детекции на выборке."""
    model.eval()
    total_loss = 0.0
    matched = predicted = labelled = 0
    offsets: list[float] = []
    with torch.no_grad():
        for start in range(0, len(fold), batch_size):
            batch = as_input(fold.frames[start : start + batch_size]).to(device)
            target = fold.targets[start : start + batch_size].to(device)
            logits = model(batch)
            total_loss += float(focal_heatmap_loss(logits, target)) * len(batch)
            for offset, found in enumerate(decode_heatmap(logits.cpu())):
                truth = fold.centers[start + offset]
                hits, mean_offset = match_centers(found, truth)
                matched += hits
                predicted += len(found)
                labelled += len(truth)
                if hits:
                    offsets.append(mean_offset)
    metrics = DetectionMetrics(
        matched=matched,
        predicted=predicted,
        labelled=labelled,
        mean_offset=float(np.mean(offsets)) if offsets else 0.0,
    )
    return total_loss / len(fold), metrics


def save_curves(reports: list[EpochReport], path: Path) -> None:
    """Кривые: потери обучения и проверки, полнота и точность."""
    width, height, pad = 720, 320, 40
    image = Image.new("RGB", (width, height), (24, 24, 24))
    draw = ImageDraw.Draw(image)
    draw.rectangle([pad, pad, width - pad, height - pad], outline=(90, 90, 90))
    plot_w, plot_h = width - 2 * pad, height - 2 * pad

    series = (
        ("потери обучения", [r.train_loss for r in reports], (255, 160, 0)),
        ("потери проверки", [r.val_loss for r in reports], (255, 90, 90)),
        ("полнота", [r.recall for r in reports], (0, 255, 120)),
        ("точность", [r.precision for r in reports], (90, 180, 255)),
    )
    loss_max = max(max(r.train_loss, r.val_loss) for r in reports) or 1.0
    for label, values, color in series:
        scale = loss_max if "потери" in label else 1.0
        points = [
            (
                pad + int(plot_w * i / max(len(values) - 1, 1)),
                height - pad - int(plot_h * min(v / scale, 1.0)),
            )
            for i, v in enumerate(values)
        ]
        if len(points) > 1:
            draw.line(points, fill=color, width=2)
    draw.text(
        (pad, 8),
        "оранжевый/красный — потери, зелёный — полнота, синий — точность",
        fill=(200, 200, 200),
    )
    draw.text((pad, height - 26), f"эпох: {len(reports)}", fill=(160, 160, 160))
    image.save(path)


def save_predictions(
    model: CenterDetector, fold: Fold, device: torch.device, path: Path
) -> None:
    """Кадры проверки: зелёные крестики — истина, красные круги — предсказание."""
    model.eval()
    count = min(PREVIEW_FRAMES, len(fold))
    with torch.no_grad():
        found = decode_heatmap(model(as_input(fold.frames[:count]).to(device)).cpu())

    side = CANONICAL_SIDE
    sheet = Image.new("RGB", (side * count + 8 * (count - 1), side), (24, 24, 24))
    for index in range(count):
        rgb = fold.frames[index].numpy().transpose(1, 2, 0)
        tile = Image.fromarray(rgb, "RGB")
        draw = ImageDraw.Draw(tile)
        for x, y in fold.centers[index]:
            draw.line([x - 5, y, x + 5, y], fill=(0, 255, 70))
            draw.line([x, y - 5, x, y + 5], fill=(0, 255, 70))
        for x, y, confidence in found[index]:
            radius = 6
            draw.ellipse(
                [x - radius, y - radius, x + radius, y + radius], outline=(255, 80, 80)
            )
            draw.text((x + radius, y - radius), f"{confidence:.2f}", fill=(255, 80, 80))
        sheet.paste(tile, (index * (side + 8), 0))
    sheet.save(path)


def write_report(path: Path, header: dict, reports: list[EpochReport]) -> None:
    """Отчёт одним файлом: его читает человек и ассистент после прогона."""
    path.write_text(
        json.dumps(
            {
                **header,
                "epochs": [asdict(r) for r in reports],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


@dataclass(frozen=True)
class EpochSetup:
    """Всё, что нужно одной эпохе кроме самих данных."""

    optimizer: torch.optim.Optimizer
    device: torch.device
    batch_size: int
    augmentation: Augmentation = NO_AUGMENTATION
    rng: np.random.Generator | None = None


def augmented_batch(
    fold: Fold, batch_index: torch.Tensor, setup: EpochSetup
) -> tuple[torch.Tensor, torch.Tensor]:
    """Кадры и цели одной пачки; при аугментации цели строятся заново.

    Цель нельзя сдвинуть вместе с картинкой: шаг карты центров — 4 пикселя,
    а сдвиг мельче клетки. Поэтому центры двигаются в пикселях, и карта
    строится из них заново.
    """
    if setup.augmentation == NO_AUGMENTATION or setup.rng is None:
        return as_input(fold.frames[batch_index]), fold.targets[batch_index]

    frames = np.empty((len(batch_index), 3, CANONICAL_SIDE, CANONICAL_SIDE), dtype=np.float32)
    targets = np.empty((len(batch_index), 1, HEATMAP_SIDE, HEATMAP_SIDE), dtype=np.float32)
    for position, index in enumerate(batch_index.tolist()):
        frame, centers = augment_frame(
            fold.frames[index].numpy().astype(np.float32) / BYTE_SCALE,
            fold.centers[index],
            setup.rng,
            setup.augmentation,
        )
        frames[position] = frame
        targets[position] = make_heatmap(centers)
    return torch.from_numpy(frames), torch.from_numpy(targets)


def run_epoch(model: CenterDetector, fold: Fold, setup: EpochSetup) -> float:
    """Одна эпоха обучения; возвращает средние потери."""
    model.train()
    optimizer, device, batch_size = setup.optimizer, setup.device, setup.batch_size
    shuffled = torch.randperm(len(fold))
    total = 0.0
    for start in range(0, len(shuffled), batch_size):
        batch_index = shuffled[start : start + batch_size]
        frames, targets = augmented_batch(fold, batch_index, setup)
        batch = frames.to(device)
        target = targets.to(device)
        optimizer.zero_grad()
        loss = focal_heatmap_loss(model(batch), target)
        loss.backward()
        optimizer.step()
        total += float(loss.detach()) * len(batch_index)
    return total / len(shuffled)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("out/dataset"))
    parser.add_argument("--out", type=Path, default=Path("out/detector"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda")
    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="учить на кадрах как есть, без сдвига и перекраски",
    )
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

    frames, targets, centers = load_dataset(args.dataset)
    order = np.random.default_rng(args.seed).permutation(len(frames))
    split = int(len(frames) * (1 - VALIDATION_SHARE))
    train_index, val_index = order[:split], order[split:]
    print(
        f"Кадров: обучение {len(train_index)}, проверка {len(val_index)}; "
        f"чемпионов в проверке {sum(len(centers[i]) for i in val_index)}"
    )

    train = Fold(frames[train_index], targets[train_index], [centers[i] for i in train_index])
    validation = Fold(frames[val_index], targets[val_index], [centers[i] for i in val_index])

    model = CenterDetector().to(device)
    parameters = sum(p.numel() for p in model.parameters())
    print(f"Параметров в модели: {parameters / 1000:.0f} тысяч")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    epoch_setup = EpochSetup(
        optimizer=optimizer,
        device=device,
        batch_size=args.batch_size,
        augmentation=NO_AUGMENTATION if args.no_augment else Augmentation(),
        rng=np.random.default_rng(args.seed + 1),
    )
    print("Аугментации: " + ("выключены" if args.no_augment else "сдвиг, яркость, контраст"))

    reports: list[EpochReport] = []
    best_recall = -1.0
    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        train_loss = run_epoch(model, train, epoch_setup)

        val_loss, metrics = evaluate(model, validation, device, args.batch_size)
        report = EpochReport(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            precision=metrics.precision,
            recall=metrics.recall,
            mean_offset=metrics.mean_offset,
            seconds=time.perf_counter() - started,
        )
        reports.append(report)
        print(
            f"  эпоха {epoch:3d}: потери {train_loss:.3f}/{val_loss:.3f}, "
            f"точность {metrics.precision:.3f}, полнота {metrics.recall:.3f}, "
            f"ошибка центра {metrics.mean_offset:.2f} px, {report.seconds:.0f} с"
        )

        torch.save(model.state_dict(), args.out / "last.pt")
        if metrics.recall > best_recall:
            best_recall = metrics.recall
            torch.save(model.state_dict(), args.out / "best.pt")

    (args.out / "report.json").write_text(
        json.dumps(
            {
                "dataset": str(args.dataset),
                "frames": len(frames),
                "parameters": parameters,
                "device": str(device),
                "epochs": [asdict(r) for r in reports],
                "best_recall": best_recall,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    save_curves(reports, args.out / "curves.png")
    save_predictions(model, validation, device, args.out / "predictions.png")

    final = reports[-1]
    print()
    print(
        f"Лучшая полнота: {best_recall:.3f}; на последней эпохе точность "
        f"{final.precision:.3f}, полнота {final.recall:.3f}, "
        f"ошибка центра {final.mean_offset:.2f} px"
    )
    print(f"Результаты: {args.out.resolve()}")


if __name__ == "__main__":
    main()
