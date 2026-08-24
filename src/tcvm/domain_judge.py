"""Судья доменов: сеть, отличающая синтетический кадр от настоящего.

Проверка из плана программы: если маленькая сеть легко отличает синтетику от
реальности, домены разошлись и генератор нужно чинить до всякого обучения
распознаванию. Успех судьи — плохая новость, и в этом весь смысл: мы обучаем
сеть на задачу, в которой хотим её провала.

Устройство:

- данные — квадратные вырезы (патчи) из кадров, а не кадры целиком: их много
  с каждого кадра, и судья вынужден смотреть на текстуру, а не на композицию;
- вырезы берутся из внутренней области, без рамки интерфейса по краю: рамку
  композитор не рисует, и судья опознавал бы домен по ней, а не по существу;
- разделение обучающей и проверочной выборок — **по кадрам**, а не по патчам:
  патчи одного кадра похожи, и попав в обе выборки, дали бы утечку;
- сеть намеренно маленькая: судья должен ловить систематическую разницу, а не
  запоминать конкретные кадры.

Точность около 50 % (монетка) означает неразличимые домены; около 100 % —
разрыв, который надо объяснить и починить.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

PATCH_SIDE = 48
# Отступ от края кадра: рамка интерфейса и подписи занимают около 12 px,
# берём с запасом.
FRAME_MARGIN = 20
PATCHES_PER_FRAME = 48

LABEL_REAL = 0
LABEL_SYNTHETIC = 1


@dataclass(frozen=True)
class TrainingSetup:
    """Настройки обучения судьи; значения по умолчанию — рабочие."""

    epochs: int = 12
    batch_size: int = 64
    learning_rate: float = 1e-3
    seed: int = 0


@dataclass(frozen=True)
class Split:
    """Пара выборок: обучающая и проверочная, уже разделённые по кадрам."""

    train_x: torch.Tensor
    train_y: torch.Tensor
    val_x: torch.Tensor
    val_y: torch.Tensor


def cut_patches(
    frame_bgr: np.ndarray, rng: np.random.Generator, count: int = PATCHES_PER_FRAME
) -> np.ndarray:
    """Случайные патчи из внутренней области кадра, (count, 3, side, side)."""
    height, width = frame_bgr.shape[:2]
    high_y = height - FRAME_MARGIN - PATCH_SIDE
    high_x = width - FRAME_MARGIN - PATCH_SIDE
    patches = np.empty((count, PATCH_SIDE, PATCH_SIDE, 3), dtype=np.uint8)
    for index in range(count):
        y = int(rng.integers(FRAME_MARGIN, high_y))
        x = int(rng.integers(FRAME_MARGIN, high_x))
        patches[index] = frame_bgr[y : y + PATCH_SIDE, x : x + PATCH_SIDE, :3]
    # BGR uint8 (N, H, W, C) → RGB float32 (N, C, H, W) в диапазоне [0, 1].
    return np.transpose(patches[..., ::-1], (0, 3, 1, 2)).astype(np.float32) / 255.0


class DomainJudge(nn.Module):
    """Три свёрточных блока и линейный выход на два класса."""

    def __init__(self, channels: int = 16):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(channels, channels * 2, 3, padding=1),
            nn.BatchNorm2d(channels * 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(channels * 2, channels * 4, 3, padding=1),
            nn.BatchNorm2d(channels * 4),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(channels * 4, 2)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(batch).flatten(1))


def build_dataset(
    real_frames: list[np.ndarray],
    synthetic_frames: list[np.ndarray],
    rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Патчи обоих доменов с метками; порядок кадров сохраняется."""
    patches = []
    labels = []
    for frames, label in ((real_frames, LABEL_REAL), (synthetic_frames, LABEL_SYNTHETIC)):
        for frame in frames:
            cut = cut_patches(frame, rng)
            patches.append(cut)
            labels.append(np.full(len(cut), label, dtype=np.int64))
    return (
        torch.from_numpy(np.concatenate(patches)),
        torch.from_numpy(np.concatenate(labels)),
    )


def train_judge(
    split: Split, setup: TrainingSetup | None = None
) -> tuple[DomainJudge, list[float], list[float]]:
    """Обучает судью; возвращает модель, потери по эпохам и точность на проверке."""
    setup = setup or TrainingSetup()
    torch.manual_seed(setup.seed)
    model = DomainJudge()
    optimizer = torch.optim.Adam(model.parameters(), lr=setup.learning_rate)
    loss_function = nn.CrossEntropyLoss()

    losses: list[float] = []
    accuracies: list[float] = []
    for _ in range(setup.epochs):
        model.train()
        order = torch.randperm(len(split.train_x))
        epoch_loss = 0.0
        for start in range(0, len(order), setup.batch_size):
            batch = order[start : start + setup.batch_size]
            optimizer.zero_grad()
            loss = loss_function(model(split.train_x[batch]), split.train_y[batch])
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach()) * len(batch)
        losses.append(epoch_loss / len(order))

        model.eval()
        with torch.no_grad():
            predictions = model(split.val_x).argmax(dim=1)
        accuracies.append(float((predictions == split.val_y).float().mean()))
    return model, losses, accuracies
