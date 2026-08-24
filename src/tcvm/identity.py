"""Сеть опознания «кто это»: вырез значка → чемпион или отказ.

Устройство выбрано так, чтобы простой вариант превращался в сложный малой
правкой. Ствол сети переводит вырез в **вектор** фиксированной длины; поверх
вектора стоит обычный линейный слой на число классов. Классификатор — это
ствол плюс этот слой, вложение (embedding) — тот же ствол без него, где
опознание идёт сравнением вектора с прототипами. Менять придётся последний
слой и функцию потерь, а не сеть.

Отказ сделан отдельным классом, а не порогом уверенности. Порог отвечает
«сеть не уверена», а нам нужно «в вырезе нет чемпиона» — это разные
утверждения, и второе можно показать сети на примерах: постройки, лагеря,
волны миньонов, пустая земля.

Состав матча известен из API, поэтому при опознании выбор идёт из десяти
кандидатов и отказа, а не из всех классов. Замер планки показал, что знание
состава стоит 7,6 пункта (`docs/identity.md`), и терять их незачем.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as torch_functional

# Вырез берётся в канонических пикселях: значок занимает 25 px, остаток даёт
# окружение и терпимость к неточному центру от детектора.
CANONICAL_CROP = 32
# Вход сети. Он фиксирован, а резкость вырезов — нет: игрок выбирает размер
# миникарты от ~200 до ~600, и значок несёт от 16 до 47 настоящих пикселей.
# Вход взят по верхнему краю диапазона (при карте 600 вырез занимает 60 px),
# чтобы у тех, у кого карта большая, детали не выбрасывались повторно.
INPUT_SIDE = 64
# Длина вектора, в который ствол переводит вырез. Столько же будет у вложения.
EMBEDDING_SIZE = 128
# Метка отсутствия чемпиона; в обучении это отдельный класс с индексом 0.
NO_CHAMPION = ""
REJECT_INDEX = 0
# Шкала перевода байтов изображения в доли единицы.
BYTE_SCALE = 255.0


@dataclass(frozen=True)
class IdentityMetrics:
    """Итоги проверки: доля верных отдельно по значкам и по отказам."""

    champions_right: int
    champions_total: int
    rejects_right: int
    rejects_total: int

    @property
    def champion_accuracy(self) -> float:
        return self.champions_right / self.champions_total if self.champions_total else 0.0

    @property
    def reject_accuracy(self) -> float:
        return self.rejects_right / self.rejects_total if self.rejects_total else 0.0


class IdentityNet(nn.Module):
    """Вырез INPUT_SIDE → вектор длины EMBEDDING_SIZE → баллы по классам."""

    def __init__(self, classes: int, width: int = 32):
        super().__init__()

        def block(inputs: int, outputs: int, stride: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(inputs, outputs, 3, stride=stride, padding=1),
                nn.BatchNorm2d(outputs),
                nn.ReLU(inplace=True),
            )

        self.trunk = nn.Sequential(
            block(3, width, 1),  # 64
            block(width, width * 2, 2),  # 32
            block(width * 2, width * 2, 1),
            block(width * 2, width * 4, 2),  # 16
            block(width * 4, width * 4, 1),
            block(width * 4, width * 8, 2),  # 8
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(width * 8, EMBEDDING_SIZE),
        )
        self.head = nn.Linear(EMBEDDING_SIZE, classes)

    def embed(self, batch: torch.Tensor) -> torch.Tensor:
        """Вектор выреза — то, что останется от сети при переходе к вложению."""
        return self.trunk(batch)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return self.head(self.embed(batch))


def as_input(crops: torch.Tensor) -> torch.Tensor:
    """Вырезы (N, H, W, 3) uint8 → тензор (N, 3, INPUT_SIDE, INPUT_SIDE).

    Вырез любого размера приводится ко входу сети. Резкость при этом остаётся
    той, какую дала игра: вырез с большой карты растягивается слабо и остаётся
    подробным, с маленькой — растягивается сильно и остаётся мутным. Сеть
    видит одинаковый размер и разную резкость, а не наоборот.
    """
    batch = crops.permute(0, 3, 1, 2).float() / BYTE_SCALE
    if batch.shape[-1] != INPUT_SIDE or batch.shape[-2] != INPUT_SIDE:
        batch = torch_functional.interpolate(
            batch, size=(INPUT_SIDE, INPUT_SIDE), mode="bilinear", align_corners=False
        )
    return batch


def build_vocabulary(champions: list[str]) -> list[str]:
    """Список классов: отказ первым, дальше чемпионы по алфавиту."""
    return [NO_CHAMPION, *sorted(set(champions) - {NO_CHAMPION})]


def choose(
    logits: torch.Tensor, vocabulary: list[str], roster: list[str] | None = None
) -> list[str]:
    """Решение по каждому вырезу: имя чемпиона или отказ.

    Если состав матча известен, выбор идёт только из него и отказа — так
    работает приложение, и так же надо мерить. Прочие классы не подавляются,
    а просто не рассматриваются: это ровно то, что даёт знание состава.
    """
    allowed = set(range(len(vocabulary)))
    if roster is not None:
        wanted = {name.lower() for name in roster}
        allowed = {
            index
            for index, name in enumerate(vocabulary)
            if index == REJECT_INDEX or name.lower() in wanted
        }
    mask = torch.full_like(logits, float("-inf"))
    for index in sorted(allowed):
        mask[:, index] = logits[:, index]
    return [vocabulary[index] for index in mask.argmax(dim=1).tolist()]


def evaluate_choices(chosen: list[str], truth: list[str]) -> IdentityMetrics:
    """Сводка по решениям: значки и отказы считаются раздельно."""
    champions_right = champions_total = rejects_right = rejects_total = 0
    for taken, wanted in zip(chosen, truth, strict=True):
        if wanted == NO_CHAMPION:
            rejects_total += 1
            rejects_right += taken == NO_CHAMPION
        else:
            champions_total += 1
            champions_right += taken.lower() == wanted.lower()
    return IdentityMetrics(champions_right, champions_total, rejects_right, rejects_total)


def augment_crops(
    crops: np.ndarray, rng: np.random.Generator, brightness: tuple[float, float]
) -> np.ndarray:
    """Разброс яркости и контраста — те же оси, что у детектора.

    Смещение центра здесь не нужно: оно уже внесено при нарезке вырезов, где
    известно настоящее положение значка.
    """
    scaled = crops.astype(np.float32)
    factors = rng.uniform(*brightness, size=(len(crops), 1, 1, 1)).astype(np.float32)
    means = scaled.mean(axis=(1, 2, 3), keepdims=True)
    return np.clip((scaled - means) * factors + means * factors, 0.0, BYTE_SCALE)
