"""Чтение форматов TempoCue: `.tempocue-vision` (кадр) и `.tempocue-sequence` (запись).

Оба формата — обычные ZIP-архивы. Пиксели хранятся сырыми, без сжатия в формат
картинки: BGRA32, четыре байта на пиксель, строки подряд с шагом (stride) из
манифеста. Рядом с архивом может лежать файл ручной разметки `<имя>.labels.json`.
"""

from __future__ import annotations

import json
import math
import re
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

BYTES_PER_PIXEL = 4

AFFILIATION_ALLY = 0
AFFILIATION_ENEMY = 1


def decode_bgra(data: bytes, width: int, height: int, stride: int) -> np.ndarray:
    """Сырые байты BGRA32 → массив (высота, ширина, 4) uint8."""
    expected = stride * height
    if len(data) < expected:
        raise ValueError(
            f"BGRA32 короче ожидаемого: {len(data)} байт при {expected} требуемых")
    rows = np.frombuffer(data, np.uint8, count=expected).reshape(height, stride)
    return rows[:, : width * BYTES_PER_PIXEL] \
        .reshape(height, width, BYTES_PER_PIXEL).copy()


def bgra_to_rgb(pixels: np.ndarray) -> np.ndarray:
    """BGRA → RGB (альфа отбрасывается): Pillow и матплотлибы ждут RGB."""
    return pixels[..., [2, 1, 0]]


def parse_timestamp(value: str) -> datetime:
    """ISO-время из манифестов; дробная часть секунд бывает длиннее шести цифр."""
    trimmed = re.sub(r"\.(\d{6})\d+", r".\1", value)
    return datetime.fromisoformat(trimmed)


@dataclass(frozen=True)
class ChampionReference:
    """Эталон значка, каким его видел конвейер в момент сохранения кадра."""

    champion_id: str
    affiliation: int
    pixels: np.ndarray  # (сторона, сторона, 4) BGRA


@dataclass(frozen=True)
class LabeledRegion:
    """Одна область ручной разметки в канонической системе кадра 320×320."""

    x: int
    y: int
    width: int
    height: int
    kind: str
    champion_id: str | None
    affiliation: str | None


@dataclass(frozen=True)
class FrameLabels:
    regions: tuple[LabeledRegion, ...]
    allowed_false_positives: int
    allowed_missed_champions: int

    @property
    def champions(self) -> tuple[LabeledRegion, ...]:
        return tuple(r for r in self.regions if r.kind == "Champion")

    @property
    def hard_negatives(self) -> tuple[LabeledRegion, ...]:
        return tuple(r for r in self.regions if r.kind != "Champion")


@dataclass(frozen=True)
class ReplayFrame:
    """Один сохранённый кадр миникарты с эталонами и (если есть) разметкой."""

    name: str
    path: Path
    captured_at: datetime
    pixels: np.ndarray  # (высота, ширина, 4) BGRA
    references: tuple[ChampionReference, ...]
    labels: FrameLabels | None

    def crop(self, region: LabeledRegion) -> np.ndarray:
        return self.pixels[
            region.y : region.y + region.height,
            region.x : region.x + region.width,
        ]


@dataclass(frozen=True)
class ReplaySequence:
    """Запись подряд идущих кадров одной сцены; разметка — список видимых."""

    name: str
    path: Path
    frames: tuple[np.ndarray, ...]
    captured_at: tuple[datetime, ...]
    visible_throughout: tuple[str, ...]
    notes: str | None

    @property
    def duration_seconds(self) -> float:
        return (self.captured_at[-1] - self.captured_at[0]).total_seconds()


def _load_labels_json(archive_path: Path) -> dict | None:
    labels_path = archive_path.with_name(archive_path.name + ".labels.json")
    if not labels_path.exists():
        return None
    return json.loads(labels_path.read_text(encoding="utf-8"))


def _reference_from_entry(archive: zipfile.ZipFile, entry: dict) -> ChampionReference:
    data = archive.read(entry["entryName"])
    side = math.isqrt(len(data) // BYTES_PER_PIXEL)
    if side * side * BYTES_PER_PIXEL != len(data):
        raise ValueError(f"Эталон {entry['entryName']} не квадратный: {len(data)} байт")
    pixels = decode_bgra(data, side, side, side * BYTES_PER_PIXEL)
    return ChampionReference(
        champion_id=entry["championId"],
        affiliation=entry["affiliation"],
        pixels=pixels,
    )


def load_frame(path: Path) -> ReplayFrame:
    """Читает `.tempocue-vision` и разметку рядом, если она есть."""
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        pixels = decode_bgra(
            archive.read(manifest["frameEntryName"]),
            manifest["imageWidth"],
            manifest["imageHeight"],
            manifest["imageStride"],
        )
        references = tuple(
            _reference_from_entry(archive, entry)
            for entry in manifest.get("references", [])
        )

    labels_json = _load_labels_json(path)
    labels = None
    if labels_json is not None:
        labels = FrameLabels(
            regions=tuple(
                LabeledRegion(
                    x=r["x"],
                    y=r["y"],
                    width=r["width"],
                    height=r["height"],
                    kind=r["kind"],
                    champion_id=r.get("championId"),
                    affiliation=r.get("affiliation"),
                )
                for r in labels_json.get("regions", [])
            ),
            allowed_false_positives=labels_json.get("allowedFalsePositives", 0),
            allowed_missed_champions=labels_json.get("allowedMissedChampions", 0),
        )

    return ReplayFrame(
        name=path.name.removesuffix(".tempocue-vision"),
        path=path,
        captured_at=parse_timestamp(manifest["capturedAt"]),
        pixels=pixels,
        references=references,
        labels=labels,
    )


def load_sequence(path: Path) -> ReplaySequence:
    """Читает `.tempocue-sequence` и разметку рядом, если она есть."""
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        width = manifest["imageWidth"]
        height = manifest["imageHeight"]
        stride = manifest["imageStride"]
        frames: list[np.ndarray] = []
        captured_at: list[datetime] = []
        for entry in manifest["frames"]:
            frames.append(decode_bgra(archive.read(entry["entryName"]), width, height, stride))
            captured_at.append(parse_timestamp(entry["capturedAt"]))

    labels_json = _load_labels_json(path)
    visible = tuple((labels_json or {}).get("visibleThroughout", []))
    notes = (labels_json or {}).get("notes")

    return ReplaySequence(
        name=path.name.removesuffix(".tempocue-sequence"),
        path=path,
        frames=tuple(frames),
        captured_at=tuple(captured_at),
        visible_throughout=visible,
        notes=notes,
    )


def load_corpus(directory: Path) -> list[ReplayFrame]:
    return [load_frame(p) for p in sorted(directory.glob("*.tempocue-vision"))]


def load_sequences(directory: Path) -> list[ReplaySequence]:
    return [load_sequence(p) for p in sorted(directory.glob("*.tempocue-sequence"))]
