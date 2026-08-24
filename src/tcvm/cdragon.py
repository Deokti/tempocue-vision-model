"""Загрузка circle-иконок чемпионов из Community Dragon с кэшем на диске.

Community Dragon раздаёт распакованные ассеты самой игры. Миникарта рисует
не квадратный портрет Data Dragon, а отдельный circle-арт
(`assets/characters/<чемпион>/hud/<чемпион>_circle_<скин>.png`) — у части
чемпионов он отличается от квадратного, что доказано сверкой отрисовки
(docs/render-verification.md, этап 5). Иконка существует на каждый скин;
базовый скин — суффикс `_0`, у некоторых чемпионов файл без суффикса.

Версия здесь — патч вида «16.16» (без третьего числа сборки Data Dragon),
кэш — data/cdragon/<патч>/. Ресурсы патча неизменяемы, кэш не устаревает.
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

BASE_URL = "https://raw.communitydragon.org"
DEFAULT_CACHE = Path(__file__).resolve().parents[2] / "data" / "cdragon"
_TIMEOUT_SECONDS = 60


def patch_of(ddragon_version: str) -> str:
    """«16.16.1» Data Dragon → «16.16» Community Dragon."""
    return ".".join(ddragon_version.split(".")[:2])


def _download(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "tempocue-vision-model"})
    last_error: Exception | None = None
    for _ in range(3):  # зеркало временами обрывает соединение посреди ответа
        try:
            with urllib.request.urlopen(request, timeout=_TIMEOUT_SECONDS) as response:
                return response.read()
        except (TimeoutError, OSError) as error:
            last_error = error
    raise last_error


def _hud_listing(champion_id: str, patch: str, cache_dir: Path) -> list[str]:
    cache_path = cache_dir / patch / "hud-listing" / f"{champion_id.lower()}.json"
    if not cache_path.exists():
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(
            _download(
                f"{BASE_URL}/json/{patch}/game/assets/characters/{champion_id.lower()}/hud/"
            )
        )
    entries = json.loads(cache_path.read_text(encoding="utf-8"))
    return [e["name"] for e in entries]


def circle_icon_names(
    champion_id: str, patch: str, cache_dir: Path = DEFAULT_CACHE
) -> list[str]:
    """Имена всех circle-иконок чемпиона (базовый скин и варианты)."""
    return sorted(
        name
        for name in _hud_listing(champion_id, patch, cache_dir)
        if "circle" in name.lower() and name.endswith(".png")
    )


def base_circle_bgra(
    champion_id: str, patch: str, cache_dir: Path = DEFAULT_CACHE
) -> np.ndarray:
    """Circle-иконка базового скина как массив BGRA."""
    lower = champion_id.lower()
    names = circle_icon_names(champion_id, patch, cache_dir)
    for candidate in (f"{lower}_circle_0.png", f"{lower}_circle.png"):
        if candidate in names:
            name = candidate
            break
    else:
        if not names:
            raise FileNotFoundError(f"У {champion_id} нет circle-иконок в патче {patch}")
        name = names[0]

    cache_path = cache_dir / patch / "circle" / name
    if not cache_path.exists():
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(
            _download(f"{BASE_URL}/{patch}/game/assets/characters/{lower}/hud/{name}")
        )
    return read_png_bgra(cache_path)


def read_png_bgra(path: Path) -> np.ndarray:
    """Читает PNG как BGRA, удаляя файл, если он оборван.

    Обрывы зеркала оставляли в кэше усечённые картинки, и падение всплывало
    много позже — при генерации датасета. Битый кэш чинится перекачиванием.
    """
    try:
        with Image.open(path) as image:
            image.load()
            rgba = np.asarray(image.convert("RGBA"))
    except OSError as error:
        path.unlink(missing_ok=True)
        raise OSError(f"Битый файл кэша удалён, повтори запуск: {path}") from error
    return rgba[..., [2, 1, 0, 3]].copy()
