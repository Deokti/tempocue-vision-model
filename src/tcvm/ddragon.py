"""Загрузка портретов чемпионов из Data Dragon с кэшем на диске.

Data Dragon — официальный каталог ресурсов Riot: по HTTP отдаются список
версий (совпадают с патчами игры), реестр чемпионов и их портреты 120x120.
Всё скачанное кэшируется в data/ddragon/<версия>/ и второй раз не качается:
ресурсы версии неизменяемы, поэтому кэш не протухает. Версия патча — часть
пути кэша и обязана попадать в каждый отчёт, который на этих данных построен.

Зависимости только стандартные: HTTP — urllib, декодирование PNG — Pillow.
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

BASE_URL = "https://ddragon.leagueoflegends.com"
DEFAULT_CACHE = Path(__file__).resolve().parents[2] / "data" / "ddragon"
_TIMEOUT_SECONDS = 30


def _download(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "tempocue-vision-model"})
    with urllib.request.urlopen(request, timeout=_TIMEOUT_SECONDS) as response:
        return response.read()


def latest_version() -> str:
    """Свежайшая версия Data Dragon; список версий не кэшируется намеренно.

    Версии выходят каждые ~2 недели, и устаревший ответ здесь опаснее лишнего
    HTTP-запроса: сверка с прошлым патчем прошла бы молча.
    """
    versions = json.loads(_download(f"{BASE_URL}/api/versions.json"))
    return versions[0]


def champion_ids(version: str, cache_dir: Path = DEFAULT_CACHE) -> list[str]:
    """Список идентификаторов чемпионов версии (как в разметке корпуса)."""
    cache_path = cache_dir / version / "champion.json"
    if not cache_path.exists():
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(_download(f"{BASE_URL}/cdn/{version}/data/en_US/champion.json"))
    data = json.loads(cache_path.read_text(encoding="utf-8"))
    return sorted(data["data"].keys())


def portrait_bgra(
    champion_id: str, version: str, cache_dir: Path = DEFAULT_CACHE
) -> np.ndarray:
    """Портрет чемпиона 120x120 как массив BGRA — в соглашении проекта."""
    cache_path = cache_dir / version / "img" / f"{champion_id}.png"
    if not cache_path.exists():
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(
            _download(f"{BASE_URL}/cdn/{version}/img/champion/{champion_id}.png")
        )
    with Image.open(cache_path) as image:
        rgba = np.asarray(image.convert("RGBA"))
    return rgba[..., [2, 1, 0, 3]].copy()
