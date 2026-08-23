"""Подбор параметров отрисовки: может ли исходный арт объяснить настоящий вырез.

Этапы 5-6 сверки отрисовки. Для каждого размеченного значка корпуса
перебираются параметры отрисовки (docs/render-verification.md) по двум
источникам арта независимо:

- square — квадратный портрет Data Dragon;
- circle — circle-иконка базового скина из ассетов игры (Community Dragon);
  именно её рисует миникарта, у части чемпионов она отличается от квадрата.

Побеждает источник с меньшим остатком. Два режима подбора:

- индивидуальный: каждому вырезу свои параметры — «способен ли источник
  объяснить эти пиксели вообще»;
- общий: один набор параметров на все чистые вырезы (субпиксельный сдвиг
  остаётся индивидуальным — фаза значка на сетке у каждого своя) — «можно ли
  эти параметры зашить в генератор»; считается по circle-источнику.

Перед подбором центр каждого выреза уточняется совмещением с наивным синтезом
источника в окне +-3 px: центры ручных рамок гуляют, и без этого подбор чинил
бы разметку сдвигом вместо отрисовки.

Заведомо испорченные вырезы (перекрытия, свечение, край карты) перечислены в
annotations/corpus-exclusions.json и в сводку порогов не входят; подбираются
и печатаются они наравне с чистыми — как отрицательный контроль.

Запуск: .venv/Scripts/python tools/fit_render.py
Тройки «настоящий | синтез | разница» — в out/render-fit/, имя файла
начинается с остатка, поэтому сортировка по имени = сортировка по качеству.
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np
from PIL import Image

from tcvm.cdragon import base_circle_bgra, patch_of
from tcvm.ddragon import latest_version, portrait_bgra
from tcvm.formats import load_corpus
from tcvm.matching import (
    CENTER_SIDE,
    ICON_SIDE,
    INNER_RADIUS,
    center_square_mask,
    circular_mask,
    crop_centered,
    find_best_match,
)
from tcvm.render import RenderParams, render_icon

DEFAULT_DATA_ROOT = Path(
    r"C:\Users\deokn\.codex\worktrees\739a\PROJECT\tests\TempoCue.Vision.Tests"
)
EXCLUSIONS_PATH = Path(__file__).resolve().parents[1] / "annotations" / "corpus-exclusions.json"
UPSCALE = 8
THRESHOLD = 0.20
ALIGN_RADIUS = 3

INNER_MASK = circular_mask(ICON_SIDE, INNER_RADIUS)
CENTER_MASK = center_square_mask(ICON_SIDE, CENTER_SIDE)

COARSE_CROP_FRAC = (0.80, 0.85, 0.90, 0.95, 1.00)
COARSE_NATIVE = (22, 27, 32, 37, 43)
COARSE_RESAMPLER = ("bilinear", "bicubic", "lanczos", "box")
COARSE_GAMMA = ("srgb", "linear")
COARSE_BLUR = (0.0, 0.6)
COARSE_SHIFT = (0.0, 0.5)
COARSE_GRID = list(
    product(COARSE_CROP_FRAC, COARSE_NATIVE, COARSE_RESAMPLER, COARSE_GAMMA, COARSE_BLUR)
)

NAIVE = RenderParams(1.0, ICON_SIDE, "box", "srgb", 0.0, "bilinear", 0.0, 0.0)


def prepared_vector(crop: np.ndarray, mask: np.ndarray) -> np.ndarray:
    v = crop[mask][:, :3].astype(np.float64).ravel()
    v -= v.mean()
    norm = np.linalg.norm(v)
    return v / norm if norm else v


def ncc_against(prepared: np.ndarray, candidate: np.ndarray, mask: np.ndarray) -> float:
    v = candidate[mask][:, :3].astype(np.float64).ravel()
    v -= v.mean()
    norm = np.linalg.norm(v)
    if norm == 0:
        return 0.0
    return float(prepared @ v / norm)


def coarse_scores(portrait: np.ndarray, prepared: np.ndarray) -> np.ndarray:
    """Лучший балл каждого грубого конфига (по грубым субпиксельным сдвигам)."""
    scores = np.full(len(COARSE_GRID), -2.0)
    for index, (frac, native, resampler, gamma, blur) in enumerate(COARSE_GRID):
        for dx, dy in product(COARSE_SHIFT, repeat=2):
            params = RenderParams(frac, native, resampler, gamma, blur, "bilinear", dx, dy)
            score = ncc_against(prepared, render_icon(portrait, params), INNER_MASK)
            scores[index] = max(scores[index], score)
    return scores


def refine_shift(
    portrait: np.ndarray, base: RenderParams, prepared: np.ndarray
) -> tuple[RenderParams, float]:
    """Уточняет субпиксельный сдвиг и финальный ресемплер при прочих равных."""
    best = (base, -2.0)
    for final in ("bilinear", "bicubic"):
        for dx, dy in product(np.arange(0.0, 1.0, 0.125), repeat=2):
            params = RenderParams(
                base.crop_frac,
                base.native_size,
                base.resampler,
                base.gamma,
                base.blur_sigma,
                final,
                float(dx),
                float(dy),
            )
            score = ncc_against(prepared, render_icon(portrait, params), INNER_MASK)
            if score > best[1]:
                best = (params, score)
    return best


@dataclass(frozen=True)
class FitResult:
    """Итог подбора по одному вырезу; поля читаются по имени, а не по индексу."""

    champion: str
    frame: str
    source: str  # "square" или "circle"
    residual: float  # по внутреннему диску
    center_residual: float
    is_clean: bool


def fit_source(
    portrait: np.ndarray, frame_pixels: np.ndarray, center: tuple[int, int]
) -> tuple[np.ndarray, RenderParams, float, np.ndarray]:
    """Выравнивание, грубая сетка и уточнение для одного источника арта."""
    naive = render_icon(portrait, NAIVE)
    x, y, _ = find_best_match(frame_pixels, naive, INNER_MASK, center, ALIGN_RADIUS)
    real = crop_centered(frame_pixels, x, y)
    prepared = prepared_vector(real, INNER_MASK)
    scores = coarse_scores(portrait, prepared)
    best_params, best_score = None, -2.0
    for index in np.argsort(scores)[-5:]:
        frac, native, resampler, gamma, blur = COARSE_GRID[index]
        base = RenderParams(frac, native, resampler, gamma, blur, "bilinear", 0.0, 0.0)
        params, score = refine_shift(portrait, base, prepared)
        if score > best_score:
            best_params, best_score = params, score
    return real, best_params, best_score, scores


def triptych(real: np.ndarray, synth: np.ndarray, path: Path) -> None:
    diff = np.abs(real[..., :3].astype(np.int16) - synth.astype(np.int16))
    diff = np.clip(diff * 3, 0, 255).astype(np.uint8)
    cell = ICON_SIDE * UPSCALE
    sheet = Image.new("RGB", (cell * 3 + 16, cell), (24, 24, 24))
    for column, bgr in enumerate((real[..., :3], synth, diff)):
        image = Image.fromarray(bgr[..., ::-1], "RGB").resize((cell, cell), Image.NEAREST)
        sheet.paste(image, (column * (cell + 8), 0))
    sheet.save(path)


def describe(values: list[float]) -> str:
    return (
        f"медиана {statistics.median(values):.3f}, "
        f"p95 {np.percentile(values, 95):.3f}, максимум {max(values):.3f}"
    )


def load_exclusions() -> dict[tuple[str, str], str]:
    """Чистое множество: (чемпион, кадр) → причина исключения."""
    if not EXCLUSIONS_PATH.exists():
        return {}
    data = json.loads(EXCLUSIONS_PATH.read_text(encoding="utf-8"))
    return {(e["championId"], e["frame"]): e["reason"] for e in data["exclusions"]}


def fit_corpus(
    data_root: Path, out_root: Path, version: str, patch: str
) -> tuple[list[FitResult], list[np.ndarray], list[int]]:
    """Подбирает параметры по каждому вырезу корпуса и печатает ход работы.

    Возвращает итоги, грубые баллы circle-источника и номера чистых вырезов
    среди них — последние два нужны для расчёта общего набора параметров.
    """
    excluded = load_exclusions()
    squares: dict[str, np.ndarray] = {}
    circles: dict[str, np.ndarray | None] = {}
    results: list[FitResult] = []
    circle_scores: list[np.ndarray] = []
    clean_indices: list[int] = []

    for frame in load_corpus(data_root / "ReplayCorpus"):
        for region in frame.labels.champions if frame.labels else ():
            champion = region.champion_id
            if champion not in squares:
                squares[champion] = portrait_bgra(champion, version)
                try:
                    circles[champion] = base_circle_bgra(champion, patch)
                except (OSError, ValueError, FileNotFoundError) as error:
                    circles[champion] = None
                    print(f"  нет circle-иконки у {champion}: {error}")
            center = (region.x + region.width // 2, region.y + region.height // 2)

            fits = {}
            try:
                fits["square"] = fit_source(squares[champion], frame.pixels, center)
            except ValueError:
                print(f"  пропущен у края кадра: {champion} в {frame.name}")
                continue
            if circles[champion] is not None:
                fits["circle"] = fit_source(circles[champion], frame.pixels, center)

            source = max(fits, key=lambda name: fits[name][2])
            real, params, score, _ = fits[source]
            synth = render_icon(
                squares[champion] if source == "square" else circles[champion], params
            )
            result = FitResult(
                champion=champion,
                frame=frame.name,
                source=source,
                residual=1.0 - score,
                center_residual=1.0
                - ncc_against(prepared_vector(real, CENTER_MASK), synth, CENTER_MASK),
                is_clean=(champion, frame.name) not in excluded,
            )
            results.append(result)

            if "circle" in fits:
                if result.is_clean:
                    clean_indices.append(len(circle_scores))
                circle_scores.append(fits["circle"][3])

            triptych(
                real,
                synth,
                out_root / f"{result.residual:.3f}-{champion}-{frame.name}-{source}.png",
            )
            mark = "" if result.is_clean else f"  [исключён: {excluded[champion, frame.name]}]"
            print(
                f"  {champion} @ {frame.name}: {source} {result.residual:.3f}, "
                f"центр {result.center_residual:.3f}{mark}"
            )
    return results, circle_scores, clean_indices


def print_verdict(results: list[FitResult]) -> None:
    """Сводка по чистому множеству и отрицательному контролю."""
    clean = [r for r in results if r.is_clean]
    dirty = [r for r in results if not r.is_clean]
    clean_residuals = sorted(r.residual for r in clean)
    inside = sum(1 for r in clean_residuals if r <= THRESHOLD)

    print()
    print(f"Чистые вырезы: n={len(clean_residuals)}, {describe(clean_residuals)}")
    print(
        f"Не выше порога {THRESHOLD}: {inside} из {len(clean_residuals)} "
        f"({100 * inside / len(clean_residuals):.0f} %)"
    )
    print(
        f"Источник у чистых: square {sum(1 for r in clean if r.source == 'square')}, "
        f"circle {sum(1 for r in clean if r.source == 'circle')}"
    )
    if dirty:
        residuals = sorted(r.residual for r in dirty)
        print(f"Исключённые (контроль): n={len(dirty)}, {describe(residuals)}")


def print_shared_params(circle_scores: list[np.ndarray], clean_indices: list[int]) -> None:
    """Один набор параметров на все чистые вырезы — умолчания генератора."""
    stacked = np.stack(circle_scores)  # (вырезы, конфиги)
    medians = np.median(1.0 - stacked[clean_indices], axis=0)
    best = int(np.argmin(medians))
    frac, native, resampler, gamma, blur = COARSE_GRID[best]
    print()
    print(
        f"Общий набор для генератора (circle, чистые, грубые сдвиги): "
        f"crop {frac:.2f}, native {native}, {resampler}, {gamma}, blur {blur} "
        f"— медиана остатка {medians[best]:.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out", type=Path, default=Path("out/render-fit"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    version = latest_version()
    patch = patch_of(version)
    print(f"Data Dragon {version}, Community Dragon {patch}")

    results, circle_scores, clean_indices = fit_corpus(args.data, args.out, version, patch)
    print_verdict(results)
    print_shared_params(circle_scores, clean_indices)
    print(f"Тройки: {args.out.resolve()}")


if __name__ == "__main__":
    main()
