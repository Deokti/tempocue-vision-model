"""Подбор параметров отрисовки: может ли Data Dragon объяснить настоящий вырез.

Этап 5 сверки отрисовки. Для каждого размеченного значка корпуса перебираются
параметры отрисовки (docs/render-verification.md), максимизируя NCC синтеза с
настоящим вырезом по внутреннему диску. Два режима:

- индивидуальный: каждому вырезу свои параметры — «способен ли источник
  объяснить эти пиксели вообще»;
- общий: один набор параметров на все вырезы (субпиксельный сдвиг остаётся
  индивидуальным — фаза значка на сетке у каждого своя) — «можно ли эти
  параметры зашить в генератор».

Перед подбором центр каждого выреза уточняется совмещением с наивным синтезом
в окне +-3 px: центры ручных рамок гуляют, и без этого подбор чинил бы
разметку сдвигом вместо отрисовки.

Запуск: .venv/Scripts/python tools/fit_render.py
Тройки «настоящий | синтез | разница» — в out/render-fit/, имя файла
начинается с остатка, поэтому сортировка по имени = сортировка по качеству.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from itertools import product
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tcvm.ddragon import latest_version, portrait_bgra  # noqa: E402
from tcvm.formats import bgra_to_rgb, load_corpus  # noqa: E402
from tcvm.matching import (  # noqa: E402
    CENTER_SIDE,
    ICON_SIDE,
    INNER_RADIUS,
    center_square_mask,
    circular_mask,
    crop_centered,
    find_best_match,
)
from tcvm.render import RenderParams, render_icon  # noqa: E402

DEFAULT_DATA_ROOT = Path(
    r"C:\Users\deokn\.codex\worktrees\739a\PROJECT\tests\TempoCue.Vision.Tests")
UPSCALE = 8

INNER_MASK = circular_mask(ICON_SIDE, INNER_RADIUS)
CENTER_MASK = center_square_mask(ICON_SIDE, CENTER_SIDE)

# Грубая сетка. Субпиксель в ней намеренно редкий: он уточняется отдельно.
COARSE_CROP_FRAC = (0.80, 0.85, 0.90, 0.95, 1.00)
COARSE_NATIVE = (22, 27, 32, 37, 43)
COARSE_RESAMPLER = ("bilinear", "bicubic", "lanczos", "box")
COARSE_GAMMA = ("srgb", "linear")
COARSE_BLUR = (0.0, 0.6)
COARSE_SHIFT = (0.0, 0.5)

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


def refine_shift(portrait: np.ndarray, base: RenderParams,
                 prepared: np.ndarray) -> tuple[RenderParams, float]:
    """Уточняет субпиксельный сдвиг и финальный ресемплер при прочих равных."""
    best = (base, -2.0)
    for final in ("bilinear", "bicubic"):
        for dx, dy in product(np.arange(0.0, 1.0, 0.125), repeat=2):
            params = RenderParams(base.crop_frac, base.native_size, base.resampler,
                                  base.gamma, base.blur_sigma, final,
                                  float(dx), float(dy))
            score = ncc_against(prepared, render_icon(portrait, params), INNER_MASK)
            if score > best[1]:
                best = (params, score)
    return best


def triptych(real: np.ndarray, synth: np.ndarray, path: Path) -> None:
    diff = np.abs(real[..., :3].astype(np.int16) - synth.astype(np.int16))
    diff = np.clip(diff * 3, 0, 255).astype(np.uint8)
    cell = ICON_SIDE * UPSCALE
    sheet = Image.new("RGB", (cell * 3 + 16, cell), (24, 24, 24))
    for column, bgr in enumerate((real[..., :3], synth, diff)):
        image = Image.fromarray(bgr[..., ::-1], "RGB").resize((cell, cell), Image.NEAREST)
        sheet.paste(image, (column * (cell + 8), 0))
    sheet.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out", type=Path, default=Path("out/render-fit"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    version = latest_version()
    print(f"Data Dragon: {version}")
    portraits: dict[str, np.ndarray] = {}

    # Сбор вырезов с уточнением центра по наивному синтезу.
    crops: list[tuple[str, str, np.ndarray]] = []  # (чемпион, кадр, вырез)
    for frame in load_corpus(args.data / "ReplayCorpus"):
        for region in frame.labels.champions if frame.labels else ():
            champion = region.champion_id
            if champion not in portraits:
                portraits[champion] = portrait_bgra(champion, version)
            naive = render_icon(portraits[champion], NAIVE)
            cx = region.x + region.width // 2
            cy = region.y + region.height // 2
            x, y, _ = find_best_match(frame.pixels, naive, INNER_MASK, cx, cy, 3)
            try:
                crops.append((champion, frame.name, crop_centered(frame.pixels, x, y)))
            except ValueError:
                print(f"  пропущен у края кадра: {champion} в {frame.name}")

    print(f"Вырезов: {len(crops)}")

    coarse_grid = list(product(COARSE_CROP_FRAC, COARSE_NATIVE, COARSE_RESAMPLER,
                               COARSE_GAMMA, COARSE_BLUR))
    # score_by_config[конфиг][вырез] — лучший балл по грубым сдвигам.
    score_by_config = np.full((len(coarse_grid), len(crops)), -2.0)

    results = []
    for crop_index, (champion, frame_name, real) in enumerate(crops):
        prepared = prepared_vector(real, INNER_MASK)
        portrait = portraits[champion]
        for config_index, (frac, native, resampler, gamma, blur) in enumerate(coarse_grid):
            for dx, dy in product(COARSE_SHIFT, repeat=2):
                params = RenderParams(frac, native, resampler, gamma, blur,
                                      "bilinear", dx, dy)
                score = ncc_against(prepared, render_icon(portrait, params), INNER_MASK)
                if score > score_by_config[config_index, crop_index]:
                    score_by_config[config_index, crop_index] = score

        # Индивидуальный подбор: уточнить сдвиг у лучших грубых конфигов.
        top = np.argsort(score_by_config[:, crop_index])[-5:]
        best_params, best_score = None, -2.0
        for config_index in top:
            frac, native, resampler, gamma, blur = coarse_grid[config_index]
            base = RenderParams(frac, native, resampler, gamma, blur,
                                "bilinear", 0.0, 0.0)
            params, score = refine_shift(portrait, base, prepared)
            if score > best_score:
                best_params, best_score = params, score

        synth = render_icon(portrait, best_params)
        residual = 1.0 - best_score
        center_residual = 1.0 - ncc_against(
            prepared_vector(real, CENTER_MASK), synth, CENTER_MASK)
        results.append((residual, center_residual, champion, frame_name, best_params))
        triptych(real, synth,
                 args.out / f"{residual:.3f}-{champion}-{frame_name}.png")
        p = best_params
        print(f"  {champion} @ {frame_name}: диск {residual:.3f}, "
              f"центр {center_residual:.3f} | crop {p.crop_frac:.2f}, "
              f"native {p.native_size}, {p.resampler}/{p.final_resampler}, "
              f"{p.gamma}, blur {p.blur_sigma}, сдвиг ({p.dx:.2f}, {p.dy:.2f})")

    # Общий набор параметров: конфиг с лучшей медианой по всем вырезам.
    medians = np.median(1.0 - score_by_config, axis=1)
    global_index = int(np.argmin(medians))
    frac, native, resampler, gamma, blur = coarse_grid[global_index]
    print()
    print(f"Общий набор (медиана остатка {medians[global_index]:.3f}, "
          f"грубые сдвиги): crop {frac:.2f}, native {native}, "
          f"{resampler}, {gamma}, blur {blur}")

    residuals = sorted(r for r, *_ in results)
    inside = sum(1 for r in residuals if r <= 0.20)
    print()
    print(f"Индивидуальные остатки по диску: медиана "
          f"{statistics.median(residuals):.3f}, p95 "
          f"{np.percentile(residuals, 95):.3f}, максимум {max(residuals):.3f}")
    print(f"Не выше порога 0,20: {inside} из {len(residuals)} "
          f"({100 * inside / len(residuals):.0f} %)")
    print(f"Тройки: {args.out.resolve()}")


if __name__ == "__main__":
    main()
