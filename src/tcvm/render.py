"""Параметрическая отрисовка значка миникарты из портрета Data Dragon.

Моделируется цепочка, которой игра и захват превращают портрет в вырез:

    портрет 120x120
      -> центральное кадрирование (crop_frac: какая доля портрета в значке)
      -> уменьшение до родного размера значка native_size (resampler, gamma)
      -> лёгкое размытие (blur_sigma)
      -> одно аффинное преобразование к каноническим 25 px с субпиксельным
         сдвигом (dx, dy) — это и второе уменьшение, и несовпадение сетки.

Кольцо команды и фон не отрисовываются: сравнение идёт по внутреннему диску,
куда они не попадают. Гамма: при gamma="linear" все операции выполняются в
линейном свете (sRGB раскодируется и кодируется обратно) — игра могла
смешивать пиксели и так, и так, поэтому это перебираемый параметр.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from .matching import ICON_SIDE

RESAMPLERS = {
    "nearest": Image.NEAREST,
    "bilinear": Image.BILINEAR,
    "bicubic": Image.BICUBIC,
    "lanczos": Image.LANCZOS,
    "box": Image.BOX,
}

_SRGB_TO_LINEAR = None


def _srgb_to_linear_lut() -> np.ndarray:
    global _SRGB_TO_LINEAR
    if _SRGB_TO_LINEAR is None:
        x = np.arange(256, dtype=np.float64) / 255.0
        _SRGB_TO_LINEAR = np.where(
            x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
    return _SRGB_TO_LINEAR


def _linear_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055)


def _gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    """Разделяемая гауссова свёртка по двум осям; Pillow не умеет float-каналы."""
    radius = max(1, int(3.0 * sigma + 0.5))
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-(offsets**2) / (2.0 * sigma**2))
    kernel /= kernel.sum()
    padded = np.pad(image, ((radius, radius), (radius, radius), (0, 0)), mode="edge")
    blurred_rows = np.apply_along_axis(
        lambda row: np.convolve(row, kernel, mode="valid"), 1, padded)
    return np.apply_along_axis(
        lambda column: np.convolve(column, kernel, mode="valid"), 0, blurred_rows)


@dataclass(frozen=True)
class RenderParams:
    """Один вариант отрисовки; частота перебора задаётся инструментом."""

    crop_frac: float      # доля стороны портрета, попадающая в значок
    native_size: int      # родной размер значка на миникарте игрока, px
    resampler: str        # чем портрет уменьшался до родного размера
    gamma: str            # "srgb" — как есть, "linear" — в линейном свете
    blur_sigma: float     # гауссово размытие в родных пикселях
    final_resampler: str  # чем родной значок приводился к каноническим 25
    dx: float             # субпиксельный сдвиг в канонических пикселях
    dy: float


def render_icon(portrait_bgra: np.ndarray, params: RenderParams) -> np.ndarray:
    """Строит канонический вырез 25x25 BGR (uint8) по параметрам отрисовки."""
    side = portrait_bgra.shape[0]
    crop_side = max(2, round(side * params.crop_frac))
    start = (side - crop_side) // 2
    bgr = portrait_bgra[start:start + crop_side, start:start + crop_side, :3]

    channels = bgr.astype(np.float32) / 255.0
    if params.gamma == "linear":
        channels = _srgb_to_linear_lut()[bgr].astype(np.float32)

    planes = [Image.fromarray(channels[..., i], mode="F") for i in range(3)]
    resampler = RESAMPLERS[params.resampler]
    planes = [p.resize((params.native_size, params.native_size), resampler)
              for p in planes]
    if params.blur_sigma > 0:
        native = np.stack([np.asarray(p) for p in planes], axis=-1)
        native = _gaussian_blur(native, params.blur_sigma)
        planes = [Image.fromarray(native[..., i].astype(np.float32), mode="F")
                  for i in range(3)]

    # Аффинное отображение выхода в родные координаты: масштаб плюс сдвиг.
    scale = params.native_size / ICON_SIDE
    coefficients = (scale, 0.0, params.dx * scale,
                    0.0, scale, params.dy * scale)
    final = RESAMPLERS[params.final_resampler]
    planes = [p.transform((ICON_SIDE, ICON_SIDE), Image.AFFINE,
                          coefficients, resample=final) for p in planes]

    result = np.stack([np.asarray(p) for p in planes], axis=-1)
    if params.gamma == "linear":
        result = _linear_to_srgb(result)
    return (np.clip(result, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
