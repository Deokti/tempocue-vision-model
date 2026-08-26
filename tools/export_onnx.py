"""Перевод обученных весов в ONNX для распознавателя приложения.

    .venv/Scripts/python tools/export_onnx.py \\
        --detector out/detector-04/best.pt --identity out/identity-04/best.pt \\
        --out out/onnx

Отдаёт три файла: `detector.onnx`, `identity.onnx` и `model.json` с
словарём классов и всеми числами, которые нужны вызывающей стороне. Модельный
репозиторий отдаёт только это; решение «годится или нет» принимается в
основном репозитории существующими инструментами.

Главное решение: **подготовка пикселей уходит внутрь графа**. На вход подаётся
сырой BGRA точно в том виде, в каком его хранит запись приложения, а выбор
каналов, деление на 255 и растяжение выреза делает сама сеть. Причина не в
удобстве: перепутанный порядок каналов или забытое деление — ошибка, которая
не роняет программу, а тихо портит ответы. За программу таких было девять, и
каждая обнаруживалась много позже. Здесь этот класс ошибок закрыт устройством:
ошибиться попросту негде.

Экспорт сверяется с PyTorch на настоящих кадрах корпуса. Расхождение больше
`TOLERANCE` — отказ записывать файлы: молча разошедшийся экспорт хуже, чем
отсутствующий.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime
import torch
from torch import nn

from tcvm.detector import (
    CANONICAL_SIDE,
    MATCH_DISTANCE,
    PEAK_THRESHOLD,
    STRIDE,
    CenterDetector,
)
from tcvm.formats import default_corpus_dir, load_corpus
from tcvm.identity import (
    BYTE_SCALE,
    CANONICAL_CROP,
    EMBEDDING_SIZE,
    INPUT_SIDE,
    NO_CHAMPION,
    REJECT_INDEX,
    IdentityNet,
    check_setup,
)

# Порядок каналов записи приложения: B, G, R, A. Сети учились на RGB.
RGB_FROM_BGRA = (2, 1, 0)
# Допуск сверки с PyTorch. Порядок сложения в ONNX Runtime свой, поэтому
# побитового совпадения не бывает; 1e-4 на логитах — это заведомо меньше
# любого различия, способного изменить ответ.
TOLERANCE = 1e-4
# Кадров корпуса на сверку: больше не нужно, расхождение либо есть везде,
# либо его нет вовсе.
CHECK_FRAMES = 6
# Версия набора операций ONNX. 17 понимает всё, что нам нужно, и поддержан
# той версией ONNX Runtime, что уже стоит в приложении (1.20).
OPSET = 17
# Экспорт идёт старым путём (dynamo=False) намеренно. Новый выводит граф
# через torch.export и вставляет операции посвежее; проверять их пришлось бы
# рантаймом 1.20 приложения, а выигрыша для двух маленьких свёрточных сетей
# нет никакого. Предупреждение о том, что старый путь устарел, ожидаемо.
USE_DYNAMO_EXPORT = False


class DetectorGraph(nn.Module):
    """Детектор вместе с подготовкой кадра: BGRA uint8 → карта центров."""

    def __init__(self, detector: CenterDetector) -> None:
        super().__init__()
        self.detector = detector

    def forward(self, frame_bgra: torch.Tensor) -> torch.Tensor:
        """(1, 320, 320, 4) uint8 → (1, 1, 80, 80) вероятности центра."""
        rgb = frame_bgra[..., RGB_FROM_BGRA].permute(0, 3, 1, 2).float() / BYTE_SCALE
        return torch.sigmoid(self.detector(rgb))


class IdentityGraph(nn.Module):
    """Опознание вместе с подготовкой выреза: BGRA uint8 → доли по классам."""

    def __init__(self, identity: IdentityNet) -> None:
        super().__init__()
        self.identity = identity

    def forward(self, crops_bgra: torch.Tensor) -> torch.Tensor:
        """(N, 32, 32, 4) uint8 → (N, классов) доли, сумма единица."""
        rgb = crops_bgra[..., RGB_FROM_BGRA].permute(0, 3, 1, 2).float() / BYTE_SCALE
        resized = nn.functional.interpolate(
            rgb, size=(INPUT_SIDE, INPUT_SIDE), mode="bilinear", align_corners=False
        )
        return torch.softmax(self.identity(resized), dim=1)


def corpus_samples(corpus: Path, count: int) -> tuple[np.ndarray, np.ndarray]:
    """Настоящие кадры и вырезы из них — материал для сверки экспорта.

    Сверять на случайном шуме нельзя: сеть на шуме отвечает вяло, и различие,
    заметное на настоящих значках, там утонет.
    """
    frames = [frame for frame in load_corpus(corpus) if frame.labels][:count]
    if not frames:
        raise FileNotFoundError(f"В корпусе {corpus} нет размеченных кадров")

    half = CANONICAL_CROP // 2
    crops = []
    for frame in frames:
        for region in frame.labels.champions:
            left = round(region.x + region.width / 2) - half
            top = round(region.y + region.height / 2) - half
            if (
                left < 0
                or top < 0
                or left + CANONICAL_CROP > frame.pixels.shape[1]
                or top + CANONICAL_CROP > frame.pixels.shape[0]
            ):
                continue
            crops.append(frame.pixels[top : top + CANONICAL_CROP, left : left + CANONICAL_CROP])
    return np.stack([frame.pixels for frame in frames]), np.stack(crops)


def compare(graph: nn.Module, path: Path, sample: np.ndarray, title: str) -> float:
    """Наибольшее расхождение ONNX с PyTorch на этом материале."""
    session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    name = session.get_inputs()[0].name
    with torch.no_grad():
        expected = graph(torch.from_numpy(sample)).numpy()
    actual = session.run(None, {name: sample})[0]
    difference = float(np.abs(expected - actual).max())
    print(f"  {title}: наибольшее расхождение {difference:.2e} на {len(sample)} шт")
    return difference


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detector", type=Path, required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--vocabulary", type=Path, default=None)
    parser.add_argument("--corpus", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("out/onnx"))
    args = parser.parse_args()
    corpus = args.corpus or default_corpus_dir()
    args.out.mkdir(parents=True, exist_ok=True)

    vocabulary_path = args.vocabulary or args.identity.parent / "vocabulary.json"
    vocabulary = json.loads(vocabulary_path.read_text(encoding="utf-8"))
    check_setup(args.identity)

    detector = CenterDetector()
    detector.load_state_dict(torch.load(args.detector, map_location="cpu"))
    detector.eval()
    identity = IdentityNet(len(vocabulary))
    identity.load_state_dict(torch.load(args.identity, map_location="cpu"))
    identity.eval()

    detector_graph = DetectorGraph(detector).eval()
    identity_graph = IdentityGraph(identity).eval()

    print(f"Детектор: {args.detector}")
    print(f"Опознание: {args.identity} ({len(vocabulary)} классов)")
    print(f"Корпус для сверки: {corpus}")
    print()

    frames, crops = corpus_samples(corpus, CHECK_FRAMES)
    print(f"Материал сверки: кадров {len(frames)}, вырезов {len(crops)}")

    detector_path = args.out / "detector.onnx"
    torch.onnx.export(
        detector_graph,
        (torch.from_numpy(frames[:1]),),
        str(detector_path),
        input_names=["frame_bgra"],
        output_names=["centers"],
        opset_version=OPSET,
        dynamo=USE_DYNAMO_EXPORT,
    )
    identity_path = args.out / "identity.onnx"
    torch.onnx.export(
        identity_graph,
        (torch.from_numpy(crops[:1]),),
        str(identity_path),
        input_names=["crops_bgra"],
        output_names=["scores"],
        dynamic_axes={"crops_bgra": {0: "count"}, "scores": {0: "count"}},
        opset_version=OPSET,
        dynamo=USE_DYNAMO_EXPORT,
    )

    print()
    print("Сверка с PyTorch на настоящих кадрах:")
    worst = max(
        compare(detector_graph, detector_path, frames[:1], "детектор"),
        compare(identity_graph, identity_path, crops, "опознание"),
    )
    if worst > TOLERANCE:
        detector_path.unlink(missing_ok=True)
        identity_path.unlink(missing_ok=True)
        raise SystemExit(
            f"Расхождение {worst:.2e} больше допуска {TOLERANCE:.0e}. "
            "Файлы удалены: молча разошедшийся экспорт хуже отсутствующего."
        )

    (args.out / "model.json").write_text(
        json.dumps(
            {
                "detector": detector_path.name,
                "identity": identity_path.name,
                "frameSide": CANONICAL_SIDE,
                "stride": STRIDE,
                "peakThreshold": PEAK_THRESHOLD,
                "matchDistance": MATCH_DISTANCE,
                "cropSide": CANONICAL_CROP,
                "embeddingSize": EMBEDDING_SIZE,
                "rejectIndex": REJECT_INDEX,
                "rejectName": NO_CHAMPION,
                "channelOrder": "bgra",
                "vocabulary": vocabulary,
                "source": {
                    "detector": str(args.detector),
                    "identity": str(args.identity),
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    for path in sorted(args.out.iterdir()):
        print(f"  {path.name:16} {path.stat().st_size / 1024:8.0f} КБ")
    print()
    print(f"Готово: {args.out.resolve()}")


if __name__ == "__main__":
    main()
