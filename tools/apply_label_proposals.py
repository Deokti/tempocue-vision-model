"""Внесение проверенных предложений в файлы разметки корпуса.

Второй шаг после `tools/propose_labels.py`. Разделение намеренное: первый
инструмент только предлагает и рисует контрольные листы, второй вносит правку
в основной репозиторий — и лишь то, что человек уже посмотрел глазами.

    .venv/Scripts/python tools/apply_label_proposals.py            # показать
    .venv/Scripts/python tools/apply_label_proposals.py --apply    # записать

Без `--apply` инструмент ничего не пишет, а только печатает, что изменится.

Соглашение о координатах: в файле хранится левый верхний угол коробки 25x25,
а предложения приходят центрами. Записывается `round(центр - 25/2)`, то есть
центр из файла (`x + width/2`) совпадает с предложенным с точностью до
половины пикселя — точнее формат и не позволяет.

Перед записью инструмент проверяет себя: пересобирает файл **без правок** и
сверяет с оригиналом побайтово. Не совпало — файл не трогается вовсе. Это
гарантирует, что в diff попадёт правка, а не переформатирование: корпус
неоднороден, в нём три разных соглашения записи на двенадцать файлов.

Поле `allowedMissedChampions` инструмент **не трогает**: это допуск приёмки
приложения, и его пересмотр — отдельное решение. После пополнения разметки он
стал завышенным, о чём инструмент напоминает в конце.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

DEFAULT_CORPUS = Path(
    r"C:\Users\deokn\.codex\worktrees\739a\PROJECT\tests\TempoCue.Vision.Tests\ReplayCorpus"
)
DEFAULT_PROPOSALS = Path("out/label-proposals/proposals.json")
# Сторона коробки чемпиона в разметке корпуса — как во всех существующих метках.
BOX_SIDE = 25
CHAMPION_KIND = "Champion"
# Поля, которые в записи корпуса стоят в первой строке области; остальные
# переносятся на вторую. Формат воспроизводится ради читаемого diff: обычный
# json.dumps раздувает каждую область до девяти строк, и настоящая правка в
# шестистах изменённых строках теряется.
HEAD_FIELDS = ("x", "y", "width", "height", "kind")
# Отступ развёрнутого стиля — как в единственном таком файле корпуса.
INDENT = 2


def format_region(region: dict, last: bool) -> list[str]:
    """Одна область в том же виде, в каком её пишет разметчик корпуса."""
    head = ", ".join(
        f'"{name}": {json.dumps(region[name], ensure_ascii=False)}'
        for name in HEAD_FIELDS
        if name in region
    )
    tail = ", ".join(
        f'"{name}": {json.dumps(value, ensure_ascii=False)}'
        for name, value in region.items()
        if name not in HEAD_FIELDS
    )
    comma = "" if last else ","
    if not tail:
        return [f"    {{ {head} }}{comma}"]
    return [f"    {{ {head},", f"      {tail} }}{comma}"]


@dataclass(frozen=True)
class FileStyle:
    """Как записан конкретный файл разметки: корпус не единообразен.

    Из двенадцати файлов десять записаны компактно с CRLF, один компактно с
    LF, один — развёрнуто, по полю на строку. Инструмент сохраняет стиль
    каждого файла: иначе переформатирование раздувает diff до сотен строк, и
    настоящая правка в нём теряется.
    """

    newline: str
    expanded: bool


def detect_style(text: str) -> FileStyle:
    """Определяет соглашения файла по его собственному тексту."""
    newline = "\r\n" if "\r\n" in text else "\n"
    expanded = any(line.strip().startswith('"x":') for line in text.splitlines())
    return FileStyle(newline=newline, expanded=expanded)


def format_document(document: dict, style: FileStyle) -> str:
    """Весь файл разметки как текст в стиле этого же файла."""
    if style.expanded:
        body = json.dumps(document, ensure_ascii=False, indent=INDENT)
        return style.newline.join(body.splitlines()) + style.newline
    lines = ["{"]
    keys = list(document)
    for index, key in enumerate(keys):
        last_key = index == len(keys) - 1
        if key == "regions":
            lines.append('  "regions": [')
            regions = document["regions"]
            for position, region in enumerate(regions):
                lines.extend(format_region(region, position == len(regions) - 1))
            lines.append("  ]" + ("" if last_key else ","))
        else:
            rendered = json.dumps(document[key], ensure_ascii=False)
            lines.append(f'  "{key}": {rendered}' + ("" if last_key else ","))
    lines.append("}")
    return style.newline.join(lines) + style.newline


def box_corner(center: float) -> int:
    """Центр значка → левый верхний угол коробки в записи разметки."""
    return round(center - BOX_SIDE / 2)


def labels_path(corpus: Path, frame: str) -> Path:
    return corpus / f"{frame}.tempocue-vision.labels.json"


def apply_frame(document: dict, frame: dict) -> list[str]:
    """Меняет разметку одного кадра на месте; возвращает список правок словами."""
    changes: list[str] = []
    regions = document["regions"]
    by_champion = {
        region.get("championId"): region
        for region in regions
        if region.get("kind") == CHAMPION_KIND
    }

    for proposal in frame["moved"]:
        region = by_champion.get(proposal["championId"])
        if region is None:
            changes.append(f"  ! {proposal['championId']}: метки нет, сдвиг пропущен")
            continue
        was = (region["x"], region["y"])
        region["x"] = box_corner(proposal["centerX"])
        region["y"] = box_corner(proposal["centerY"])
        changes.append(
            f"  ~ {proposal['championId']:14} ({was[0]},{was[1]}) → "
            f"({region['x']},{region['y']})  сдвиг {proposal['shift']} px, "
            f"совпадение {proposal['score']}"
        )

    last_champion = max(
        (index for index, region in enumerate(regions) if region.get("kind") == CHAMPION_KIND),
        default=-1,
    )
    for offset, proposal in enumerate(frame["added"]):
        if proposal["championId"] in by_champion:
            changes.append(f"  ! {proposal['championId']}: уже размечен, пропущен")
            continue
        region = {
            "x": box_corner(proposal["centerX"]),
            "y": box_corner(proposal["centerY"]),
            "width": BOX_SIDE,
            "height": BOX_SIDE,
            "kind": CHAMPION_KIND,
            "championId": proposal["championId"],
            "affiliation": proposal["affiliation"],
        }
        regions.insert(last_champion + 1 + offset, region)
        changes.append(
            f"  + {proposal['championId']:14} ({region['x']},{region['y']})  "
            f"совпадение {proposal['score']}"
        )
    return changes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--proposals", type=Path, default=DEFAULT_PROPOSALS)
    parser.add_argument("--apply", action="store_true", help="записать правки в файлы")
    args = parser.parse_args()

    payload = json.loads(args.proposals.read_text(encoding="utf-8"))
    print(f"Предложения: {args.proposals} (порог добавления {payload['addScore']})")
    print(f"Корпус: {args.corpus}")
    print("Режим: " + ("ЗАПИСЬ" if args.apply else "показ, файлы не меняются"))
    print()

    added = moved = touched = skipped = 0
    for frame in payload["frames"]:
        if not frame["added"] and not frame["moved"]:
            continue
        path = labels_path(args.corpus, frame["frame"])
        if not path.exists():
            print(f"{frame['frame']}: файла разметки нет — {path.name}")
            continue
        text = path.read_bytes().decode("utf-8")
        document = json.loads(text)
        style = detect_style(text)
        if format_document(document, style) != text:
            print(f"{frame['frame']}: стиль файла не воспроизводится, пропущен")
            skipped += 1
            continue
        changes = apply_frame(document, frame)
        if not changes:
            continue

        print(frame["frame"])
        for line in changes:
            print(line)
        added += len(frame["added"])
        moved += len(frame["moved"])
        touched += 1
        if args.apply:
            path.write_bytes(format_document(document, style).encode("utf-8"))

    print()
    print(f"Кадров затронуто: {touched}; добавлено чемпионов: {added}; сдвинуто меток: {moved}")
    if skipped:
        print(f"Пропущено файлов из-за незнакомого стиля: {skipped}")
    if args.apply:
        print("Файлы записаны. Проверить: git diff в репозитории приложения.")
        print(
            "Напоминание: allowedMissedChampions не тронут. После пополнения разметки "
            "этот допуск стал завышенным — пересмотр отдельным решением."
        )
    else:
        print("Ничего не записано. Повторить с --apply, когда предложения проверены.")


if __name__ == "__main__":
    main()
