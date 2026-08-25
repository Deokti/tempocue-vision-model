"""Перевод спорных кандидатов в метки после проверки человеком.

`tools/propose_labels.py` делит найденное на уверенное и спорное. Спорное —
это полоса совпадения 0,6-0,8, где вперемешку настоящие значки и пустая земля;
разделить их порогом нельзя, замер это показал прямо. Разделяет человек, глядя
на пронумерованный лист.

    .venv/Scripts/python tools/accept_candidates.py --numbers 1,2,3,15,20

Номера берутся из `check.json`, записанного вместе с листом. Принятые
кандидаты переносятся в предложения на добавление, дальше их вносит
`tools/apply_label_proposals.py`.

Перед переносом инструмент проверяет имя: в точке кандидата сравниваются все
чемпионы состава, и если лучшим оказывается не тот, чьё имя стоит в листе,
кандидат отклоняется. Человек подтверждает, что значок есть; кто именно на нём
изображён — вопрос к замеру, а не к глазам.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tcvm.cdragon import base_circle_bgra, patch_of
from tcvm.formats import load_frame
from tcvm.matching import (
    ICON_SIDE,
    INNER_RADIUS,
    MIN_VISIBLE_SHARE,
    circular_mask,
    crop_centered,
    masked_ncc,
    visible_mask,
)
from tcvm.render import RenderParams, render_icon

PATCH_VERSION = "16.16.1"
NAIVE = RenderParams(1.0, ICON_SIDE, "box", "srgb", 0.0, "bilinear", 0.0, 0.0)
FULL_MASK = circular_mask(ICON_SIDE, INNER_RADIUS)
CROWD_DISTANCE = 30.0


def icon_of(champion: str, cache: dict[str, np.ndarray | None]) -> np.ndarray | None:
    if champion not in cache:
        try:
            cache[champion] = render_icon(
                base_circle_bgra(champion, patch_of(PATCH_VERSION)), NAIVE
            )
        except (OSError, ValueError):
            cache[champion] = None
    return cache[champion]


def best_name(frame, center: tuple[int, int], busy: list[tuple[float, float]], cache: dict):
    """Кто из состава лучше всего совпадает в этой точке."""
    neighbours = [
        point
        for point in busy
        if np.hypot(point[0] - center[0], point[1] - center[1]) < CROWD_DISTANCE
    ]
    mask = visible_mask(center, neighbours) if neighbours else FULL_MASK
    if mask.sum() / float(FULL_MASK.sum()) < MIN_VISIBLE_SHARE:
        mask = FULL_MASK
    scored = []
    for reference in frame.references:
        template = icon_of(reference.champion_id, cache)
        if template is None:
            continue
        try:
            scored.append(
                (masked_ncc(crop_centered(frame.pixels, *center), template, mask), reference)
            )
        except ValueError:
            continue
    scored.sort(key=lambda item: -item[0])
    return scored[0] if scored else (0.0, None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--numbers", required=True, help="через запятую, из check.json")
    parser.add_argument("--out", type=Path, default=Path("out/blind-labels"))
    parser.add_argument("--frames", type=Path, required=True, help="папка с кадрами")
    args = parser.parse_args()

    wanted = {int(part) for part in args.numbers.split(",") if part.strip()}
    check = json.loads((args.out / "check.json").read_text(encoding="utf-8"))
    payload = json.loads((args.out / "proposals.json").read_text(encoding="utf-8"))
    by_frame = {entry["frame"]: entry for entry in payload["frames"]}

    cache: dict[str, np.ndarray | None] = {}
    frames: dict[str, object] = {}
    accepted = refused = 0
    for item in check:
        if item["n"] not in wanted:
            continue
        name = item["frame"]
        if name not in frames:
            frames[name] = load_frame(args.frames / f"{name}.tempocue-vision")
        frame = frames[name]
        entry = by_frame[name]
        busy = [(p["centerX"], p["centerY"]) for p in entry["added"]]
        score, reference = best_name(
            frame, (round(item["centerX"]), round(item["centerY"])), busy, cache
        )
        if reference is None or reference.champion_id != item["championId"]:
            found = reference.champion_id if reference else "никто"
            print(
                f"  {item['n']:3} отклонён: в листе {item['championId']}, "
                f"лучше подходит {found}"
            )
            refused += 1
            continue
        entry["added"].append(
            {
                "championId": item["championId"],
                "affiliation": "Ally" if reference.affiliation == 0 else "Enemy",
                "centerX": item["centerX"],
                "centerY": item["centerY"],
                "boxX": round(item["centerX"] - ICON_SIDE / 2),
                "boxY": round(item["centerY"] - ICON_SIDE / 2),
                "score": round(score, 3),
                "shift": 0.0,
                "byEye": True,
            }
        )
        entry["rejected"] = [
            p
            for p in entry["rejected"]
            if p["championId"] != item["championId"] or abs(p["centerX"] - item["centerX"]) > 1
        ]
        accepted += 1

    (args.out / "proposals.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    total = sum(len(entry["added"]) for entry in payload["frames"])
    print()
    print(f"Принято глазами: {accepted}; отклонено по имени: {refused}")
    print(f"Всего меток к внесению: {total}")


if __name__ == "__main__":
    main()
