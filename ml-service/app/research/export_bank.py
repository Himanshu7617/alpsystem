"""Export the canonical item bank in the shape the live platform serves.

Phase 10's requirement is that the research pipeline and the running
application are the same system. They were not: the platform served hand-built
topic files whose "concepts" are subject names and whose difficulty is a word,
while every policy in Phases 8 and 9 acts on a concept id, a difficulty bin
1-10 and an IRT *b*. An action chosen by the served policy could not be
resolved to an item.

So the bank the platform serves is generated from the same two files the
experiments read — `data/items/item-bank-v1.json` and the Phase 1 calibration
`artifacts/datasets/item-parameters-v1.csv` — and it carries the curriculum
`closed_loop.select_curriculum` derives, so the live concept order is the
experiment's concept order rather than a second opinion about it.

    python -m app.research.export_bank        # writes the backend bank file
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT_PATH = ROOT / "backend" / "src" / "data" / "questions" / "item-bank-v1.json"

#: The platform's three difficulty words, over the ten bins. Only used for
#: display and for the legacy `Question.difficulty` column; every decision uses
#: the bin.
def label(bin_: int) -> str:
    if bin_ <= 3:
        return "easy"
    return "medium" if bin_ <= 7 else "hard"


def export(out_path: Path = OUT_PATH) -> dict:
    import numpy as np

    from app.research.closed_loop import select_curriculum
    from app.research.simulator import BANK_PATH, Params, load_items

    # The dataclass carries the calibrated *b* and the difficulty bin; the stem,
    # the options and the explanation stay in the bank file. Both are read here
    # so the exported item is the same item the experiments served.
    calibrated = {item.item_id: item
                  for item in load_items(np.random.default_rng(0), Params.load())}
    curriculum = select_curriculum(list(calibrated.values()))
    content = json.loads(BANK_PATH.read_text(encoding="utf-8"))["items"]

    questions = []
    for entry in content:
        item = calibrated[entry["item_id"]]
        if item.concept_id not in curriculum:
            continue
        options = list(entry["options"])
        questions.append({
            "id": item.item_id,
            "concept_id": item.concept_id,
            "difficulty": label(item.difficulty_score),
            "difficultyScore": int(item.difficulty_score),
            "b": round(float(item.b), 6),
            "questionType": entry.get("format", "mcq"),
            "question": entry["stem"],
            "options": options,
            "correctAnswer": options[int(entry["correct_index"])],
            "explanation": entry.get("explanation", ""),
            "estimatedTimeSeconds": 45,
            "concepts": [item.concept_id],
            "tags": [entry.get("subject", "")],
            "learningObjective": "",
            "prerequisiteLevel": 1,
            "sourceType": "canonical-item-bank-v1",
        })
    questions.sort(key=lambda question: (curriculum.index(question["concept_id"]),
                                         question["difficultyScore"], question["id"]))

    payload = {
        "bank_version": "item-bank-v1",
        "generated_by": "ml-service/app/research/export_bank.py",
        "curriculum": curriculum,
        "questions": questions,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    arguments = parser.parse_args()
    payload = export(arguments.out)
    print(f"wrote: {arguments.out.relative_to(ROOT)} — {len(payload['questions'])} items "
          f"over {len(payload['curriculum'])} concepts")


if __name__ == "__main__":
    main()
