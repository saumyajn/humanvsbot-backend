import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from main import build_mock_reply, SETTINGS  # noqa: E402


def word_count(text: str) -> int:
    return len(re.findall(r"\b\S+\b", text))


def main():
    cases = json.loads((Path(__file__).parent / "eval_cases.json").read_text(encoding="utf-8"))
    failures = []

    for case in cases:
        reply = build_mock_reply(case["input"])
        lowered = reply.lower()
        blocked = [term for term in case.get("must_not_contain", []) if term.lower() in lowered]
        too_long = word_count(reply) > case.get("max_words", 999)

        if blocked or too_long:
            failures.append(
                {
                    "id": case["id"],
                    "reply": reply,
                    "blocked_terms": blocked,
                    "too_long": too_long,
                }
            )

    print(
        json.dumps(
            {
                "prompt_version": SETTINGS["prompt_version"],
                "cases": len(cases),
                "failures": failures,
            },
            indent=2,
        )
    )

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
