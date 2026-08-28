"""Write the reproducible local retrieval comparison used in the README."""

from __future__ import annotations

import json

from enterprise_qa.evaluation import GOLD_PATH, compare_configurations


def main() -> None:
    output = GOLD_PATH.with_name("local_results.json")
    payload = compare_configurations()
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
