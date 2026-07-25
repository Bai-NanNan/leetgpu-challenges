from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

from .runner import run_job


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--job", required=True, type=Path)
    parser.add_argument("--result", required=True, type=Path)
    args = parser.parse_args()
    job = json.loads(args.job.read_text())
    try:
        result = run_job(job)
    except Exception as exc:
        result = {
            "status": "error",
            "failure": {"stage": "worker", "message": str(exc)},
        }
        if job.get("verbose"):
            result["failure"]["traceback"] = traceback.format_exc()
    args.result.write_text(json.dumps(result))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
