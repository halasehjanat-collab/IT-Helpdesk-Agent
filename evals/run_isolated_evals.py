"""Run evals/run_evals.py against throw-away servers and temporary databases.

LIVE and optional: the app makes real model calls. Nothing here touches
services/data/*.sqlite3; only the temporary directory created by this run is
removed afterwards.

    python evals/run_isolated_evals.py [--only S2-05,S2-10] [-v]

Every case runs in its own run_evals.py call. The simulated ticket API rewinds
its ticket ids on a case reset, so the temporary local ticket-reference table is
emptied between cases; otherwise a recycled id (IT-2001) would collide with the
previous case's reference. Combined scorecards are written to
evals/isolated_scorecard.md and evals/isolated_scorecard.json.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
import urllib.request
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

APP_PORT = 8110
HELPDESK_PORT = 8112
CASES = ROOT / "evals" / "cases.sample.jsonl"
OUT_MD = ROOT / "evals" / "isolated_scorecard.md"


def isolated_env(tmp: Path) -> dict[str, str]:
    """Environment that points every writable store at `tmp`."""
    env = dict(os.environ)
    env.update(
        {
            "HELPDESK_TICKET_DB_PATH": str(tmp / "tickets.sqlite3"),
            "HELPDESK_METADATA_DB_PATH": str(tmp / "metadata.sqlite3"),
            "CHAT_MEMORY_DB_PATH": str(tmp / "chat_memory.sqlite3"),
            "HELPDESK_ALLOW_DESTRUCTIVE_RESET": "1",
            "HELPDESK_API_URL": f"http://127.0.0.1:{HELPDESK_PORT}",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    return env


def clear_local_references(tmp: Path) -> None:
    """Empty the temporary ticket-reference table (never a project database)."""
    database = (tmp / "metadata.sqlite3").resolve()
    if tmp.resolve() not in database.parents or not database.exists():
        return
    with closing(sqlite3.connect(database, timeout=10)) as connection, connection:
        connection.execute("DELETE FROM ticket_references")


def case_ids(only: str | None) -> list[str]:
    wanted = {x.strip() for x in only.split(",")} if only else None
    ids = []
    for line in CASES.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.lstrip().startswith("//"):
            case_id = json.loads(line).get("id")
            if case_id and (wanted is None or case_id in wanted):
                ids.append(case_id)
    return ids


def _wait(url: str, timeout: float = 60.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=2)
            return
        except Exception:
            time.sleep(0.5)
    raise RuntimeError(f"server did not become ready: {url}")


def write_combined(results: list[dict], started: float, out_md: Path = OUT_MD) -> int:
    import run_evals

    md, summary = run_evals.write_scorecard(
        results, SimpleNamespace(cases=CASES), started
    )
    out_md.write_text(md, encoding="utf-8")
    out_md.with_suffix(".json").write_text(
        json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    print(f"Passed {summary['passed']}/{summary['runs']}; wrote {out_md} and {out_md.with_suffix('.json')}")
    return 0 if summary["passed"] == summary["runs"] else 1


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only")
    parser.add_argument(
        "--pipeline-diagnostics",
        action="store_true",
        help="Enable safe stage logs (document IDs, verdicts, ticket metadata; no answer or requester text).",
    )
    options, passthrough = parser.parse_known_args(argv)

    tmp = Path(tempfile.mkdtemp(prefix="helpdesk-eval-"))
    env = isolated_env(tmp)
    if options.pipeline_diagnostics:
        env["HELPDESK_PIPELINE_DIAGNOSTICS"] = "1"
    procs: list[subprocess.Popen] = []
    results: list[dict] = []
    started = time.time()
    try:
        for module, port in (
            ("services.helpdesk_api:app", HELPDESK_PORT),
            ("app:app", APP_PORT),
        ):
            procs.append(
                subprocess.Popen(
                    [sys.executable, "-m", "uvicorn", module,
                     "--port", str(port), "--log-level",
                     "info" if options.pipeline_diagnostics else "warning"],
                    cwd=ROOT,
                    env=env,
                )
            )
        _wait(f"http://127.0.0.1:{HELPDESK_PORT}/health")
        _wait(f"http://127.0.0.1:{APP_PORT}/health")

        for case_id in case_ids(options.only):
            clear_local_references(tmp)
            case_out = tmp / f"{case_id}.md"
            subprocess.run(
                [
                    sys.executable, str(ROOT / "evals" / "run_evals.py"),
                    "--app", f"http://127.0.0.1:{APP_PORT}",
                    "--helpdesk", f"http://127.0.0.1:{HELPDESK_PORT}",
                    "--only", case_id,
                    "--out", str(case_out),
                    *passthrough,
                ],
                cwd=ROOT,
                env=env,
            )
            case_json = case_out.with_suffix(".json")
            if case_json.exists():
                results.extend(json.loads(case_json.read_text(encoding="utf-8"))["results"])
        return write_combined(results, started) if results else 1
    finally:
        for proc in procs:
            proc.terminate()
        for proc in procs:
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
