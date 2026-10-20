"""
Run the answer-quality eval cases in evals/cases.toml.

Each case is run --repeat times with fixed seeds, so runs are comparable between prompt
versions. A full transcript is written to evals/runs/ for reading the answers themselves.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
import tomllib
from datetime import datetime

from .assistant import Assistant
from .config import PROJECT_ROOT, load_config
from .llm import OllamaClient, OllamaError
from .rag import Hit

CASES = PROJECT_ROOT / "evals" / "cases.toml"
RUNS = PROJECT_ROOT / "evals" / "runs"

# the answer structure the prompt asks for, checked when a case has format = true
FORMAT_SECTIONS = ["Assessment", "Tools", "Steps", "References"]


def check(case: dict, answer: str) -> list[tuple[str, bool]]:
    results = []
    for pattern in case.get("must_match", []):
        results.append((f"match {pattern}", bool(re.search(pattern, answer, re.I))))
    for pattern in case.get("must_not_match", []):
        results.append((f"no {pattern}", not re.search(pattern, answer, re.I)))
    if case.get("format", True):
        for name in FORMAT_SECTIONS:
            heading = rf"^#{{1,4}}\s*(?:\*\*)?{name}"
            results.append((f"section {name}", bool(re.search(heading, answer, re.M))))
    return results


def main() -> int:
    cfg = load_config()
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--prompt", default=cfg.chat.prompt, help="prompts/<name>.md")
    parser.add_argument(
        "--context", choices=["inline", "native"], default=cfg.rag.context_style
    )
    parser.add_argument("--repeat", type=int, default=3, help="runs per case")
    parser.add_argument("--case", nargs="*", help="only run these case ids")
    args = parser.parse_args()

    cases = tomllib.loads(CASES.read_text(encoding="utf-8"))["case"]
    if args.case:
        cases = [c for c in cases if c["id"] in args.case]

    client = OllamaClient(cfg.llm)
    try:
        client.check()
    except OllamaError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    RUNS.mkdir(parents=True, exist_ok=True)
    out_path = RUNS / f"{stamp}_{args.prompt}_{args.context}.md"
    log = [
        f"# Eval run {stamp}",
        f"model `{cfg.llm.model}` | prompt `{args.prompt}` | context `{args.context}` | "
        f"temperature {cfg.llm.temperature} | repeat {args.repeat}\n",
    ]

    total_ok = total = 0
    print(f"prompt {args.prompt} | context {args.context} | repeat {args.repeat}\n")
    for case in cases:
        hits = [
            Hit(text=n["text"].strip(), source=n["label"], heading="", corpus="eval", distance=0.0)
            for n in case.get("notes", [])
        ]
        log.append(f"## {case['id']}\n\n> {case['question']}\n")
        clean = 0
        failed: set[str] = set()
        elapsed = 0.0
        for seed in range(1, args.repeat + 1):
            assistant = Assistant(
                cfg, client, kb=None, prompt=args.prompt,
                context_style=args.context, options={"seed": seed},
            )
            start = time.monotonic()
            try:
                answer = "".join(assistant.ask(case["question"], hits=hits).stream)
            except OllamaError as e:
                print(f"error: {e}", file=sys.stderr)
                return 1
            elapsed += time.monotonic() - start

            results = check(case, answer)
            ok = sum(passed for _, passed in results)
            total_ok += ok
            total += len(results)
            misses = [name for name, passed in results if not passed]
            clean += not misses
            failed.update(misses)

            log.append(f"### seed {seed}: {ok}/{len(results)} checks\n")
            if misses:
                log.append("Failed: " + ", ".join(f"`{m}`" for m in misses) + "\n")
            log.append(answer + "\n")

        print(f"  {case['id']:<26} {clean}/{args.repeat} clean  "
              f"{elapsed / args.repeat:5.1f}s/answer", flush=True)
        for name in sorted(failed):
            print(f"      fails: {name}")

    score = f"checks passed: {total_ok}/{total} ({100 * total_ok / max(total, 1):.0f}%)"
    print(f"\n{score}\ntranscript: {out_path}")
    log.insert(2, f"**{score}**\n")
    out_path.write_text("\n".join(log), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
