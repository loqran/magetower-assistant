"""
Run the answer-quality eval cases in evals/cases.toml.

Each case is run --repeat times with fixed seeds, so runs are comparable between prompt
versions. A full transcript is written to evals/runs/ for reading the answers themselves.

Modes:
    default            cases use their fixed `notes`, tests the prompt alone
    --retrieve         notes come from the real knowledge base, tests KB + prompt
    --retrieval-only   no answers, prints the nearest chunks per case to calibrate
                       rag.max_distance and check expect_source
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
from .rag import Hit, KnowledgeBase

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


def check_retrieval(case: dict, hits: list[Hit]) -> list[tuple[str, bool]]:
    """Retrieval checks on the hits that passed the distance cutoff"""
    results = []
    if pattern := case.get("expect_source"):
        found = any(re.search(pattern, h.source, re.I) for h in hits)
        results.append((f"retrieved {pattern}", found))
    if case.get("expect_no_hits"):
        results.append(("retrieved nothing", not hits))
    return results


def retrieval_only(cases: list[dict], kb: KnowledgeBase, cutoff: float, top_k: int) -> int:
    ok = total = 0
    for case in cases:
        hits = kb.search(case["question"], n=top_k)
        kept = [h for h in hits if h.distance <= cutoff]
        results = check_retrieval(case, kept)
        ok += sum(passed for _, passed in results)
        total += len(results)
        status = "  ".join(("ok " if passed else "FAIL ") + name for name, passed in results)
        print(f"{case['id']}  {status}")
        for h in hits:
            mark = " " if h.distance <= cutoff else "x"
            print(f"  {mark} {h.distance:.3f}  {h.label}")
    print(f"\nretrieval checks passed: {ok}/{total} (x = beyond max_distance {cutoff})")
    return 0


def main() -> int:
    cfg = load_config()
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--prompt", default=cfg.chat.prompt, help="prompts/<name>.md")
    parser.add_argument(
        "--context", choices=["inline", "native"], default=cfg.rag.context_style
    )
    parser.add_argument("--repeat", type=int, default=3, help="runs per case")
    parser.add_argument("--case", nargs="*", help="only run these case ids")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--retrieve", action="store_true", help="notes from the real KB")
    mode.add_argument(
        "--retrieval-only", action="store_true", help="only show retrieved chunks"
    )
    parser.add_argument(
        "--top-k", type=int, default=8, help="chunks shown with --retrieval-only"
    )
    args = parser.parse_args()

    cases = tomllib.loads(CASES.read_text(encoding="utf-8"))["case"]
    if args.case:
        cases = [c for c in cases if c["id"] in args.case]

    client = OllamaClient(cfg.llm)
    kb = None
    try:
        client.check()
        if args.retrieve or args.retrieval_only:
            client.check(cfg.rag.embed_model)
            kb = KnowledgeBase(cfg.rag, client)
    except (OllamaError, RuntimeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    if kb is not None and kb.count() == 0:
        print("warning: knowledge base is empty, run magetower-ingest\n", file=sys.stderr)

    if args.retrieval_only:
        return retrieval_only(cases, kb, cfg.rag.max_distance, args.top_k)

    if not args.retrieve:
        # retrieval-only cases (off-topic negatives) mean nothing without the KB
        cases = [c for c in cases if not c.get("retrieval_only")]

    source = "kb" if args.retrieve else "fixed"
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    RUNS.mkdir(parents=True, exist_ok=True)
    out_path = RUNS / f"{stamp}_{args.prompt}_{args.context}_{source}.md"
    log = [
        f"# Eval run {stamp}",
        f"model `{cfg.llm.model}` | prompt `{args.prompt}` | context `{args.context}` | "
        f"notes `{source}` | temperature {cfg.llm.temperature} | repeat {args.repeat}\n",
    ]

    total_ok = total = 0
    print(f"prompt {args.prompt} | context {args.context} | notes {source} | "
          f"repeat {args.repeat}\n")
    for case in cases:
        log.append(f"## {case['id']}\n\n> {case['question']}\n")
        failed: set[str] = set()

        if kb is not None:
            try:
                hits = kb.retrieve(case["question"])
            except OllamaError as e:
                print(f"error: {e}", file=sys.stderr)
                return 1
            results = check_retrieval(case, hits)
            total_ok += sum(passed for _, passed in results)
            total += len(results)
            failed.update(name for name, passed in results if not passed)
            log.append("Retrieved:\n")
            log.extend(f"- {h.distance:.3f} {h.label}" for h in hits)
            log.append("" if hits else "- (nothing)\n")
        else:
            hits = [
                Hit(text=n["text"].strip(), source=n["label"], heading="", corpus="eval",
                    distance=0.0)
                for n in case.get("notes", [])
            ]

        if case.get("retrieval_only"):
            status = "ok" if not failed else "FAIL"
            print(f"  {case['id']:<26} retrieval {status}", flush=True)
            for name in sorted(failed):
                print(f"      fails: {name}")
            continue

        clean = 0
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
