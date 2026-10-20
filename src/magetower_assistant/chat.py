"""
Terminal chat front end.

Commands:
    /reset  clear conversation history
    /quit   exit (Ctrl-D works too)
"""

from __future__ import annotations

import sys

from .assistant import Assistant
from .config import load_config
from .llm import OllamaClient, OllamaError
from .rag import BASELINE, KnowledgeBase


def main() -> int:
    cfg = load_config()
    client = OllamaClient(cfg.llm)

    kb: KnowledgeBase | None = None
    try:
        if cfg.rag.enabled:
            client.check(cfg.rag.embed_model)
            kb = KnowledgeBase(cfg.rag, client)
        else:
            client.check()
    except (OllamaError, RuntimeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    assistant = Assistant(cfg, client, kb)

    src = cfg.source or "built-in defaults"
    print(f"magetower-assistant | model {cfg.llm.model} | config: {src}")
    print(f"prompt: {cfg.chat.prompt} | context: {assistant.context_style}")
    if kb is None:
        print("knowledge base: disabled")
    else:
        corpora = ", ".join([BASELINE, *cfg.rag.active_corpora])
        print(f"knowledge base: {kb.count()} chunks | corpora: {corpora}")
        if kb.count() == 0:
            print("  (empty, run magetower-ingest)")
    print("Describe what you're seeing. /reset clears history, /quit exits.\n")

    while True:
        try:
            user = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        if not user:
            continue
        if user in ("/quit", "/exit"):
            return 0
        if user == "/reset":
            assistant.reset()
            print("(history cleared)\n")
            continue

        turn = assistant.ask(user)
        if turn.retrieval_error:
            print(f"(retrieval failed: {turn.retrieval_error})", file=sys.stderr)

        print("guide> ", end="", flush=True)
        try:
            for chunk in turn.stream:
                print(chunk, end="", flush=True)
        except KeyboardInterrupt:
            print("\n(interrupted)")
        except OllamaError as e:
            print(f"\nerror: {e}", file=sys.stderr)
            continue
        finally:
            turn.stream.close()  # commit to history now, before the next question
        print("\n")

        for i, h in enumerate(turn.hits, start=1):
            print(f"  [{i}] {h.label} ({h.distance:.2f})")
        if turn.hits:
            print()


if __name__ == "__main__":
    sys.exit(main())
