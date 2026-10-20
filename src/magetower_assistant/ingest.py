"""
Embed vetted knowledge-base notes into the vector store.

Every run rebuilds the store from scratch, so removed or un-vetted notes drop out.
"""

from __future__ import annotations

import sys
from collections import Counter

from .config import load_config
from .llm import OllamaClient, OllamaError
from .rag import KnowledgeBase, load_notes


def main() -> int:
    cfg = load_config()
    if not cfg.rag.kb_dir.is_dir():
        print(f"error: kb_dir not found: {cfg.rag.kb_dir}", file=sys.stderr)
        return 1

    client = OllamaClient(cfg.llm)
    try:
        client.check(cfg.rag.embed_model)
        chunks, skipped = load_notes(cfg.rag.kb_dir)
        kb = KnowledgeBase(cfg.rag, client, fresh=True)
        kb.ingest(chunks)
    except OllamaError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    for corpus, n in sorted(Counter(c.corpus for c in chunks).items()):
        print(f"  {corpus}: {n} chunks")
    for path in skipped:
        print(f"  skipped (not vetted): {path}")
    print(f"done, {kb.count()} chunks in store")
    return 0


if __name__ == "__main__":
    sys.exit(main())
