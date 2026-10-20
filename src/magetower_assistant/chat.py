"""
Minimal terminal chat loop against the local model (Stage 1).

Commands:
    /reset  clear conversation history
    /quit   exit (Ctrl-D works too)
"""

from __future__ import annotations

import sys

from .config import load_config
from .llm import Message, OllamaClient, OllamaError
from .rag import BASELINE, Hit, KnowledgeBase, format_context

# first draft of the role prompt, refined in Stage 4
SYSTEM_PROMPT = """\
You are a methodology and tool-selection guide for CTF challenges and \
sanctioned security competitions. The user describes what they are seeing; you help them \
work out the approach. The user runs every command themselves.

For each question:
1. Identify the likely challenge category (web, pwn, crypto, forensics, reversing, \
OSINT, misc, etc.) and say how confident you are.
2. Suggest the tools that fit and explain *why* each one fits.
3. Outline the methodology as ordered steps, with what to look for at each step.
4. Point to references (MITRE ATT&CK, CWE, OWASP, tool documentation) where relevant.

Rules:
- You are a guide, not an exploit generator. Explain techniques and tool usage, but do \
not write finished, weaponized exploits aimed at specific real-world targets.
- Assume an authorized context (the user's own lab or a sanctioned competition). If a \
request clearly targets systems outside that scope, decline and say why.
- If you are unsure, say so and suggest how the user could find out. Do not invent tool \
flags or CVE numbers.
- When reference notes are provided, prefer them over general knowledge and cite them \
by number, e.g. [1]. If they don't cover the question, say so and answer from general \
knowledge.
- Be concise and structured.
"""


def _trim(history: list[Message], max_turns: int) -> list[Message]:
    """Keep the system prompt plus the last `max_turns` user/assistant exchanges."""
    system, rest = history[:1], history[1:]
    return system + rest[-max_turns * 2 :]


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

    src = cfg.source or "built-in defaults"
    print(f"magetower-assistant | model {cfg.llm.model} | config: {src}")
    if kb is None:
        print("knowledge base: disabled")
    else:
        corpora = ", ".join([BASELINE, *cfg.rag.active_corpora])
        print(f"knowledge base: {kb.count()} chunks | corpora: {corpora}")
        if kb.count() == 0:
            print("  (empty, run magetower-ingest)")
    print("Describe what you're seeing. /reset clears history, /quit exits.\n")

    history: list[Message] = [{"role": "system", "content": SYSTEM_PROMPT}]

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
            history = history[:1]
            print("(history cleared)\n")
            continue

        history.append({"role": "user", "content": user})
        history = _trim(history, cfg.chat.max_history_turns)

        # retrieved notes go into this request only; history keeps the bare question
        # so old context doesn't pile up across turns
        hits: list[Hit] = []
        messages = history
        if kb is not None:
            try:
                hits = kb.retrieve(user)
            except OllamaError as e:
                print(f"(retrieval failed: {e})", file=sys.stderr)
            messages = history[:-1] + [
                {"role": "user", "content": format_context(hits, user)}
            ]

        print("guide> ", end="", flush=True)
        parts: list[str] = []
        try:
            for chunk in client.stream_chat(messages):
                print(chunk, end="", flush=True)
                parts.append(chunk)
        except KeyboardInterrupt:
            print("\n(interrupted)")
        except OllamaError as e:
            print(f"\nerror: {e}", file=sys.stderr)
            history.pop()  # drop the unanswered question
            continue
        print("\n")
        for i, h in enumerate(hits, start=1):
            print(f"  [{i}] {h.label} ({h.distance:.2f})")
        if hits:
            print()

        if parts:
            history.append({"role": "assistant", "content": "".join(parts)})
        else:
            history.pop()


if __name__ == "__main__":
    sys.exit(main())
