"""
Conversation logic shared by every front end (terminal chat, evals, later the web GUI).

One Assistant = one conversation: system prompt, trimmed history, retrieval per turn.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from importlib import resources

from .config import Config
from .llm import Message, OllamaClient, OllamaError
from .rag import Hit, KnowledgeBase, format_context


def load_prompt(name: str) -> str:
    """Load a versioned system prompt from prompts/<name>.md"""
    path = resources.files(__package__).joinpath("prompts", f"{name}.md")
    return path.read_text(encoding="utf-8").strip()


@dataclass
class Turn:
    hits: list[Hit]
    stream: Iterator[str]  # must be consumed, it commits the answer to history when done
    retrieval_error: str | None = None


class Assistant:
    def __init__(
        self,
        cfg: Config,
        client: OllamaClient,
        kb: KnowledgeBase | None,
        prompt: str | None = None,
        context_style: str | None = None,
        options: dict | None = None,
    ):
        self.cfg = cfg
        self.client = client
        self.kb = kb
        self.system = load_prompt(prompt or cfg.chat.prompt)
        self.context_style = context_style or cfg.rag.context_style
        self.options = options or {}  # extra Ollama options, e.g. a seed for evals
        self.history: list[Message] = []

    def reset(self) -> None:
        self.history.clear()

    def ask(self, question: str, hits: list[Hit] | None = None) -> Turn:
        """Start a turn; pass `hits` to skip retrieval (evals use fixed notes)"""
        error = None
        if hits is None:
            hits = []
            if self.kb is not None:
                try:
                    hits = self.kb.retrieve(question)
                except OllamaError as e:
                    error = str(e)

        self.history.append({"role": "user", "content": question})
        # keep the last N exchanges; odd length so history always starts with a user turn
        del self.history[: -(self.cfg.chat.max_history_turns * 2 - 1)]

        return Turn(hits, self._stream(self._messages(hits)), error)

    def _messages(self, hits: list[Hit]) -> list[Message]:
        # retrieved notes go into this request only; history keeps the bare question
        # so old context doesn't pile up across turns
        system: Message = {"role": "system", "content": self.system}
        *earlier, last = self.history
        if not hits:
            return [system, *earlier, last]

        if self.context_style == "native":
            # Granite's template renders "document <id>" roles in its trained RAG format;
            # the id suffix also stops Ollama merging consecutive document messages
            docs: list[Message] = [
                {"role": f"document {i}", "content": f"{h.label}\n{h.text}"}
                for i, h in enumerate(hits, start=1)
            ]
            return [system, *docs, *earlier, last]

        inline = {"role": "user", "content": format_context(hits, last["content"])}
        return [system, *earlier, inline]

    def _stream(self, messages: list[Message]) -> Iterator[str]:
        parts: list[str] = []
        try:
            for chunk in self.client.stream_chat(messages, **self.options):
                parts.append(chunk)
                yield chunk
        finally:
            # keep partial answers (Ctrl-C), drop the question if nothing came back
            if parts:
                self.history.append({"role": "assistant", "content": "".join(parts)})
            else:
                self.history.pop()
