"""
Knowledge base: markdown notes (an Obsidian vault) -> chunks -> Ollama embeddings -> Chroma.

Layout: <kb_dir>/<corpus>/**/*.md
    - the top-level folder name is the corpus ("baseline", "maritime", ...)
    - baseline is always searched, specialist corpora only when listed in config
    - folders starting with "_" or "." are not corpora (_templates, .obsidian, .trash)

Only notes with `vetted: true` in their YAML frontmatter are ingested.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import chromadb
import yaml

from .config import RAGConfig
from .llm import OllamaClient

BASELINE = "baseline"
COLLECTION = "kb_notes"

# nomic-embed-text was trained with these task prefixes, retrieval degrades without them
DOC_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "

MAX_CHUNK_CHARS = 1500  # sections longer than this split on paragraphs
MIN_CHUNK_CHARS = 200  # shorter sections merge into a neighbour, tiny chunks are retrieval noise
EMBED_BATCH = 32

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_FENCE = re.compile(r"^\s*(```|~~~)")
_EMBED = re.compile(r"!\[\[[^\]]*\]\]")
_WIKILINK = re.compile(r"\[\[([^\]|#]*)(?:#([^\]|]*))?(?:\|([^\]]*))?\]\]")
_HIGHLIGHT = re.compile(r"==(.+?)==")
_CALLOUT = re.compile(r"^>\s*\[!([\w-]+)\][+-]?\s*(.*)$")


@dataclass
class Chunk:
    id: str
    text: str
    source: str  # path relative to kb_dir
    heading: str  # "Title > Section > Subsection"
    corpus: str
    type: str = ""  # methodology, tool, writeup
    category: str = ""
    tags: str = ""  # comma separated, Chroma metadata must be scalar
    links: str = ""  # wikilink targets, for backlinks later


@dataclass
class Hit:
    text: str
    source: str
    heading: str
    corpus: str
    distance: float
    type: str = ""
    category: str = ""
    tags: str = ""
    links: str = ""

    @property
    def label(self) -> str:
        return f"{self.source} > {self.heading}" if self.heading else self.source


# -- parsing


def parse_frontmatter(text: str) -> tuple[dict, str]:
    """Split YAML frontmatter (Obsidian properties) from the body; raises ValueError if malformed"""
    text = text.replace("\r\n", "\n")
    if not text.startswith("---\n"):
        return {}, text
    end = text.find("\n---", 3)
    if end == -1:
        return {}, text
    try:
        meta = yaml.safe_load(text[4:end]) or {}
    except yaml.YAMLError as e:
        raise ValueError(f"bad frontmatter: {e}") from None
    if not isinstance(meta, dict):
        raise ValueError("frontmatter is not a key: value mapping")
    return meta, text[end + 4 :].lstrip("\n")


def clean_obsidian(body: str) -> tuple[str, list[str]]:
    """Turn Obsidian markup into plain markdown; returns the text and the wikilink targets.

    Code blocks and `inline code` are left alone (bash `[[ -f x ]]` is not a link).
    """
    links: list[str] = []

    def link(m: re.Match) -> str:
        target, heading, alias = (g.strip() if g else "" for g in m.groups())
        if target:
            links.append(target)
        return alias or target or heading

    def inline(text: str) -> str:
        parts = text.split("`")
        for i in range(0, len(parts), 2):  # even parts are outside inline code
            p = _EMBED.sub("", parts[i])
            p = _WIKILINK.sub(link, p)
            parts[i] = _HIGHLIGHT.sub(r"\1", p)
        return "`".join(parts)

    out: list[str] = []
    in_code = in_comment = in_callout = False
    for line in body.splitlines():
        if _FENCE.match(line) and not in_comment:
            in_code = not in_code
            out.append(line)
            continue
        if in_code:
            out.append(line)
            continue

        # %% comments %% can span lines
        parts = line.split("%%")
        kept = [p for i, p in enumerate(parts) if (i % 2 == 1) == in_comment]
        touched = in_comment or len(parts) > 1
        in_comment ^= len(parts) % 2 == 0
        if touched and not "".join(kept).strip():
            continue  # line was only a comment
        line = "".join(kept)

        m = _CALLOUT.match(line)
        if m:
            in_callout = True
            line = m.group(2) or m.group(1).capitalize()
        elif in_callout and line.startswith(">"):
            line = line[2:] if line.startswith("> ") else line[1:]
        else:
            in_callout = False
        out.append(inline(line))
    return "\n".join(out), links


def _blocks(text: str) -> list[str]:
    """Paragraphs split on blank lines, a fenced code block is never split"""
    blocks: list[str] = []
    buf: list[str] = []
    in_code = False
    for line in text.splitlines():
        if _FENCE.match(line):
            in_code = not in_code
        if not line.strip() and not in_code:
            if buf:
                blocks.append("\n".join(buf))
                buf = []
            continue
        buf.append(line)
    if buf:
        blocks.append("\n".join(buf))
    return blocks


def chunk_markdown(body: str, title: str = "") -> list[tuple[str, str]]:
    """Split on headings, keeping the heading path rooted at the note title.

    An H1 replaces the title; sections under MIN_CHUNK_CHARS merge into the next one
    (the last into the previous); sections over MAX_CHUNK_CHARS split on paragraphs.
    """
    sections: list[tuple[str, str]] = []
    path: list[str] = [title] if title else []
    lines: list[str] = []
    in_code = False

    def flush():
        text = "\n".join(lines).strip()
        if text:
            sections.append((" > ".join(path), text))
        lines.clear()

    for line in body.splitlines():
        if _FENCE.match(line):
            in_code = not in_code
        m = None if in_code else _HEADING.match(line)
        if m:
            flush()
            level = len(m.group(1))
            path[:] = path[: level - 1] + [m.group(2).strip()]
        else:
            lines.append(line)
    flush()

    # merge tiny sections, keeping their heading name as a lead-in line
    merged: list[tuple[str, str]] = []
    carry = ""
    for heading, text in sections:
        text = carry + text
        carry = ""
        if len(text) < MIN_CHUNK_CHARS:
            name = heading.rsplit(" > ", 1)[-1]
            carry = f"{name}:\n{text}\n\n" if name and name != title else f"{text}\n\n"
            continue
        merged.append((heading, text))
    if carry:
        if merged:
            heading, text = merged[-1]
            merged[-1] = (heading, f"{text}\n\n{carry.strip()}")
        else:
            merged.append((sections[-1][0], carry.strip()))

    chunks = []
    for heading, text in merged:
        if len(text) <= MAX_CHUNK_CHARS:
            chunks.append((heading, text))
            continue
        buf = ""
        for block in _blocks(text):
            if buf and len(buf) + len(block) > MAX_CHUNK_CHARS:
                chunks.append((heading, buf.strip()))
                buf = ""
            buf += block + "\n\n"
        if buf.strip():
            chunks.append((heading, buf.strip()))
    return chunks


def _as_text(value) -> str:
    """Frontmatter values for Chroma metadata: lists become comma separated strings"""
    if value is None:
        return ""
    if isinstance(value, list):
        return ", ".join(str(v) for v in value)
    return str(value)


def is_corpus_note(rel: Path) -> bool:
    """Notes inside a corpus folder; skips loose root files, _templates and dot folders"""
    if len(rel.parts) < 2 or rel.parts[0].startswith("_"):
        return False
    return not any(p.startswith(".") for p in rel.parts)


@dataclass
class LoadResult:
    chunks: list[Chunk]
    skipped: list[str]  # not vetted
    invalid: list[tuple[str, str]]  # (path, reason)


def load_notes(kb_dir: Path) -> LoadResult:
    """Chunk every vetted note; unvetted and malformed notes are reported, not ingested"""
    result = LoadResult([], [], [])
    for path in sorted(kb_dir.rglob("*.md")):
        rel = path.relative_to(kb_dir)
        if not is_corpus_note(rel):
            continue
        try:
            meta, body = parse_frontmatter(path.read_text(encoding="utf-8"))
        except (ValueError, UnicodeDecodeError) as e:
            result.invalid.append((str(rel), str(e)))
            continue
        if meta.get("vetted") is not True:
            result.skipped.append(str(rel))
            continue

        body, links = clean_obsidian(body)
        common = dict(
            source=str(rel),
            corpus=rel.parts[0],
            type=_as_text(meta.get("type")),
            category=_as_text(meta.get("category")),
            tags=_as_text(meta.get("tags")),
            links=", ".join(dict.fromkeys(links)),
        )
        for i, (heading, text) in enumerate(chunk_markdown(body, title=path.stem)):
            result.chunks.append(Chunk(id=f"{rel}#{i}", text=text, heading=heading, **common))
    return result


# -- store


class KnowledgeBase:
    def __init__(self, cfg: RAGConfig, client: OllamaClient, fresh: bool = False):
        """fresh=True drops the stored collection first (used by ingest)"""
        self.cfg = cfg
        self.client = client
        self._db = chromadb.PersistentClient(path=str(cfg.db_path))
        if fresh and COLLECTION in {c.name for c in self._db.list_collections()}:
            self._db.delete_collection(COLLECTION)
        self._col = self._db.get_or_create_collection(
            COLLECTION,
            configuration={"hnsw": {"space": "cosine"}},
            embedding_function=None,
            metadata={"embed_model": cfg.embed_model},
        )
        stored = (self._col.metadata or {}).get("embed_model")
        if stored != cfg.embed_model:
            raise RuntimeError(
                f"KB was embedded with '{stored}' but config says '{cfg.embed_model}'. "
                "Run magetower-ingest to re-embed."
            )

    def count(self) -> int:
        return self._col.count()

    def ingest(self, chunks: list[Chunk]) -> None:
        for i in range(0, len(chunks), EMBED_BATCH):
            batch = chunks[i : i + EMBED_BATCH]
            # heading path and tags go into the embedded text so short sections keep their context
            inputs = [
                f"{DOC_PREFIX}{c.heading}\n" + (f"tags: {c.tags}\n" if c.tags else "") + c.text
                for c in batch
            ]
            self._col.add(
                ids=[c.id for c in batch],
                embeddings=self.client.embed(inputs, self.cfg.embed_model),
                documents=[c.text for c in batch],
                metadatas=[
                    {
                        "source": c.source,
                        "heading": c.heading,
                        "corpus": c.corpus,
                        "type": c.type,
                        "category": c.category,
                        "tags": c.tags,
                        "links": c.links,
                    }
                    for c in batch
                ],
            )

    def search(self, query: str, n: int | None = None) -> list[Hit]:
        """Nearest chunks in the active corpora, without the distance cutoff"""
        if self.count() == 0:
            return []
        corpora = [BASELINE, *self.cfg.active_corpora]
        (vec,) = self.client.embed([QUERY_PREFIX + query], self.cfg.embed_model)
        res = self._col.query(
            query_embeddings=[vec],
            n_results=n or self.cfg.top_k,
            where={"corpus": {"$in": corpora}},
        )
        known = Hit.__dataclass_fields__.keys()
        return [
            Hit(text=doc, distance=dist, **{k: v for k, v in meta.items() if k in known})
            for doc, meta, dist in zip(
                res["documents"][0], res["metadatas"][0], res["distances"][0]
            )
        ]

    def retrieve(self, query: str) -> list[Hit]:
        return [h for h in self.search(query) if h.distance <= self.cfg.max_distance]


def format_context(hits: list[Hit], question: str) -> str:
    """Wrap the user's question with retrieved notes for the model"""
    if not hits:
        return question
    notes = "\n\n".join(
        f"[{i}] {h.label}\n{h.text}" for i, h in enumerate(hits, start=1)
    )
    return (
        "Reference notes from the vetted knowledge base:\n\n"
        f"{notes}\n\n---\n\nQuestion: {question}"
    )
