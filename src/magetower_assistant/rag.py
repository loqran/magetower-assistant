"""
Knowledge base: markdown notes -> chunks -> Ollama embeddings -> Chroma.

Layout: <kb_dir>/<corpus>/**/*.md
    - the top-level folder name is the corpus ("baseline", "maritime", ...)
    - baseline is always searched, specialist corpora only when listed in config

Only notes with `vetted: true` in their frontmatter are ingested.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import chromadb

from .config import RAGConfig
from .llm import OllamaClient

BASELINE = "baseline"
COLLECTION = "kb_notes"

# nomic-embed-text was trained with these task prefixes, retrieval degrades without them
DOC_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "

MAX_CHUNK_CHARS = 1500  # placeholder until the Stage 3 chunking strategy
EMBED_BATCH = 32

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


@dataclass
class Chunk:
    id: str
    text: str
    source: str  # path relative to kb_dir
    heading: str  # "Title > Section > Subsection"
    corpus: str


@dataclass
class Hit:
    text: str
    source: str
    heading: str
    corpus: str
    distance: float

    @property
    def label(self) -> str:
        return f"{self.source} > {self.heading}" if self.heading else self.source


# -- parsing


def parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Minimal `key: value` frontmatter parser, enough for flags like vetted/tags"""
    if not text.startswith("---\n"):
        return {}, text
    end = text.find("\n---", 4)
    if end == -1:
        return {}, text
    meta = {}
    for line in text[4:end].splitlines():
        key, sep, value = line.partition(":")
        if sep:
            meta[key.strip()] = value.strip()
    return meta, text[end + 4 :].lstrip("\n")


def chunk_markdown(body: str) -> list[tuple[str, str]]:
    """Split on headings, keeping the heading path; oversized sections split on paragraphs"""
    sections: list[tuple[str, str]] = []
    path: list[str] = []
    lines: list[str] = []
    in_code = False

    def flush():
        text = "\n".join(lines).strip()
        if text:
            sections.append((" > ".join(path), text))
        lines.clear()

    for line in body.splitlines():
        if line.lstrip().startswith("```"):
            in_code = not in_code
        m = None if in_code else _HEADING.match(line)
        if m:
            flush()
            level = len(m.group(1))
            path[:] = path[: level - 1] + [m.group(2).strip()]
        else:
            lines.append(line)
    flush()

    chunks = []
    for heading, text in sections:
        if len(text) <= MAX_CHUNK_CHARS:
            chunks.append((heading, text))
            continue
        buf = ""
        for para in text.split("\n\n"):
            if buf and len(buf) + len(para) > MAX_CHUNK_CHARS:
                chunks.append((heading, buf.strip()))
                buf = ""
            buf += para + "\n\n"
        if buf.strip():
            chunks.append((heading, buf.strip()))
    return chunks


def load_notes(kb_dir: Path) -> tuple[list[Chunk], list[str]]:
    """Return chunks from vetted notes, plus the paths of skipped unvetted notes"""
    chunks: list[Chunk] = []
    skipped: list[str] = []
    for path in sorted(kb_dir.rglob("*.md")):
        rel = path.relative_to(kb_dir)
        if len(rel.parts) < 2:
            continue  # loose files at the kb root (README etc) belong to no corpus
        meta, body = parse_frontmatter(path.read_text(encoding="utf-8"))
        if meta.get("vetted", "").lower() != "true":
            skipped.append(str(rel))
            continue
        for i, (heading, text) in enumerate(chunk_markdown(body)):
            chunks.append(
                Chunk(
                    id=f"{rel}#{i}",
                    text=text,
                    source=str(rel),
                    heading=heading,
                    corpus=rel.parts[0],
                )
            )
    return chunks, skipped


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
            # the heading path goes into the embedded text so short sections keep their context
            inputs = [f"{DOC_PREFIX}{c.heading}\n{c.text}" for c in batch]
            self._col.add(
                ids=[c.id for c in batch],
                embeddings=self.client.embed(inputs, self.cfg.embed_model),
                documents=[c.text for c in batch],
                metadatas=[
                    {"source": c.source, "heading": c.heading, "corpus": c.corpus}
                    for c in batch
                ],
            )

    def retrieve(self, query: str) -> list[Hit]:
        corpora = [BASELINE, *self.cfg.active_corpora]
        (vec,) = self.client.embed([QUERY_PREFIX + query], self.cfg.embed_model)
        res = self._col.query(
            query_embeddings=[vec],
            n_results=self.cfg.top_k,
            where={"corpus": {"$in": corpora}},
        )
        hits = [
            Hit(text=doc, distance=dist, **meta)
            for doc, meta, dist in zip(
                res["documents"][0], res["metadatas"][0], res["distances"][0]
            )
        ]
        return [h for h in hits if h.distance <= self.cfg.max_distance]


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
