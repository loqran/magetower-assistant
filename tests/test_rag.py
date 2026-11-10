from pathlib import Path

import pytest

from magetower_assistant import rag
from magetower_assistant.rag import (
    chunk_markdown,
    clean_obsidian,
    is_corpus_note,
    load_notes,
    parse_frontmatter,
)

LONG = "word " * 60  # over MIN_CHUNK_CHARS


# -- frontmatter


def test_frontmatter_yaml_lists_and_bools():
    meta, body = parse_frontmatter("---\nvetted: true\ntags: [jwt, auth]\n---\n# Title\n")
    assert meta == {"vetted": True, "tags": ["jwt", "auth"]}
    assert body == "# Title\n"


def test_frontmatter_obsidian_list_style():
    meta, _ = parse_frontmatter("---\ntags:\n  - a\n  - b\n---\nbody")
    assert meta["tags"] == ["a", "b"]


def test_no_frontmatter():
    assert parse_frontmatter("just text") == ({}, "just text")


def test_empty_frontmatter():
    assert parse_frontmatter("---\n---\nbody") == ({}, "body")


def test_bad_frontmatter_raises():
    with pytest.raises(ValueError):
        parse_frontmatter("---\ntags: [unclosed\n---\nbody")


def test_crlf_frontmatter():
    meta, body = parse_frontmatter("---\r\nvetted: true\r\n---\r\nbody")
    assert meta == {"vetted": True} and body == "body"


# -- obsidian cleanup


def test_wikilinks():
    text, links = clean_obsidian("use [[tshark]] or [[wireshark|the GUI]], see [[tshark#Fields]]")
    assert text == "use tshark or the GUI, see tshark"
    assert links == ["tshark", "wireshark", "tshark"]


def test_heading_only_link():
    text, links = clean_obsidian("see [[#Steps]]")
    assert text == "see Steps" and links == []


def test_embeds_highlights_comments():
    text, _ = clean_obsidian("a ![[diagram.png]]b ==key== c %%hidden%% d")
    assert text == "a b key c  d"


def test_multiline_comment_dropped():
    text, _ = clean_obsidian("keep\n%%\nsecret\nlines\n%%\nalso keep")
    assert text == "keep\nalso keep"


def test_callout_unwrapped():
    text, _ = clean_obsidian("> [!check] Verify this flag\n> the body\nafter")
    assert text == "Verify this flag\nthe body\nafter"


def test_callout_without_title_uses_type():
    text, _ = clean_obsidian("> [!warning]\n> careful")
    assert text == "Warning\ncareful"


def test_code_left_alone():
    body = "```bash\n[[ -f x ]] && echo ==y== %%\n```\nrun `[[ -n $a ]]` here"
    text, links = clean_obsidian(body)
    assert text == body and links == []


# -- chunking


def test_title_roots_heading_path():
    chunks = chunk_markdown(f"## Steps\n{LONG}\n### Detail\n{LONG}", title="web-jwt")
    assert [h for h, _ in chunks] == ["web-jwt > Steps", "web-jwt > Steps > Detail"]


def test_h1_replaces_title():
    chunks = chunk_markdown(f"# JSON Web Tokens\n{LONG}\n## Attacks\n{LONG}", title="web-jwt")
    assert [h for h, _ in chunks] == ["JSON Web Tokens", "JSON Web Tokens > Attacks"]


def test_no_headings():
    assert chunk_markdown(LONG, title="note") == [("note", LONG.strip())]


def test_tiny_section_merges_into_next():
    chunks = chunk_markdown(f"## Recognise it\nthree dots\n## Approach\n{LONG}", title="t")
    assert len(chunks) == 1
    heading, text = chunks[0]
    assert heading == "t > Approach"
    assert text.startswith("Recognise it:\nthree dots")


def test_tiny_last_section_merges_into_previous():
    chunks = chunk_markdown(f"## Approach\n{LONG}\n## References\nRFC 7519", title="t")
    assert len(chunks) == 1
    assert chunks[0][1].endswith("References:\nRFC 7519")


def test_heading_inside_code_is_not_a_heading():
    chunks = chunk_markdown(f"## Steps\n```\n# comment\n```\n{LONG}", title="t")
    assert len(chunks) == 1 and "# comment" in chunks[0][1]


def test_long_section_never_splits_code_block(monkeypatch):
    monkeypatch.setattr(rag, "MAX_CHUNK_CHARS", 300)
    code = "```\n" + "\n\n".join(f"line {i} " * 5 for i in range(10)) + "\n```"
    chunks = chunk_markdown(f"## A\n{LONG}\n\n{code}\n\n{LONG}", title="t")
    assert any(text == code for _, text in chunks)
    for _, text in chunks:
        assert text.count("```") % 2 == 0


# -- loading


def test_corpus_folders():
    assert is_corpus_note(Path("baseline/methodology/web-jwt.md"))
    assert not is_corpus_note(Path("README.md"))
    assert not is_corpus_note(Path("_templates/tool.md"))
    assert not is_corpus_note(Path(".obsidian/x.md"))
    assert not is_corpus_note(Path("baseline/.trash/old.md"))


def _note(path: Path, front: str, body: str = f"## Steps\n{LONG}") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"---\n{front}\n---\n{body}", encoding="utf-8")


def test_load_notes(tmp_path):
    _note(
        tmp_path / "baseline/methodology/web-jwt.md",
        "type: methodology\ncategory: web\ntags: [jwt, auth]\nvetted: true",
        f"## Steps\nsee [[jwt_tool]] {LONG}",
    )
    _note(tmp_path / "baseline/methodology/draft.md", "vetted: false")
    _note(tmp_path / "baseline/methodology/stringy.md", 'vetted: "true"')
    _note(tmp_path / "baseline/methodology/broken.md", "tags: [oops")
    _note(tmp_path / "_templates/tool.md", "vetted: true")

    result = load_notes(tmp_path)
    assert [c.source for c in result.chunks] == ["baseline/methodology/web-jwt.md"]
    c = result.chunks[0]
    assert (c.corpus, c.type, c.category, c.tags, c.links) == (
        "baseline", "methodology", "web", "jwt, auth", "jwt_tool"
    )
    assert c.heading == "web-jwt > Steps"
    assert sorted(result.skipped) == [
        "baseline/methodology/draft.md", "baseline/methodology/stringy.md"
    ]
    assert [p for p, _ in result.invalid] == ["baseline/methodology/broken.md"]
