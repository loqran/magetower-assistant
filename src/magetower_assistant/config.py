"""
    load runtime config from config.toml

Lookup order:
    1. $CTF_ASSIST_CONFIG
    2. ./config.toml in working directory
    3. config.toml at project root
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class LLMConfig:
    host: str = "http://localhost:11434"
    model: str = "granite3.3:8b"
    temperature: float = 0.3
    num_ctx: int = 8192


@dataclass
class ChatConfig:
    max_history_turns: int = 10


@dataclass
class Config:
    llm: LLMConfig = field(default_factory=LLMConfig)
    chat: ChatConfig = field(default_factory=ChatConfig)
    source: Path | None = None


def _find_config_file() -> Path | None:
    env = os.environ.get("CTF_ASSIST_CONFIG")
    if env:
        path = Path(env).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"CTF_ASSIST_CONFIG points at missing file: {path}")
        return path
    for candidate in (Path.cwd() / "config.toml", PROJECT_ROOT / "config.toml"):
        if candidate.is_file():
            return candidate
    return None


def load_config() -> Config:
    path = _find_config_file()
    if path is None:
        return Config()

    with path.open("rb") as f:
        raw: dict[str, Any] = tomllib.load(f)

    # only pass keys the dataclasses know about, prevent crashes
    def pick(cls, section: dict):
        known = cls.__dataclass_fields__.keys()
        return cls(**{k: v for k, v in section.items() if k in known})

    return Config(
        llm=pick(LLMConfig, raw.get("llm", {})),
        chat=pick(ChatConfig, raw.get("chat", {})),
        source=path,
    )
