"""runs/<timestamp>-<slug>/ artifact management."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image


def slugify(text: str, max_len: int = 40) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:max_len] or "build"


@dataclass
class RunDir:
    root: Path

    @classmethod
    def create(cls, prompt: str, base: str = "runs") -> RunDir:
        ts = time.strftime("%Y%m%d-%H%M%S")
        slug = slugify(prompt)
        root = Path(base) / f"{ts}-{slug}"
        root.mkdir(parents=True, exist_ok=True)
        rd = cls(root)
        rd.write_text("prompt.txt", prompt)
        return rd

    def path(self, name: str) -> Path:
        p = self.root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def write_text(self, name: str, content: str) -> Path:
        p = self.path(name)
        p.write_text(content, encoding="utf-8")
        return p

    def write_bytes(self, name: str, content: bytes) -> Path:
        p = self.path(name)
        p.write_bytes(content)
        return p

    def write_json(self, name: str, data: Any) -> Path:
        return self.write_text(name, json.dumps(data, indent=2, default=str))

    def save_image(self, name: str, img: Image.Image) -> Path:
        p = self.path(name)
        img.save(p)
        return p

    def iter_dir(self, n: int) -> Path:
        d = self.root / f"iter_{n:02d}"
        d.mkdir(parents=True, exist_ok=True)
        return d
