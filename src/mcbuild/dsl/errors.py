"""BlueprintError: exceptions mapped to blueprint source line numbers with a code excerpt."""

from __future__ import annotations

import traceback


class BlueprintError(Exception):
    """A blueprint execution error, annotated with the offending line and a code excerpt."""

    def __init__(self, message: str, line: int | None = None, source: str | None = None):
        self.line = line
        self.source = source
        self.raw_message = message
        super().__init__(self._format(message))

    def _format(self, message: str) -> str:
        if self.line is None or not self.source:
            return message
        lines = self.source.splitlines()
        lo = max(1, self.line - 3)
        hi = min(len(lines), self.line + 3)
        excerpt_lines = []
        for i in range(lo, hi + 1):
            marker = ">>" if i == self.line else "  "
            excerpt_lines.append(f"{marker} {i:4d} | {lines[i - 1]}")
        excerpt = "\n".join(excerpt_lines)
        return f"{message} (line {self.line})\n{excerpt}"


def from_syntax_error(exc: SyntaxError, source: str) -> BlueprintError:
    return BlueprintError(f"SyntaxError: {exc.msg}", line=exc.lineno, source=source)


def from_sandbox_violation(exc: Exception, source: str) -> BlueprintError:
    return BlueprintError(str(exc), line=getattr(exc, "lineno", None), source=source)


def from_exception(exc: BaseException, source: str, filename: str = "<blueprint>") -> BlueprintError:
    """Wrap an arbitrary exception raised during blueprint execution.

    Walks the traceback for the deepest frame whose filename matches the blueprint's
    compiled filename, so the reported line is the one nearest the actual mistake.
    """
    lineno = None
    for frame_summary in traceback.extract_tb(exc.__traceback__):
        if frame_summary.filename == filename:
            lineno = frame_summary.lineno
    message = f"{type(exc).__name__}: {exc}"
    return BlueprintError(message, line=lineno, source=source)
