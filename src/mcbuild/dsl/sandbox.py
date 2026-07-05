"""AST-whitelist sandbox + op-budget executor for blueprint code.

Threat model: LLM accidents (infinite loops, silly mistakes), not a malicious
adversary — execution stays in-process. Import statements are stripped rather
than rejected (everything a blueprint needs is already injected as globals;
see dsl/REFERENCE.md), dunder/attribute escapes and a handful of dangerous
builtins are blocked outright, and execution runs under a line-count +
wall-clock budget via sys.settrace.
"""

from __future__ import annotations

import ast
import builtins as _builtins
import sys
import time

from mcbuild import palette
from mcbuild import voxel as _voxel
from mcbuild.dsl import errors, stdlib
from mcbuild.palette import PaletteError

BLUEPRINT_FILENAME = "<blueprint>"

MAX_LINES = 2_000_000
MAX_SECONDS = 10.0

BANNED_NAMES = {
    "eval",
    "exec",
    "open",
    "__import__",
    "getattr",
    "setattr",
    "delattr",
    "globals",
    "locals",
    "vars",
    "type",
    "compile",
    "input",
    "breakpoint",
    "__builtins__",
    "memoryview",
    "staticmethod",
    "classmethod",
    "super",
    "exit",
    "quit",
    "help",
    "__build_class__",
}

_SAFE_BUILTIN_NAMES = (
    "range",
    "len",
    "enumerate",
    "zip",
    "min",
    "max",
    "abs",
    "round",
    "int",
    "float",
    "str",
    "list",
    "dict",
    "set",
    "tuple",
    "sorted",
    "sum",
    "any",
    "all",
    "bool",
    "print",
)

SAFE_BUILTINS: dict[str, object] = {
    name: getattr(_builtins, name) for name in _SAFE_BUILTIN_NAMES if hasattr(_builtins, name)
}


class SandboxViolation(Exception):
    """Raised when a blueprint's AST contains disallowed constructs."""

    def __init__(self, message: str, lineno: int | None = None):
        self.lineno = lineno
        super().__init__(message)


class BudgetExceeded(Exception):
    """Raised when a blueprint exceeds its executed-line or wall-clock budget."""


def validate_ast(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            raise SandboxViolation(f"Access to attribute '{node.attr}' is not allowed.", node.lineno)
        if isinstance(node, ast.Name) and node.id in BANNED_NAMES:
            raise SandboxViolation(f"Use of '{node.id}' is not allowed.", node.lineno)


class _ImportStripper(ast.NodeTransformer):
    """Removes import statements wholesale instead of rejecting the whole blueprint.

    Everything a blueprint needs is already injected as globals (see dsl/REFERENCE.md),
    so a stray `import` is harmless to drop; if the blueprint actually relied on the
    imported name, it fails naturally with a NameError at the point of use instead.
    """

    def visit_Import(self, node: ast.Import) -> None:
        return None

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        return None

    def generic_visit(self, node: ast.AST) -> ast.AST:
        node = super().generic_visit(node)
        body = getattr(node, "body", None)
        if isinstance(body, list) and not body:
            # A block can't be empty; fix_missing_locations (called by the caller) fills lineno.
            setattr(node, "body", [ast.Pass()])  # noqa: B010
        return node


def compile_blueprint(source: str, filename: str = BLUEPRINT_FILENAME) -> ast.Module:
    tree = ast.parse(source, filename=filename)
    tree = _ImportStripper().visit(tree)
    ast.fix_missing_locations(tree)
    validate_ast(tree)
    return tree


def run_blueprint(
    source: str,
    grid: _voxel.VoxelGrid,
    seed: int = 0,
    max_lines: int = MAX_LINES,
    max_seconds: float = MAX_SECONDS,
) -> None:
    """Validate, compile, and execute a blueprint against the given VoxelGrid.

    Raises BlueprintError (with line + excerpt) on any failure: syntax error,
    sandbox violation, budget overrun, or a runtime exception from user code.
    """
    filename = BLUEPRINT_FILENAME
    try:
        tree = compile_blueprint(source, filename)
    except SyntaxError as e:
        raise errors.from_syntax_error(e, source) from e
    except SandboxViolation as e:
        raise errors.from_sandbox_violation(e, source) from e

    code = compile(tree, filename, "exec")
    stdlib_ns = stdlib.make_stdlib(grid, seed=seed)
    global_ns: dict[str, object] = {"__builtins__": SAFE_BUILTINS, **stdlib_ns}

    counter = {"n": 0}
    start = time.monotonic()

    def tracer(frame, event, arg):
        if frame.f_code.co_filename != filename:
            return None
        if event == "line":
            counter["n"] += 1
            if counter["n"] > max_lines:
                raise BudgetExceeded(f"Blueprint exceeded {max_lines:,} executed lines (possible infinite loop).")
            if counter["n"] % 2000 == 0 and time.monotonic() - start > max_seconds:
                raise BudgetExceeded(f"Blueprint exceeded {max_seconds}s execution budget.")
        return tracer

    old_trace = sys.gettrace()
    sys.settrace(tracer)
    palette.reset_warnings()
    try:
        exec(code, global_ns)
    except BudgetExceeded as e:
        raise errors.BlueprintError(str(e), source=source) from e
    except (_voxel.VoxelLimitError, PaletteError) as e:
        raise errors.from_exception(e, source, filename) from e
    except Exception as e:
        raise errors.from_exception(e, source, filename) from e
    finally:
        sys.settrace(old_trace)
    grid.palette_warnings = palette.pop_warnings()
