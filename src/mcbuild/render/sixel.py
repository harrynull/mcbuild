"""Pure-python sixel encoder + terminal capability detection (DA1 query)."""

from __future__ import annotations

import sys
import time

from PIL import Image


def _emit_run(ch: str, length: int) -> str:
    if length <= 3:
        return ch * length
    return f"!{length}{ch}"


def encode_sixel(img: Image.Image, max_colors: int = 255, bg: tuple[int, int, int] = (24, 24, 28)) -> str:
    """Encode a PIL image as a sixel DCS string (flattens alpha onto `bg`)."""
    rgba = img.convert("RGBA")
    flat = Image.new("RGB", rgba.size, bg)
    flat.paste(rgba, (0, 0), rgba)
    quantized = flat.quantize(colors=max_colors, method=Image.MEDIANCUT)
    palette = quantized.getpalette() or []
    width, height = quantized.size
    pixels = quantized.load()

    used_colors = sorted({pixels[x, y] for y in range(height) for x in range(width)})

    # "1;1;<Ph>;<Pv>" raster attributes: 1:1 pixel aspect ratio + explicit image size.
    # Without this many terminals guess the aspect ratio (often 2:1) and stretch/squish
    # the image instead of rendering it at its true pixel dimensions.
    parts = ["\x1bPq", f'"1;1;{width};{height}']
    for c in used_colors:
        r = palette[c * 3] if c * 3 < len(palette) else 0
        g = palette[c * 3 + 1] if c * 3 + 1 < len(palette) else 0
        b = palette[c * 3 + 2] if c * 3 + 2 < len(palette) else 0
        parts.append(f"#{c};2;{r * 100 // 255};{g * 100 // 255};{b * 100 // 255}")

    for band_y in range(0, height, 6):
        band_h = min(6, height - band_y)
        colors_in_band: set[int] = set()
        for yy in range(band_h):
            for xx in range(width):
                colors_in_band.add(pixels[xx, band_y + yy])

        first = True
        for color in sorted(colors_in_band):
            if not first:
                parts.append("$")
            first = False
            run_char: str | None = None
            run_len = 0
            row_chars: list[str] = []
            for xx in range(width):
                bits = 0
                for yy in range(band_h):
                    if pixels[xx, band_y + yy] == color:
                        bits |= 1 << yy
                ch = chr(63 + bits)
                if ch == run_char:
                    run_len += 1
                else:
                    if run_char is not None:
                        row_chars.append(_emit_run(run_char, run_len))
                    run_char = ch
                    run_len = 1
            if run_char is not None:
                row_chars.append(_emit_run(run_char, run_len))
            parts.append(f"#{color}" + "".join(row_chars))
        parts.append("-")
    parts.append("\x1b\\")
    return "".join(parts)


def supports_sixel(timeout: float = 0.3) -> bool:
    """Detect sixel support via a DA1 (Device Attributes) terminal query."""
    try:
        if not sys.stdout.isatty():
            return False
    except Exception:
        return False
    try:
        if sys.platform == "win32":
            return _query_da1_windows(timeout)
        return _query_da1_posix(timeout)
    except Exception:
        return False


def _query_da1_windows(timeout: float) -> bool:
    import msvcrt

    sys.stdout.write("\x1b[c")
    sys.stdout.flush()
    end = time.monotonic() + timeout
    response = ""
    while time.monotonic() < end:
        if msvcrt.kbhit():
            ch = msvcrt.getwch()
            response += ch
            if ch == "c":
                break
        else:
            time.sleep(0.01)
    return ";4;" in response or ";4c" in response


def _query_da1_posix(timeout: float) -> bool:
    import select
    import termios
    import tty

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    response = ""
    try:
        tty.setraw(fd)
        sys.stdout.write("\x1b[c")
        sys.stdout.flush()
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            r, _, _ = select.select([fd], [], [], 0.05)
            if r:
                ch = sys.stdin.read(1)
                response += ch
                if ch == "c":
                    break
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return ";4;" in response or ";4c" in response
