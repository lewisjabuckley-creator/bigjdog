"""Images without required dependencies: what format an image is, its size, whether it is intact, and getting it
ready for a vision model.

Headers of PNG, JPEG, GIF, BMP, WebP and TIFF are parsed in pure Python, so JARVIS can validate and describe any
image with nothing installed. A truncated or corrupt file is refused with a reason instead of being sent to a model.
Shrinking large images needs Pillow (optional); without it a large image that a model would choke on is refused and
the reply says how to enable shrinking. PNG is also decoded in pure Python (8-bit, the format screen captures use) so
two screen captures can be compared pixel by pixel without Pillow.
"""

from __future__ import annotations

import io
import struct
import zlib
from dataclasses import dataclass
from itertools import accumulate


class ImageError(ValueError):
    """The input is not a usable image (unsupported, corrupt, truncated, or too large to handle)."""


@dataclass
class ImageInfo:
    format: str          # png | jpeg | gif | bmp | webp | tiff
    mime: str
    width: int
    height: int
    size_bytes: int

    @property
    def pixels(self) -> int:
        return self.width * self.height


_MIME = {"png": "image/png", "jpeg": "image/jpeg", "gif": "image/gif", "bmp": "image/bmp", "webp": "image/webp",
         "tiff": "image/tiff"}
# formats a vision model reliably accepts as-is; others are converted (needs Pillow)
MODEL_FORMATS = {"png", "jpeg"}


def sniff(data: bytes) -> str | None:
    """The image format from its first bytes, or None if it isn't an image JARVIS knows."""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if data.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "gif"
    if data.startswith(b"BM") and len(data) > 26:
        return "bmp"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    if data[:4] in (b"II*\x00", b"MM\x00*"):
        return "tiff"
    return None


def image_info(data: bytes) -> ImageInfo:
    """Format and dimensions, checking the file is complete. Raises ImageError when it isn't a usable image."""
    fmt = sniff(data)
    if fmt is None:
        raise ImageError("it isn't an image format I can read (PNG, JPEG, GIF, BMP, WebP or TIFF)")
    try:
        width, height = {"png": _png_size, "jpeg": _jpeg_size, "gif": _gif_size, "bmp": _bmp_size,
                         "webp": _webp_size, "tiff": _tiff_size}[fmt](data)
    except (struct.error, IndexError, ValueError) as exc:
        raise ImageError(f"the {fmt.upper()} file is damaged ({exc})") from None
    if width <= 0 or height <= 0 or width > 100_000 or height > 100_000:
        raise ImageError(f"the {fmt.upper()} file reports an impossible size ({width}×{height})")
    return ImageInfo(fmt, _MIME[fmt], width, height, len(data))


def _png_size(data: bytes) -> tuple[int, int]:
    if data[12:16] != b"IHDR":
        raise ValueError("missing header chunk")
    width, height = struct.unpack(">II", data[16:24])
    if b"IEND" not in data[-16:]:
        raise ValueError("the file is truncated")
    return width, height


def _jpeg_size(data: bytes) -> tuple[int, int]:
    if not data.rstrip(b"\x00").endswith(b"\xff\xd9"):
        raise ValueError("the file is truncated")
    i = 2
    while i < len(data):
        if data[i] != 0xFF:
            raise ValueError("unexpected data between segments")
        marker = data[i + 1]
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            i += 2
            continue
        length = struct.unpack(">H", data[i + 2:i + 4])[0]
        if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
            height, width = struct.unpack(">HH", data[i + 5:i + 9])
            return width, height
        i += 2 + length
    raise ValueError("no frame header")


def _gif_size(data: bytes) -> tuple[int, int]:
    width, height = struct.unpack("<HH", data[6:10])
    if not data.rstrip(b"\x00").endswith(b";"):
        raise ValueError("the file is truncated")
    return width, height


def _bmp_size(data: bytes) -> tuple[int, int]:
    size = struct.unpack("<I", data[2:6])[0]
    if size and size > len(data):
        raise ValueError("the file is truncated")
    width, height = struct.unpack("<ii", data[18:26])
    return width, abs(height)


def _webp_size(data: bytes) -> tuple[int, int]:
    riff = struct.unpack("<I", data[4:8])[0]
    if riff + 8 > len(data):
        raise ValueError("the file is truncated")
    chunk = data[12:16]
    if chunk == b"VP8 ":
        width, height = struct.unpack("<HH", data[26:30])
        return width & 0x3FFF, height & 0x3FFF
    if chunk == b"VP8L":
        b = data[21:25]
        width = 1 + (((b[1] & 0x3F) << 8) | b[0])
        height = 1 + (((b[3] & 0x0F) << 10) | (b[2] << 2) | ((b[1] & 0xC0) >> 6))
        return width, height
    if chunk == b"VP8X":
        width = 1 + int.from_bytes(data[24:27], "little")
        height = 1 + int.from_bytes(data[27:30], "little")
        return width, height
    raise ValueError("unknown WebP layout")


def _tiff_size(data: bytes) -> tuple[int, int]:
    endian = "<" if data[:2] == b"II" else ">"
    offset = struct.unpack(endian + "I", data[4:8])[0]
    count = struct.unpack(endian + "H", data[offset:offset + 2])[0]
    width = height = 0
    for n in range(count):
        entry = data[offset + 2 + n * 12: offset + 14 + n * 12]
        tag, typ = struct.unpack(endian + "HH", entry[:4])
        value = struct.unpack(endian + ("H" if typ == 3 else "I"), entry[8:10] if typ == 3 else entry[8:12])[0]
        if tag == 256:
            width = value
        elif tag == 257:
            height = value
    return width, height


# -- getting an image ready for a model ------------------------------------------------------------------------------

def pillow_available() -> bool:
    import importlib.util
    try:
        return importlib.util.find_spec("PIL") is not None
    except (ImportError, ValueError):
        return False


@dataclass
class Prepared:
    data: bytes
    info: ImageInfo          # of what will be sent
    original: ImageInfo
    note: str = ""           # e.g. "shrunk from 3840×2160 to 1568×882"


def prepare_for_model(data: bytes, *, max_side: int = 1568, max_pixels: int = 40_000_000,
                      max_bytes: int = 20_000_000) -> Prepared:
    """The image as a vision model should get it: PNG or JPEG, no bigger than it needs to be.

    With Pillow: large images are shrunk (aspect ratio kept) and other formats converted. Without it: PNG and JPEG
    within the limits go as they are; anything that would need converting or shrinking is refused with a reason."""
    info = image_info(data)
    needs_resize = max(info.width, info.height) > max_side
    needs_convert = info.format not in MODEL_FORMATS
    if not needs_resize and not needs_convert and len(data) <= max_bytes:
        return Prepared(data, info, info)
    if pillow_available():
        return _pillow_prepare(data, info, max_side)
    if needs_convert:
        raise ImageError(f"{info.format.upper()} images need converting before a vision model can read them, and the "
                         "Pillow package isn't installed (py -m pip install pillow)")
    if info.pixels > max_pixels or len(data) > max_bytes:
        raise ImageError(f"the image is too large to send as it is ({info.width}×{info.height}, "
                         f"{len(data) / 1e6:.1f} MB) and can't be shrunk without the Pillow package "
                         "(py -m pip install pillow)")
    return Prepared(data, info, info, note=f"sent at full size ({info.width}×{info.height}); install Pillow to let me "
                                          "shrink large images, which is faster")


def _pillow_prepare(data: bytes, info: ImageInfo, max_side: int) -> Prepared:
    from PIL import Image
    try:
        with Image.open(io.BytesIO(data)) as img:
            img.load()
            if getattr(img, "is_animated", False):
                img.seek(0)
            scale = min(1.0, max_side / max(img.width, img.height))
            out = img.convert("RGBA" if img.mode in ("RGBA", "LA", "P") else "RGB")
            if scale < 1.0:
                out = out.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))), Image.LANCZOS)
            buf = io.BytesIO()
            if out.mode == "RGBA":
                out.save(buf, format="PNG", optimize=True)
            else:
                out.save(buf, format="JPEG", quality=90)
    except Exception as exc:
        raise ImageError(f"the image couldn't be decoded ({exc})") from None
    prepared = buf.getvalue()
    new = image_info(prepared)
    note = f"shrunk from {info.width}×{info.height} to {new.width}×{new.height}" if scale < 1.0 else \
        f"converted from {info.format.upper()}"
    return Prepared(prepared, new, info, note)


# -- PNG in pure Python (screen captures, tests, simulated screens) ----------------------------------------------------

def encode_png(width: int, height: int, rgb: bytes) -> bytes:
    """An 8-bit RGB PNG from raw pixel bytes (width*height*3)."""
    if len(rgb) != width * height * 3:
        raise ValueError("pixel data doesn't match the size")
    rows = b"".join(b"\x00" + rgb[y * width * 3:(y + 1) * width * 3] for y in range(height))

    def chunk(tag: bytes, body: bytes) -> bytes:
        return struct.pack(">I", len(body)) + tag + body + struct.pack(">I", zlib.crc32(tag + body) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(rows, 6)) + chunk(b"IEND", b""))


def solid_png(width: int, height: int, color: tuple[int, int, int] = (255, 255, 255),
              boxes: list[tuple[int, int, int, int, tuple[int, int, int]]] | None = None) -> bytes:
    """A test/simulation image: a background with filled rectangles (x, y, w, h, colour)."""
    pixels = bytearray(bytes(color) * (width * height))
    for x0, y0, w, h, c in boxes or []:
        for y in range(max(0, y0), min(height, y0 + h)):
            start = (y * width + max(0, x0)) * 3
            end = (y * width + min(width, x0 + w)) * 3
            pixels[start:end] = bytes(c) * ((end - start) // 3)
    return encode_png(width, height, bytes(pixels))


def decode_png(data: bytes) -> tuple[int, int, bytes]:
    """(width, height, RGB bytes) for 8-bit non-interlaced PNGs (grey, RGB, palette, with or without alpha)."""
    if sniff(data) != "png":
        raise ImageError("not a PNG")
    pos, idat, palette = 8, [], b""
    width = height = depth = ctype = interlace = 0
    while pos < len(data):
        length = struct.unpack(">I", data[pos:pos + 4])[0]
        tag = data[pos + 4:pos + 8]
        body = data[pos + 8:pos + 8 + length]
        if tag == b"IHDR":
            width, height, depth, ctype, _c, _f, interlace = struct.unpack(">IIBBBBB", body)
        elif tag == b"PLTE":
            palette = body
        elif tag == b"IDAT":
            idat.append(body)
        elif tag == b"IEND":
            break
        pos += 12 + length
    if depth != 8 or interlace:
        raise ImageError("only 8-bit, non-interlaced PNGs can be decoded without Pillow")
    channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}[ctype]
    raw = zlib.decompress(b"".join(idat))
    stride = width * channels
    if len(raw) < height * (stride + 1):
        raise ImageError("the image data is incomplete")
    low7, high = int.from_bytes(b"\x7f" * stride, "big"), int.from_bytes(b"\x80" * stride, "big")
    out = bytearray()
    prev = bytearray(stride)
    i = 0
    for _ in range(height):
        f = raw[i]
        line = bytearray(raw[i + 1:i + 1 + stride])
        i += 1 + stride
        if f == 1:                              # Sub: a running sum along each channel
            for ch in range(channels):
                line[ch::channels] = bytes(map((255).__and__, accumulate(line[ch::channels])))
        elif f == 2:                            # Up: bytewise addition of the row above, all at once
            a, b = int.from_bytes(line, "big"), int.from_bytes(prev, "big")
            line = bytearray((((a & low7) + (b & low7)) ^ ((a ^ b) & high)).to_bytes(stride, "big"))
        elif f == 3:                            # Average
            for x in range(stride):
                left = line[x - channels] if x >= channels else 0
                line[x] = (line[x] + ((left + prev[x]) >> 1)) & 0xFF
        elif f == 4:                            # Paeth
            for x in range(stride):
                a = line[x - channels] if x >= channels else 0
                b = prev[x]
                c = prev[x - channels] if x >= channels else 0
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                line[x] = (line[x] + (a if pa <= pb and pa <= pc else b if pb <= pc else c)) & 0xFF
        elif f != 0:
            raise ImageError(f"unknown PNG row filter {f}")
        prev = line
        out += _to_rgb(line, ctype, width, palette)
    return width, height, bytes(out)


def _to_rgb(line: bytearray, ctype: int, width: int, palette: bytes) -> bytes:
    """One decoded PNG row as RGB bytes (grey, palette and alpha converted), using slices rather than a pixel loop."""
    if ctype == 2:
        return bytes(line)
    rgb = bytearray(width * 3)
    if ctype == 6:
        rgb[0::3], rgb[1::3], rgb[2::3] = line[0::4], line[1::4], line[2::4]
    elif ctype in (0, 4):
        grey = line[0::2] if ctype == 4 else line
        rgb[0::3] = rgb[1::3] = rgb[2::3] = grey
    else:
        pal = palette.ljust(768, b"\x00")
        idx = bytes(line)
        rgb[0::3] = idx.translate(pal[0::3])
        rgb[1::3] = idx.translate(pal[1::3])
        rgb[2::3] = idx.translate(pal[2::3])
    return bytes(rgb)


def rgb_pixels(data: bytes) -> tuple[int, int, bytes] | None:
    """Decoded pixels for comparison: Pillow for any format if present, else pure-Python PNG; None if neither."""
    if pillow_available():
        try:
            from PIL import Image
            with Image.open(io.BytesIO(data)) as img:
                rgb = img.convert("RGB")
                return rgb.width, rgb.height, rgb.tobytes()
        except Exception:
            return None
    try:
        return decode_png(data)
    except (ImageError, zlib.error, struct.error, KeyError, ValueError, IndexError):
        return None


@dataclass
class PixelDiff:
    comparable: bool
    changed_fraction: float = 0.0          # share of the picture that differs (of the pixels compared)
    regions: list[str] = None              # e.g. ["top left", "bottom centre"]
    note: str = ""
    exact: bool = False                    # every pixel was compared (otherwise an even sample across the picture)
    identical: bool = False                # the very same image data

    def describe(self) -> str:
        if not self.comparable:
            return self.note or "the images couldn't be compared pixel by pixel"
        if self.identical:
            return "the images are identical"
        if self.changed_fraction == 0:
            return ("no pixel differences (every pixel compared)" if self.exact else
                    "no pixel differences at the points compared across the picture (a very small change could be "
                    "missed)")
        share = "less than 1%" if self.changed_fraction < 0.01 else f"{self.changed_fraction:.0%}"
        where = ", ".join(self.regions or []) or "scattered areas"
        return f"{'' if self.exact else 'about '}{share} of the picture changed ({where})"


def pixel_diff(a: bytes, b: bytes, *, grid: int = 6, threshold: int = 48, samples: int = 24) -> PixelDiff:
    """Which parts of two images differ, measured (not guessed).

    Pixels are compared one by one (a pixel counts as changed when its colour moved by more than ``threshold``,
    summed over red, green and blue, so compression noise doesn't count). Small images are compared pixel for pixel;
    larger ones at an even grid of points (``samples`` per side of each of the grid's cells), which is said."""
    if a == b:
        return PixelDiff(True, 0.0, [], exact=True, identical=True)
    pa, pb = rgb_pixels(a), rgb_pixels(b)
    if pa is None or pb is None:
        return PixelDiff(False, note="pixel comparison needs PNG images or the Pillow package")
    (wa, ha, ra), (wb, hb, rb) = pa, pb
    if (wa, ha) != (wb, hb):
        return PixelDiff(False, note=f"the images have different sizes ({wa}×{ha} and {wb}×{hb}), so only their "
                                     "content can be compared, not their pixels")
    changed_cells: list[tuple[int, int]] = []
    changed_total = compared_total = 0
    exact = True
    for gy in range(grid):
        for gx in range(grid):
            x0, x1 = gx * wa // grid, max(gx * wa // grid + 1, (gx + 1) * wa // grid)
            y0, y1 = gy * ha // grid, max(gy * ha // grid + 1, (gy + 1) * ha // grid)
            x1, y1 = min(x1, wa), min(y1, ha)
            sx, sy = max(1, (x1 - x0) // samples), max(1, (y1 - y0) // samples)
            exact = exact and sx == 1 and sy == 1
            changed = compared = 0
            for y in range(y0, y1, sy):
                for x in range(x0, x1, sx):
                    i = (y * wa + x) * 3
                    compared += 1
                    if abs(ra[i] - rb[i]) + abs(ra[i + 1] - rb[i + 1]) + abs(ra[i + 2] - rb[i + 2]) > threshold:
                        changed += 1
            changed_total += changed
            compared_total += compared
            if compared and changed / compared > 0.02:
                changed_cells.append((gx, gy))
    names = set()
    for gx, gy in changed_cells:
        v = "top" if gy < grid / 3 else "bottom" if gy >= 2 * grid / 3 else "middle"
        h = "left" if gx < grid / 3 else "right" if gx >= 2 * grid / 3 else "centre"
        names.add(f"{v} {h}" if not (v == "middle" and h == "centre") else "centre")
    order = ["top left", "top centre", "top right", "middle left", "centre", "middle right", "bottom left",
             "bottom centre", "bottom right"]
    fraction = changed_total / compared_total if compared_total else 0.0
    return PixelDiff(True, fraction, [n for n in order if n in names], exact=exact)
