from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import cv2
from PIL import Image

from .errors import WSIReadError

# Optional pyvips
try:
    import pyvips  # type: ignore
    HAS_PYVIPS = True
except Exception:
    HAS_PYVIPS = False

try:
    import openslide  # type: ignore
    from openslide import OpenSlideError, OpenSlideUnsupportedFormatError  # type: ignore
    HAS_OPENSLIDE = True
except Exception:
    HAS_OPENSLIDE = False


Image.MAX_IMAGE_PIXELS = None


@dataclass
class WSIInfo:
    path: str
    backend: str
    width: int
    height: int
    level_count: int
    level_dims: Tuple[Tuple[int, int], ...]
    level_downsamples: Tuple[float, ...]
    mpp_x: Optional[float] = None
    mpp_y: Optional[float] = None


class WSIBackend:
    """
    Unified WSI reader wrapper.
    Supports:
      - OpenSlide (preferred; supports pyramid levels)
      - pyvips (fallback; single resolution behavior in this wrapper)
      - PIL (last resort; single resolution)
    """

    def __init__(self, path: str):
        self.path = str(path)
        p = Path(self.path)
        if not p.exists():
            raise WSIReadError(f"WSI not found: {self.path}")

        self.backend: str = ""
        self.slide_os = None
        self.slide_vips = None
        self.img_pil = None

        self.width = 0
        self.height = 0
        self.level_count = 1
        self.level_dims = ((0, 0),)
        self.level_downsamples = (1.0,)
        self.mpp_x = None
        self.mpp_y = None

        # 1) OpenSlide
        if HAS_OPENSLIDE:
            try:
                self.slide_os = openslide.OpenSlide(self.path)
                self.backend = "openslide"
                self.width, self.height = self.slide_os.dimensions
                self.level_count = int(self.slide_os.level_count)
                self.level_dims = tuple(tuple(map(int, d)) for d in self.slide_os.level_dimensions)
                self.level_downsamples = tuple(float(x) for x in self.slide_os.level_downsamples)
                self._try_read_mpp()
                return
            except OpenSlideUnsupportedFormatError:
                pass
            except OpenSlideError:
                pass
            except Exception:
                pass

        # 2) pyvips
        if HAS_PYVIPS:
            try:
                self.slide_vips = pyvips.Image.new_from_file(self.path, access="sequential")
                self.backend = "pyvips"
                self.width = int(self.slide_vips.width)
                self.height = int(self.slide_vips.height)
                self.level_count = 1
                self.level_dims = ((self.width, self.height),)
                self.level_downsamples = (1.0,)
                return
            except Exception:
                pass

        # 3) PIL
        try:
            self.img_pil = Image.open(self.path).convert("RGB")
            self.backend = "pil"
            self.width, self.height = self.img_pil.size
            self.level_count = 1
            self.level_dims = ((self.width, self.height),)
            self.level_downsamples = (1.0,)
            return
        except Exception as e:
            raise WSIReadError(
                f"Cannot open {self.path} with OpenSlide/pyvips/PIL. last_error={e}"
            )

    def _try_read_mpp(self) -> None:
        if self.backend != "openslide" or self.slide_os is None:
            return
        props = self.slide_os.properties
        mx = props.get("openslide.mpp-x", None)
        my = props.get("openslide.mpp-y", None)
        try:
            if mx is not None:
                self.mpp_x = float(mx)
            if my is not None:
                self.mpp_y = float(my)
        except Exception:
            self.mpp_x = None
            self.mpp_y = None

    def info(self) -> WSIInfo:
        return WSIInfo(
            path=self.path,
            backend=self.backend,
            width=self.width,
            height=self.height,
            level_count=self.level_count,
            level_dims=self.level_dims,
            level_downsamples=self.level_downsamples,
            mpp_x=self.mpp_x,
            mpp_y=self.mpp_y,
        )

    def get_level_dim(self, level: int) -> Tuple[int, int]:
        level = int(level)
        level = max(0, min(level, self.level_count - 1))
        return self.level_dims[level]

    def read_patch(self, x0: int, y0: int, size: int, level: int = 0) -> Image.Image:
        """
        Read RGB patch at specified pyramid level.
        Coordinates are in the chosen level coordinate system.

        Behavior:
          - OpenSlide: true level access
          - pyvips/PIL: only level=0 is supported; if level>0, we approximate by resizing
        """
        level = int(level)
        size = int(size)
        x0 = int(x0)
        y0 = int(y0)

        if self.backend == "openslide" and self.slide_os is not None:
            level = max(0, min(level, self.level_count - 1))
            Wl, Hl = self.level_dims[level]
            x1 = min(x0 + size, Wl)
            y1 = min(y0 + size, Hl)
            w = max(0, x1 - x0)
            h = max(0, y1 - y0)
            if w <= 0 or h <= 0:
                return Image.new("RGB", (size, size), (255, 255, 255))

            ds = float(self.level_downsamples[level])
            bx0 = int(round(x0 * ds))
            by0 = int(round(y0 * ds))
            patch = self.slide_os.read_region((bx0, by0), level, (w, h)).convert("RGB")

            if w < size or h < size:
                canvas = Image.new("RGB", (size, size), (255, 255, 255))
                canvas.paste(patch, (0, 0))
                return canvas
            return patch

        if level <= 0:
            return self._read_base_patch(x0, y0, size)

        ds = float(2 ** level)
        bx0 = int(round(x0 * ds))
        by0 = int(round(y0 * ds))
        base_patch = self._read_base_patch(bx0, by0, int(round(size * ds)))
        return base_patch.resize((size, size), Image.BILINEAR)

    def _read_base_patch(self, x0: int, y0: int, size: int) -> Image.Image:
        x0 = int(x0)
        y0 = int(y0)
        size = int(size)

        x1 = min(x0 + size, self.width)
        y1 = min(y0 + size, self.height)
        w = max(0, x1 - x0)
        h = max(0, y1 - y0)
        if w <= 0 or h <= 0:
            return Image.new("RGB", (size, size), (255, 255, 255))

        if self.backend == "pyvips" and self.slide_vips is not None:
            vpatch = self.slide_vips.crop(x0, y0, w, h)
            arr = np.ndarray(
                buffer=vpatch.write_to_memory(),
                dtype=np.uint8,
                shape=(vpatch.height, vpatch.width, vpatch.bands),
            )
            if arr.shape[2] > 3:
                arr = arr[:, :, :3]
            patch = Image.fromarray(arr, mode="RGB")

        elif self.backend == "pil" and self.img_pil is not None:
            patch = self.img_pil.crop((x0, y0, x1, y1))

        else:
            if self.backend == "openslide" and self.slide_os is not None:
                patch = self.slide_os.read_region((x0, y0), 0, (w, h)).convert("RGB")
            else:
                raise WSIReadError("Invalid backend state.")

        if w < size or h < size:
            canvas = Image.new("RGB", (size, size), (255, 255, 255))
            canvas.paste(patch, (0, 0))
            return canvas
        return patch

    def build_tissue_mask(self, max_thumb: int = 2048) -> Tuple[np.ndarray, float, Image.Image]:
        """
        Build a binary tissue mask at thumbnail resolution.

        Returns:
          tissue_mask: uint8 (0/1) on thumbnail grid
          base_to_mask_ds: base_pixels / mask_pixels (approx scale)
          thumb: RGB PIL image
        """
        if self.backend == "openslide" and self.slide_os is not None:
            w0, h0 = self.slide_os.dimensions
            scale = max(w0, h0) / float(max_thumb)
            scale = max(scale, 1.0)
            tw = int(w0 / scale)
            th = int(h0 / scale)
            thumb = self.slide_os.get_thumbnail((tw, th)).convert("RGB")
            ds = w0 / float(thumb.width)

        elif self.backend == "pyvips" and self.slide_vips is not None:
            scale = max(self.width, self.height) / float(max_thumb)
            scale = max(scale, 1.0)
            target = int(max(self.width / scale, self.height / scale))
            vthumb = self.slide_vips.thumbnail_image(target)
            h, w = vthumb.height, vthumb.width
            arr = np.ndarray(
                buffer=vthumb.write_to_memory(),
                dtype=np.uint8,
                shape=(h, w, vthumb.bands),
            )
            if arr.shape[2] > 3:
                arr = arr[:, :, :3]
            thumb = Image.fromarray(arr, mode="RGB")
            ds = self.width / float(w)

        else:
            img = self.img_pil
            if img is None:
                raise WSIReadError("PIL image not loaded.")
            scale = max(self.width, self.height) / float(max_thumb)
            scale = max(scale, 1.0)
            w = int(self.width / scale)
            h = int(self.height / scale)
            thumb = img.resize((w, h), Image.BILINEAR)
            ds = self.width / float(w)

        thumb_np = np.array(thumb)

        gray = cv2.cvtColor(thumb_np, cv2.COLOR_RGB2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

        T_otsu, _ = cv2.threshold(
            gray_blur, 0, 255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        delta = 15
        T_adj = max(int(T_otsu) - delta, 0)
        _, mask_gray = cv2.threshold(gray_blur, T_adj, 255, cv2.THRESH_BINARY_INV)

        hsv = cv2.cvtColor(thumb_np, cv2.COLOR_RGB2HSV)
        sat = hsv[:, :, 1]
        S_thresh = 20
        mask_sat = (sat > S_thresh).astype(np.uint8) * 255

        mask = cv2.bitwise_or(mask_gray, mask_sat)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  np.ones((3, 3), np.uint8), iterations=1)

        tissue = (mask > 0).astype(np.uint8)
        return tissue, float(ds), thumb
