"""Turns an arbitrary hand-drawn digit image into model-ready input.

Mirrors the format of scikit-learn's bundled digits dataset: an 8x8
grayscale image where brighter pixels mean more "ink" (as opposed to a
typical canvas drawing, which is usually dark ink on a light background).
This is a lightweight approximation, not full MNIST-grade preprocessing --
good enough for a small 8x8-trained model, not a state-of-the-art pipeline.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from digitrecognizer.data import IMAGE_SIZE, PIXEL_MAX_VALUE

# A pixel brighter than this (out of 255) counts as "background", used both
# to decide whether to invert an image and to find the ink bounding box.
_BRIGHT_BACKGROUND_THRESHOLD = 127


def decode_data_url(data_url: str) -> Image.Image:
    """Decodes a `data:image/...;base64,...` URL into a PIL image.

    Args:
        data_url: A data URL, such as the string an HTML canvas's
            `toDataURL()` method produces.

    Returns:
        The decoded image.

    Raises:
        ValueError: If `data_url` has no comma separating the header from
            the base64 payload (i.e. it is not a well-formed data URL).
    """
    if "," not in data_url:
        raise ValueError("Expected a data URL of the form 'data:...;base64,<data>'")
    _, encoded = data_url.split(",", 1)
    return Image.open(io.BytesIO(base64.b64decode(encoded)))


def load_image(path: str | Path) -> Image.Image:
    """Loads an image from a file path.

    Args:
        path: Path to an image file, in any format Pillow can read.

    Returns:
        The loaded image.
    """
    return Image.open(path)


def _autocrop_to_content(grayscale: Image.Image) -> Image.Image:
    """Crops a bright-ink-on-dark-background image to its content.

    Finds the bounding box of non-background pixels and crops to a square
    region around it with some margin, so a small digit drawn off-center
    on a large canvas still ends up reasonably centered and appropriately
    scaled after the later resize to 8x8 -- similar to how the digits
    dataset's own samples are already centered.

    Args:
        grayscale: A grayscale image where brighter pixels are ink.

    Returns:
        A square crop of `grayscale` around its ink content, or the
        original image unchanged if no content is found (e.g. a blank
        canvas).
    """
    array = np.asarray(grayscale)
    threshold = max(10, int(array.max() * 0.1))
    ink_rows = np.where(array.max(axis=1) > threshold)[0]
    ink_cols = np.where(array.max(axis=0) > threshold)[0]
    if ink_rows.size == 0 or ink_cols.size == 0:
        return grayscale

    top, bottom = int(ink_rows[0]), int(ink_rows[-1])
    left, right = int(ink_cols[0]), int(ink_cols[-1])

    height, width = bottom - top + 1, right - left + 1
    side = int(max(height, width) * 1.4)
    center_y, center_x = (top + bottom) // 2, (left + right) // 2
    half = side // 2

    crop_top = max(0, center_y - half)
    crop_left = max(0, center_x - half)
    crop_bottom = min(array.shape[0], center_y + half)
    crop_right = min(array.shape[1], center_x + half)

    return grayscale.crop((crop_left, crop_top, crop_right, crop_bottom))


def image_to_feature_vector(image: Image.Image) -> np.ndarray:
    """Converts an arbitrary digit image into a model-ready feature vector.

    Args:
        image: A PIL image containing a single hand-drawn digit. Any mode
            (RGB, RGBA, grayscale) and any size are accepted; color and
            alpha information is discarded.

    Returns:
        A 1D array of 64 floats (a flattened 8x8 image) with values in
        `[0, PIXEL_MAX_VALUE]`, matching the format expected by a model
        trained on `digitrecognizer.data.load_digit_dataset`.
    """
    grayscale = image.convert("L")

    # Canvas drawings are usually dark ink on a light background -- the
    # opposite of the digits dataset's bright-ink-on-dark convention -- so
    # invert when the image looks light-background overall.
    if np.asarray(grayscale).mean() > _BRIGHT_BACKGROUND_THRESHOLD:
        grayscale = ImageOps.invert(grayscale)

    cropped = _autocrop_to_content(grayscale)
    resized = cropped.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
    pixels = np.asarray(resized, dtype=np.float64)

    pixel_min, pixel_max = pixels.min(), pixels.max()
    if pixel_max > pixel_min:
        pixels = (pixels - pixel_min) / (pixel_max - pixel_min) * PIXEL_MAX_VALUE
    else:
        pixels = np.zeros_like(pixels)

    return pixels.flatten()
