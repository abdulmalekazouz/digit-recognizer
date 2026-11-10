"""Tests for digitrecognizer.preprocessing."""

import base64
import io

import numpy as np
import pytest
from PIL import Image
from sklearn.datasets import load_digits

from digitrecognizer.data import PIXEL_MAX_VALUE, load_digit_dataset
from digitrecognizer.model import train_model
from digitrecognizer.preprocessing import decode_data_url, image_to_feature_vector, load_image


def _make_png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_image_to_feature_vector_has_expected_shape_and_range():
    image = Image.new("L", (280, 280), color=255)
    vector = image_to_feature_vector(image)
    assert vector.shape == (64,)
    assert vector.min() >= 0
    assert vector.max() <= PIXEL_MAX_VALUE


def test_blank_canvas_does_not_crash_and_returns_all_background():
    # An all-white (light-background) canvas has no "ink" at all.
    image = Image.new("L", (280, 280), color=255)
    vector = image_to_feature_vector(image)
    assert np.allclose(vector, 0.0)


def test_light_and_dark_background_variants_produce_similar_output():
    # Draw a simple vertical bar: once as dark ink on a light background
    # (typical of an HTML canvas), once pre-inverted (bright ink on a dark
    # background, the digits-dataset convention). Preprocessing should
    # auto-detect and normalize both to roughly the same feature vector.
    light_bg = Image.new("L", (100, 100), color=255)
    for x in range(45, 55):
        for y in range(10, 90):
            light_bg.putpixel((x, y), 0)

    dark_bg = Image.new("L", (100, 100), color=0)
    for x in range(45, 55):
        for y in range(10, 90):
            dark_bg.putpixel((x, y), 255)

    vector_from_light = image_to_feature_vector(light_bg)
    vector_from_dark = image_to_feature_vector(dark_bg)
    assert np.allclose(vector_from_light, vector_from_dark, atol=1e-6)


def test_decode_data_url_round_trips_an_image():
    original = Image.new("RGB", (10, 10), color=(255, 0, 0))
    encoded = base64.b64encode(_make_png_bytes(original)).decode("ascii")
    data_url = f"data:image/png;base64,{encoded}"

    decoded = decode_data_url(data_url)
    assert decoded.size == (10, 10)


def test_decode_data_url_rejects_malformed_input():
    with pytest.raises(ValueError):
        decode_data_url("not-a-data-url")


def test_load_image_reads_a_file(tmp_path):
    original = Image.new("L", (20, 20), color=128)
    path = tmp_path / "digit.png"
    original.save(path)

    loaded = load_image(path)
    assert loaded.size == (20, 20)


def test_end_to_end_pipeline_predicts_known_digit_correctly():
    # Use scikit-learn's own un-flattened 8x8 digit images (already in the
    # bright-ink-on-dark-background convention) as a stand-in for a
    # "canvas" drawing, upscaled the way a real canvas drawing would be.
    sklearn_digits = load_digits()
    sample_index = 0
    expected_label = int(sklearn_digits.target[sample_index])

    small_image = Image.fromarray(sklearn_digits.images[sample_index].astype("uint8"))
    upscaled = small_image.resize((280, 280), Image.NEAREST)

    model = train_model(load_digit_dataset())
    feature_vector = image_to_feature_vector(upscaled)
    prediction = model.predict(feature_vector.reshape(1, -1))[0]

    assert int(prediction) == expected_label
