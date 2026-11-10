"""Tests for digitrecognizer.webapp."""

import base64
import io

import pytest
from PIL import Image
from sklearn.datasets import load_digits

from digitrecognizer.webapp import create_app


@pytest.fixture
def client():
    app = create_app()
    app.config.update(TESTING=True)
    return app.test_client()


def _digit_data_url(sample_index: int = 0) -> str:
    sklearn_digits = load_digits()
    image = Image.fromarray(sklearn_digits.images[sample_index].astype("uint8"))
    buffer = io.BytesIO()
    image.resize((280, 280)).save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def test_index_page_loads(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"Digit Recognizer" in response.data


def test_predict_happy_path_returns_digit_and_confidence(client):
    response = client.post("/predict", json={"image": _digit_data_url()})
    assert response.status_code == 200

    data = response.get_json()
    assert "digit" in data
    assert 0 <= data["digit"] <= 9
    assert 0.0 <= data["confidence"] <= 1.0


def test_predict_without_image_field_returns_400(client):
    response = client.post("/predict", json={})
    assert response.status_code == 400
    assert "error" in response.get_json()


def test_predict_with_malformed_data_url_returns_400(client):
    response = client.post("/predict", json={"image": "not-a-data-url"})
    assert response.status_code == 400
    assert "error" in response.get_json()
