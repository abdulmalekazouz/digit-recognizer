"""A small Flask front end over the digitrecognizer library.

Serves a page with an HTML canvas for drawing a digit, and a `/predict`
endpoint that turns the drawing into a prediction with a confidence score.
"""

from __future__ import annotations

from flask import Flask, jsonify, render_template, request

from digitrecognizer.model import get_or_train_model
from digitrecognizer.preprocessing import decode_data_url, image_to_feature_vector


def create_app() -> Flask:
    """Builds and configures the digitrecognizer Flask application.

    The classifier is loaded lazily on the first prediction request (and
    trained from scratch the very first time, since trained model files
    are not checked into version control), then reused for subsequent
    requests via `digitrecognizer.model.get_or_train_model`'s on-disk
    cache.

    Returns:
        A Flask app with `/` (the canvas page) and `/predict` (the POST
        JSON endpoint) routes registered.
    """
    app = Flask(__name__)

    @app.get("/")
    def index():
        """Renders the digit-drawing canvas page."""
        return render_template("index.html")

    @app.post("/predict")
    def predict():
        """Predicts the digit drawn on the canvas.

        Expects a JSON body `{"image": "data:image/png;base64,..."}` as
        produced by calling `toDataURL()` on the page's canvas element.

        Returns:
            A JSON response `{"digit": int, "confidence": float}` on
            success, or `{"error": str}` with a 400 status if the request
            body is missing or malformed.
        """
        payload = request.get_json(silent=True) or {}
        data_url = payload.get("image")
        if not data_url:
            return jsonify({"error": "Missing 'image' field in request body"}), 400

        try:
            image = decode_data_url(data_url)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

        model = get_or_train_model()
        feature_vector = image_to_feature_vector(image).reshape(1, -1)
        digit = int(model.predict(feature_vector)[0])
        confidence = float(model.predict_proba(feature_vector).max())

        return jsonify({"digit": digit, "confidence": confidence})

    return app


app = create_app()

if __name__ == "__main__":
    app.run(debug=True)
