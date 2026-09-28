"""Flask app: GRU inference API. Loads artifacts once at startup."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "torch")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from flask import Flask
from flask_cors import CORS

from backend.config import CORS_ORIGINS, HOST, MODEL_NAME, PORT
from backend.routes.traffic import traffic_bp
from backend.services.prediction_service import load_artifacts
from backend.utils.responses import ApiError, json_error

logger = logging.getLogger(__name__)


def create_app() -> Flask:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    app = Flask(__name__)
    CORS(app, origins=list(CORS_ORIGINS))
    app.register_blueprint(traffic_bp)

    @app.errorhandler(ApiError)
    def handle_api_error(exc: ApiError):
        return json_error(exc.message, exc.status_code)

    @app.errorhandler(404)
    def handle_404(_exc):
        return json_error("Not found.", 404)

    @app.errorhandler(500)
    def handle_500(exc):
        logger.exception("Unhandled server error: %s", exc)
        return json_error("Internal server error.", 500)

    with app.app_context():
        try:
            load_artifacts()
        except Exception:
            logger.exception("Failed to load GRU artifacts")
            raise

    logger.info("Flask API ready model=%s", MODEL_NAME)
    return app


app = create_app()


if __name__ == "__main__":
    app.run(host=HOST, port=PORT, debug=False)
