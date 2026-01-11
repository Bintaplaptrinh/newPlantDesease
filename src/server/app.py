from __future__ import annotations

import base64
import io
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image

from src.inference import load_torch_model, predict_image, saliency_heatmap_png
from src.server.config import ServerConfig, detect_device
from src.utils.json_io import read_json


def create_app(config: Optional[ServerConfig] = None) -> Flask:
    cfg = config or ServerConfig.default()

    app = Flask(__name__)
    CORS(app, resources={r"/api/*": {"origins": "*"}})

    @lru_cache(maxsize=1)
    def _classes_payload() -> Dict[str, Any]:
        classes_path = cfg.data_web_dir / "classes.json"
        if classes_path.exists():
            return read_json(classes_path)
        # fallback: empty
        return {"labels": [], "shortLabels": [], "numClasses": 0}

    @lru_cache(maxsize=8)
    def _load_model_cached(model_id: str):
        device = detect_device()
        if model_id not in cfg.models:
            raise KeyError(model_id)
        pth_path = cfg.models[model_id]
        if not pth_path.exists():
            raise FileNotFoundError(str(pth_path))
        model = load_torch_model(pth_path, device=device)
        return model, device

    def _read_web_json(filename: str):
        p = (cfg.data_web_dir / filename)
        if not p.exists():
            return None
        return read_json(p)

    def _read_model_json(model_id: str, filename: str):
        p = cfg.data_web_dir / "models" / model_id / filename
        if not p.exists():
            return None
        return read_json(p)

    @app.get("/api/health")
    def health():
        return jsonify({"status": "ok"})

    @app.get("/api/data/classes")
    def classes():
        return jsonify(_classes_payload())

    @app.get("/api/data/dataset-stats")
    def dataset_stats():
        payload = _read_web_json("dataset_stats.json")
        return jsonify(payload or {"totalImages": 0, "trainImages": 0, "validationImages": 0, "testImages": 0, "numClasses": 0})

    @app.get("/api/data/class-distribution")
    def class_distribution():
        payload = _read_web_json("class_distribution.json")
        return jsonify(payload or {"items": [], "totalImages": 0, "numClasses": 0})

    @app.get("/api/data/models")
    def models_index():
        payload = _read_web_json("models.json")
        return jsonify(payload or {"models": []})

    @app.get("/api/data/roc-micro")
    def roc_micro():
        payload = _read_web_json("roc_micro.json")
        return jsonify(payload or {"points": [], "aucs": {}})

    @app.get("/api/data/models/<model_id>/confusion-matrix")
    def model_confusion(model_id: str):
        payload = _read_model_json(model_id, "confusion_matrix.json")
        return jsonify(payload or {"modelId": model_id, "labels": [], "matrix": [], "normalized": False})

    @app.get("/api/data/models/<model_id>/training-history")
    def model_training_history(model_id: str):
        payload = _read_model_json(model_id, "training_history.json")
        return jsonify(payload or {"modelId": model_id, "history": []})

    @app.post("/api/predict")
    def predict():
        model_id = request.args.get("model", "resnet18")
        top_k = int(request.args.get("top_k", "5"))

        if "file" not in request.files:
            return jsonify({"error": "missing file field"}), 400

        f = request.files["file"]
        if not f.filename:
            return jsonify({"error": "empty filename"}), 400

        try:
            image_bytes = f.read()
            img = Image.open(io.BytesIO(image_bytes))
        except Exception:
            return jsonify({"error": "invalid image"}), 400

        classes = _classes_payload().get("labels") or []

        try:
            model, device = _load_model_cached(model_id)
        except FileNotFoundError:
            return jsonify({"error": f"model file not found for '{model_id}'"}), 404
        except KeyError:
            return jsonify({"error": f"unknown model '{model_id}'"}), 404

        if not classes:
            # Allow inference even when classes.json missing
            # (will return class_<idx> labels)
            classes = []

        label, confidence, top = predict_image(
            model,
            img,
            classes,
            image_size=cfg.default_image_size,
            top_k=top_k,
            device=device,
        )

        return jsonify(
            {
                "modelId": model_id,
                "label": label,
                "confidence": confidence,
                "topK": top,
            }
        )

    @app.post("/api/explain")
    def explain():
        model_id = request.args.get("model", "resnet18")
        method = request.args.get("method", "saliency")

        if method not in {"saliency"}:
            return jsonify({"error": f"unsupported method '{method}'"}), 400

        if "file" not in request.files:
            return jsonify({"error": "missing file field"}), 400

        f = request.files["file"]
        if not f.filename:
            return jsonify({"error": "empty filename"}), 400

        try:
            image_bytes = f.read()
            img = Image.open(io.BytesIO(image_bytes))
        except Exception:
            return jsonify({"error": "invalid image"}), 400

        classes = _classes_payload().get("labels") or []

        try:
            model, device = _load_model_cached(model_id)
        except FileNotFoundError:
            return jsonify({"error": f"model file not found for '{model_id}'"}), 404
        except KeyError:
            return jsonify({"error": f"unknown model '{model_id}'"}), 404

        try:
            target_label, target_index, heat_png = saliency_heatmap_png(
                model,
                img,
                classes,
                image_size=cfg.default_image_size,
                device=device,
            )
        except Exception as e:
            return jsonify({"error": f"explain failed: {type(e).__name__}: {e}"}), 500

        w, h = img.size
        return jsonify(
            {
                "modelId": model_id,
                "method": method,
                "target": {"index": target_index, "label": target_label},
                "image": {"width": w, "height": h},
                "heatmapPngBase64": base64.b64encode(heat_png).decode("ascii"),
            }
        )

    return app


def main():
    app = create_app()
    # default dev server
    app.run(host="0.0.0.0", port=5000, debug=True)


if __name__ == "__main__":
    main()
