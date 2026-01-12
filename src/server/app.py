from __future__ import annotations

# api server (flask) cho web demo
# cung cấp endpoints:
# - /api/predict: phân loại bệnh (có thể dùng yolo + tiền xử lí)
# - /api/explain: saliency heatmap
# - /api/data/*: metadata (classes, stats, models, ...)

import base64
import io
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image

from src.inference import load_torch_model, predict_image, predict_images, saliency_heatmap_png
from src.server.config import ServerConfig, detect_device
from src.utils.json_io import read_json

# import tuỳ chọn cho yolo và tiền xử lí (để server vẫn chạy được nếu thiếu dependency)
try:
    from src.server.yolo_detector import YOLODetector, get_yolo_detector
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

try:
    from src.server.image_preprocessing import ImagePreprocessor, create_preprocessor
    PREPROCESSING_AVAILABLE = True
except ImportError:
    PREPROCESSING_AVAILABLE = False



def create_app(config: Optional[ServerConfig] = None) -> Flask:
    # tạo flask app theo config (hoặc dùng config mặc định)
    cfg = config or ServerConfig.default()

    app = Flask(__name__)
    CORS(app, resources={r"/api/*": {"origins": "*"}})

    @lru_cache(maxsize=1)
    def _classes_payload() -> Dict[str, Any]:
        # cache classes.json để giảm IO
        classes_path = cfg.data_web_dir / "classes.json"
        if classes_path.exists():
            return read_json(classes_path)
        # fallback: rỗng
        return {"labels": [], "shortLabels": [], "numClasses": 0}

    @lru_cache(maxsize=8)
    def _load_model_cached(model_id: str):
        # cache model theo id để tránh load .pth nhiều lần
        device = detect_device()
        if model_id not in cfg.models:
            raise KeyError(model_id)
        pth_path = cfg.models[model_id]
        if not pth_path.exists():
            raise FileNotFoundError(str(pth_path))
        model = load_torch_model(pth_path, device=device)
        return model, device

    @lru_cache(maxsize=4)
    def _get_preprocessor_cached(
        grabcut_iterations: int,
        morph_kernel_size: int,
        use_green_mask: bool,
    ):
        # cache preprocessor vì khởi tạo cv2/grabcut tốn thời gian
        if not PREPROCESSING_AVAILABLE:
            raise RuntimeError("preprocessing not available")
        return create_preprocessor(
            grabcut_iterations=int(grabcut_iterations),
            morph_kernel_size=int(morph_kernel_size),
            use_green_mask=bool(use_green_mask),
        )

    def _arg_bool(name: str, default: bool = False) -> bool:
        # parse query param kiểu bool (?flag=true/false)
        return request.args.get(name, "true" if default else "false").lower() == "true"

    def _arg_int(name: str, default: int, *, min_value: int | None = None, max_value: int | None = None) -> int:
        # parse query param kiểu int, có clamp min/max
        try:
            v = int(request.args.get(name, str(default)))
        except Exception:
            v = int(default)
        if min_value is not None:
            v = max(int(min_value), v)
        if max_value is not None:
            v = min(int(max_value), v)
        return v

    def _arg_float(name: str, default: float, *, min_value: float | None = None, max_value: float | None = None) -> float:
        # parse query param kiểu float, có clamp min/max
        try:
            v = float(request.args.get(name, str(default)))
        except Exception:
            v = float(default)
        if min_value is not None:
            v = max(float(min_value), v)
        if max_value is not None:
            v = min(float(max_value), v)
        return v

    def _crop_with_padding(image: Image.Image, *, x1: int, y1: int, x2: int, y2: int, padding: int) -> Tuple[Image.Image, Dict[str, int]]:
        # crop ảnh theo bbox và thêm padding, đồng thời clamp theo kích thước ảnh
        w, h = image.size
        px1 = max(0, int(x1) - int(padding))
        py1 = max(0, int(y1) - int(padding))
        px2 = min(w, int(x2) + int(padding))
        py2 = min(h, int(y2) + int(padding))
        crop = image.crop((px1, py1, px2, py2))
        return crop, {"x1": px1, "y1": py1, "x2": px2, "y2": py2}

    def _read_web_json(filename: str):
        # đọc artifact json chung trong data/web
        p = (cfg.data_web_dir / filename)
        if not p.exists():
            return None
        return read_json(p)

    def _read_model_json(model_id: str, filename: str):
        # đọc artifact json theo model trong data/web/models/<model_id>/
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
        
        # options pipeline
        use_yolo = _arg_bool("use_yolo", default=False)
        use_preprocessing = _arg_bool("use_preprocessing", default=False)

        # tham số yolo (giá trị mặc định an toàn)
        yolo_conf = _arg_float("yolo_conf", 0.25, min_value=0.0, max_value=1.0)
        yolo_iou = _arg_float("yolo_iou", 0.45, min_value=0.0, max_value=1.0)
        yolo_padding = _arg_int("yolo_padding", 15, min_value=0, max_value=2000)
        yolo_max_detections = _arg_int("yolo_max_detections", 10, min_value=1, max_value=100)

        # rule unknown: nếu top1 prob < ngưỡng thì nhãn = unknown
        unknown_threshold = _arg_float(
            "unknown_threshold",
            0.10 if use_yolo else 0.0,
            min_value=0.0,
            max_value=1.0,
        )

        if "file" not in request.files:
            return jsonify({"error": "missing file field"}), 400

        f = request.files["file"]
        if not f.filename:
            return jsonify({"error": "empty filename"}), 400

        try:
            image_bytes = f.read()
            img = Image.open(io.BytesIO(image_bytes))
            if img.mode != "RGB":
                img = img.convert("RGB")
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
            # vẫn cho phép infer nếu thiếu classes.json (sẽ trả label dạng class_<idx>)
            classes = []

        # thông tin pipeline để frontend hiển thị
        pipeline_info = {
            "yolo_used": False,
            "preprocessing_used": False,
            "yolo_detection": None,
            "yolo_detections": [],
            "original_size": {"width": img.size[0], "height": img.size[1]},
            "mode": "yolo" if use_yolo else "full_image",
            "unknown_threshold": unknown_threshold,
        }

        # mode a: bật yolo -> detect tất cả lá, crop từng bbox, (tuỳ chọn) tiền xử lí từng crop, rồi phân loại theo batch
        if use_yolo:
            if not YOLO_AVAILABLE:
                pipeline_info["yolo_error"] = "YOLO module not available (missing ultralytics package)"
            elif not (cfg.yolo_model_path and cfg.yolo_model_path.exists()):
                pipeline_info["yolo_error"] = "YOLO model file not found"
            else:
                try:
                    yolo_detector = get_yolo_detector(str(cfg.yolo_model_path), device=device)
                    detections = yolo_detector.detect(img, conf_threshold=yolo_conf, iou_threshold=yolo_iou)
                    detections = detections[: int(yolo_max_detections)]

                    crops: List[Image.Image] = []
                    det_payloads: List[Dict[str, Any]] = []

                    for det in detections:
                        crop, padded_box = _crop_with_padding(
                            img,
                            x1=det.x1,
                            y1=det.y1,
                            x2=det.x2,
                            y2=det.y2,
                            padding=yolo_padding,
                        )
                        crops.append(crop)
                        det_payloads.append(
                            {
                                "box": padded_box,
                                "raw_box": {"x1": det.x1, "y1": det.y1, "x2": det.x2, "y2": det.y2},
                                "yolo_confidence": float(det.confidence),
                                "yolo_class_name": det.class_name,
                                "crop_size": {"width": crop.size[0], "height": crop.size[1]},
                            }
                        )

                    if det_payloads:
                        pipeline_info["yolo_used"] = True
                        pipeline_info["yolo_detection"] = det_payloads[0]
                        pipeline_info["yolo_detections"] = det_payloads

                        # phân loại theo batch (có thể so sánh ảnh gốc vs ảnh đã tiền xử lí và chọn kết quả tự tin hơn)
                        preds_original = predict_images(
                            model,
                            crops,
                            classes,
                            image_size=cfg.default_image_size,
                            top_k=top_k,
                            device=device,
                        )

                        preds_preprocessed: List[Tuple[str, float, List[Dict[str, float]]]] | None = None
                        if use_preprocessing:
                            pipeline_info["preprocessing_attempted"] = True
                            pipeline_info["preprocessing_strategy"] = "auto_best_confidence"
                            if not PREPROCESSING_AVAILABLE:
                                pipeline_info["preprocessing_error"] = "Preprocessing module not available (missing opencv-python package)"
                            else:
                                try:
                                    # giảm iterations thường ít phá ảnh hơn và chạy nhanh hơn
                                    preprocessor = _get_preprocessor_cached(3, 5, True)
                                    processed_crops = [preprocessor.segment_leaf(c) for c in crops]
                                    preds_preprocessed = predict_images(
                                        model,
                                        processed_crops,
                                        classes,
                                        image_size=cfg.default_image_size,
                                        top_k=top_k,
                                        device=device,
                                    )
                                except Exception as e:
                                    pipeline_info["preprocessing_error"] = str(e)

                        # chọn kết quả tốt nhất theo từng crop
                        batch_preds: List[Tuple[str, float, List[Dict[str, float]]]] = []
                        used_preprocessed_any = False
                        if preds_preprocessed is None:
                            batch_preds = preds_original
                        else:
                            for (lbl_o, conf_o, top_o), (lbl_p, conf_p, top_p) in zip(preds_original, preds_preprocessed):
                                if float(conf_p) > float(conf_o):
                                    batch_preds.append((lbl_p, float(conf_p), top_p))
                                    used_preprocessed_any = True
                                else:
                                    batch_preds.append((lbl_o, float(conf_o), top_o))
                        if used_preprocessed_any:
                            pipeline_info["preprocessing_used"] = True

                        detections_out: List[Dict[str, Any]] = []
                        for det_info, (lbl, conf, top) in zip(det_payloads, batch_preds):
                            is_unknown = bool(conf < float(unknown_threshold))
                            final_label = "unknown" if is_unknown else lbl
                            detections_out.append(
                                {
                                    **det_info,
                                    "label": final_label,
                                    "confidence": float(conf),
                                    "is_unknown": is_unknown,
                                    "topK": top,
                                }
                            )

                        # giữ tương thích ngược: label top-level là detection có confidence cao nhất
                        best_det = max(detections_out, key=lambda d: float(d.get("confidence", 0.0)))
                        return jsonify(
                            {
                                "modelId": model_id,
                                "label": best_det["label"],
                                "confidence": best_det["confidence"],
                                "topK": best_det["topK"],
                                "detections": detections_out,
                                "pipeline": pipeline_info,
                            }
                        )

                    # yolo chạy nhưng không detect được gì -> fallback về phân loại cả ảnh
                    pipeline_info["yolo_used"] = True
                    pipeline_info["yolo_detections"] = []
                    pipeline_info["yolo_warning"] = "no detections"
                except Exception as e:
                    pipeline_info["yolo_error"] = str(e)

        # mode b (hoặc fallback): phân loại cả ảnh, (tuỳ chọn) tiền xử lí rồi chọn kết quả tự tin hơn
        processed_img = img
        if use_preprocessing:
            pipeline_info["preprocessing_attempted"] = True
            pipeline_info["preprocessing_strategy"] = "auto_best_confidence"

        # luôn tính prediction trên ảnh gốc
        label_o, confidence_o, top_o = predict_image(
            model,
            processed_img,
            classes,
            image_size=cfg.default_image_size,
            top_k=top_k,
            device=device,
        )

        label, confidence, top = label_o, confidence_o, top_o

        if use_preprocessing:
            if not PREPROCESSING_AVAILABLE:
                pipeline_info["preprocessing_error"] = "Preprocessing module not available (missing opencv-python package)"
            else:
                try:
                    preprocessor = _get_preprocessor_cached(3, 5, True)
                    preprocessed_img = preprocessor.segment_leaf(processed_img)
                    label_p, confidence_p, top_p = predict_image(
                        model,
                        preprocessed_img,
                        classes,
                        image_size=cfg.default_image_size,
                        top_k=top_k,
                        device=device,
                    )
                    if float(confidence_p) > float(confidence_o):
                        label, confidence, top = label_p, confidence_p, top_p
                        pipeline_info["preprocessing_used"] = True
                except Exception as e:
                    pipeline_info["preprocessing_error"] = str(e)

        # ở chế độ không yolo, unknown_threshold mặc định = 0.0 nên giữ hành vi cũ
        if float(unknown_threshold) > 0.0 and float(confidence) < float(unknown_threshold):
            label = "unknown"

        return jsonify(
            {
                "modelId": model_id,
                "label": label,
                "confidence": confidence,
                "topK": top,
                "detections": None,
                "pipeline": pipeline_info,
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
