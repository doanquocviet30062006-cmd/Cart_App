from __future__ import annotations

import json
import os
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
STATIC = ROOT / "app" / "static"
MODEL = joblib.load(ART / "cart_withdrawal.joblib")
META = json.loads((ART / "model_metadata.json").read_text(encoding="utf-8"))
METRICS = json.loads((ART / "metrics.json").read_text(encoding="utf-8"))
TREE = MODEL.named_steps["tree"].tree_
FEATURE_NAMES = MODEL.named_steps["preprocess"].get_feature_names_out().tolist()


def validate_payload(payload: object) -> dict:
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object.")
    expected = META["feature_order"]
    missing = [key for key in expected if key not in payload]
    extra = [key for key in payload if key not in expected]
    if missing or extra:
        parts = []
        if missing: parts.append("Missing fields: " + ", ".join(missing))
        if extra: parts.append("Unexpected fields: " + ", ".join(extra))
        raise ValueError(". ".join(parts))
    row = {}
    for key in META["categorical_features"]:
        value = payload[key]
        if not isinstance(value, str) or value not in META["categories"][key]:
            raise ValueError(f"{key} must be one of: {', '.join(META['categories'][key])}.")
        row[key] = value
    for key in META["numeric_features"]:
        value = payload[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
            raise ValueError(f"{key} must be a finite number.")
        limits = META["numeric_ranges"][key]
        if value < limits["min"] or value > limits["max"]:
            raise ValueError(f"{key} must be within the observed training range [{limits['min']:g}, {limits['max']:g}].")
        if limits["integer"] and int(value) != value:
            raise ValueError(f"{key} must be a whole number.")
        row[key] = int(value) if limits["integer"] else float(value)
    if row["active_days_day_0_28"] > 29:
        raise ValueError("active_days_day_0_28 cannot exceed 29 days.")
    return row


def human_feature(encoded_name: str) -> str:
    prefix = "categorical__"
    if encoded_name.startswith(prefix):
        raw = encoded_name[len(prefix):]
        for col in META["categorical_features"]:
            marker = col + "_"
            if raw.startswith(marker):
                return f"one-hot({col} = {raw[len(marker):]})"
    return encoded_name.removeprefix("numeric__")


def explain(row: dict) -> list[dict]:
    frame = pd.DataFrame([row], columns=META["feature_order"])
    transformed = MODEL.named_steps["preprocess"].transform(frame)
    path = MODEL.named_steps["tree"].decision_path(transformed).indices.tolist()
    steps = []
    for node in path:
        left, right = TREE.children_left[node], TREE.children_right[node]
        if left == right:
            counts = TREE.value[node][0].tolist()
            total = float(sum(counts)) or 1.0
            fractions = [float(value) / total for value in counts]
            leaf_pred = int(np.argmax(counts))
            steps.append({"node": int(node), "leaf": True, "prediction": "Withdrawn" if leaf_pred else "Not withdrawn",
                          "samples": int(TREE.n_node_samples[node]), "class_fraction": {"not_withdrawn": fractions[0], "withdrawn": fractions[1]}})
            continue
        feat_index = int(TREE.feature[node])
        value = float(transformed[0, feat_index])
        threshold = float(TREE.threshold[node])
        went_left = value <= threshold
        steps.append({"node": int(node), "leaf": False, "feature": human_feature(FEATURE_NAMES[feat_index]),
                      "encoded_feature": FEATURE_NAMES[feat_index], "threshold": threshold,
                      "sample_value": value, "condition": f"{value:.4g} <= {threshold:.4g}",
                      "condition_result": bool(went_left), "branch": "left" if went_left else "right",
                      "next_node": int(left if went_left else right)})
    return steps


def predict_payload(payload: object) -> dict:
    row = validate_payload(payload)
    frame = pd.DataFrame([row], columns=META["feature_order"])
    predicted = int(MODEL.predict(frame)[0])
    proba = MODEL.predict_proba(frame)[0]
    # sklearn classes are [0, 1]; this is the terminal-leaf fraction, not calibrated.
    risk_score = float(proba[list(MODEL.classes_).index(1)])
    return {"prediction": "Withdrawn" if predicted else "Not withdrawn",
            "meaning": "eventual final_result = Withdrawn" if predicted else "eventual final_result is Pass, Fail, or Distinction",
            "leaf_withdrawn_fraction": risk_score, "score_note": "Class fraction in this tree leaf; not a calibrated probability.",
            "model_alpha": METRICS["pruned_tree"]["ccp_alpha"], "model_version": "oulad-cart-withdrawal-day28-v1",
            "decision_path": explain(row), "support_only": True}


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC), **kwargs)

    def log_message(self, fmt, *args):
        pass

    def _json(self, status: int, value: dict):
        body = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/schema":
            self._json(200, {"feature_order": META["feature_order"], "categories": META["categories"], "numeric_ranges": META["numeric_ranges"], "example": META["example"]})
        elif path == "/api/metrics":
            self._json(200, METRICS)
        elif path == "/api/health":
            self._json(200, {"status": "ok", "model_version": "oulad-cart-withdrawal-day28-v1"})
        else:
            if path == "/": self.path = "/index.html"
            return super().do_GET()

    def do_POST(self):
        if urlparse(self.path).path != "/api/predict":
            return self._json(404, {"error": "Route not found."})
        try:
            n = int(self.headers.get("Content-Length", "0"))
            if n <= 0 or n > 20_000:
                raise ValueError("Request body is empty or too large.")
            payload = json.loads(self.rfile.read(n))
            self._json(200, predict_payload(payload))
        except (ValueError, json.JSONDecodeError) as exc:
            self._json(422, {"error": str(exc)})


def main():
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8765"))
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"CART student withdrawal demo at http://{host}:{port}", flush=True)
    server.serve_forever()

if __name__ == "__main__":
    main()
