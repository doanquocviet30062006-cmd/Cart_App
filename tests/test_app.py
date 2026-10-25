import json
import threading
import unittest
from http.server import ThreadingHTTPServer
from urllib.request import Request, urlopen
from urllib.error import HTTPError

import pandas as pd

from app.server import Handler, META, METRICS, MODEL, predict_payload, validate_payload


class CartAppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.thread.join(timeout=3)
        cls.httpd.server_close()

    def payload(self):
        return dict(META["example"])

    def test_metrics_artifact_contains_real_heldout_evaluation(self):
        self.assertGreater(METRICS["data_counts"]["eligible_enrollments"], 0)
        self.assertEqual(METRICS["split"]["student_overlap"], 0)
        self.assertEqual(sum(map(sum, METRICS["pruned_tree_test"]["confusion_matrix"])), METRICS["split"]["test_enrollments"])

    def test_model_features_exclude_outcome_and_postcutoff_dates(self):
        forbidden = {"final_result", "date_unregistration", "id_student", "date_registration"}
        self.assertTrue(forbidden.isdisjoint(META["feature_order"]))

    def test_prediction_returns_supported_label_and_score(self):
        result = predict_payload(self.payload())
        self.assertIn(result["prediction"], {"Withdrawn", "Not withdrawn"})
        self.assertGreaterEqual(result["leaf_withdrawn_fraction"], 0.0)
        self.assertLessEqual(result["leaf_withdrawn_fraction"], 1.0)

    def test_pipeline_and_prediction_function_match(self):
        p = self.payload()
        direct = MODEL.predict(pd.DataFrame([p], columns=META["feature_order"]))[0]
        result = predict_payload(p)
        self.assertEqual(result["prediction"], "Withdrawn" if int(direct) else "Not withdrawn")

    def test_decision_path_ends_at_same_leaf_prediction(self):
        result = predict_payload(self.payload())
        leaf = result["decision_path"][-1]
        self.assertTrue(leaf["leaf"])
        self.assertEqual(leaf["prediction"], result["prediction"])

    def test_missing_feature_rejected(self):
        p = self.payload(); p.pop(META["feature_order"][0])
        with self.assertRaisesRegex(ValueError, "Missing fields"):
            validate_payload(p)

    def test_extra_feature_rejected(self):
        p = self.payload(); p["final_result"] = "Withdrawn"
        with self.assertRaisesRegex(ValueError, "Unexpected fields"):
            validate_payload(p)

    def test_wrong_numeric_type_rejected(self):
        p = self.payload(); p["clicks_day_0_28"] = "many"
        with self.assertRaisesRegex(ValueError, "finite number"):
            validate_payload(p)

    def test_out_of_training_range_rejected(self):
        p = self.payload(); p["clicks_day_0_28"] = META["numeric_ranges"]["clicks_day_0_28"]["max"] + 1
        with self.assertRaisesRegex(ValueError, "observed training range"):
            validate_payload(p)

    def test_unknown_category_rejected(self):
        p = self.payload(); p["code_module"] = "NOT_A_MODULE"
        with self.assertRaisesRegex(ValueError, "must be one of"):
            validate_payload(p)

    def test_integer_feature_requires_whole_number(self):
        p = self.payload(); p["num_of_prev_attempts"] = 1.5
        with self.assertRaisesRegex(ValueError, "whole number"):
            validate_payload(p)

    def test_health_and_schema_api(self):
        with urlopen(self.base + "/api/health") as r:
            self.assertEqual(json.loads(r.read())["status"], "ok")
        with urlopen(self.base + "/api/schema") as r:
            schema = json.loads(r.read())
        self.assertEqual(schema["feature_order"], META["feature_order"])

    def test_api_rejects_missing_field(self):
        p = self.payload(); p.pop(META["feature_order"][0])
        req = Request(self.base + "/api/predict", data=json.dumps(p).encode(), headers={"Content-Type":"application/json"}, method="POST")
        with self.assertRaises(HTTPError) as caught:
            urlopen(req)
        self.assertEqual(caught.exception.code, 422)

    def test_api_returns_prediction_and_path(self):
        req = Request(self.base + "/api/predict", data=json.dumps(self.payload()).encode(), headers={"Content-Type":"application/json"}, method="POST")
        with urlopen(req) as r:
            result=json.loads(r.read())
        self.assertIn("decision_path", result)
        self.assertTrue(result["decision_path"][-1]["leaf"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
