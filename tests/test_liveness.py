import hashlib
import io
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

import server


class LivenessTests(unittest.TestCase):
    def setUp(self):
        server.RECOGNITION_PROOFS.clear()

    def tearDown(self):
        server.RECOGNITION_PROOFS.clear()
        server.ACCESS_COOLDOWNS.clear()

    def test_pinned_model_loads_and_returns_a_bounded_score(self):
        server.load_antispoof_model()
        with open(server.LIVENESS_MODEL, "rb") as model_file:
            model_hash = hashlib.sha256(model_file.read()).hexdigest()

        self.assertEqual(model_hash, server.LIVENESS_MODEL_SHA256)
        self.assertIsNotNone(server.liveness_model)
        image = np.full((240, 240, 3), 128, dtype=np.uint8)
        score = server.assess_liveness(image, [50, 35, 140, 170])
        self.assertTrue(np.isfinite(score))
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 1)

    def test_recognition_defers_passive_liveness_until_attendance_photo(self):
        ok, image = cv2.imencode(
            ".jpg", np.full((48, 64, 3), 120, dtype=np.uint8)
        )
        self.assertTrue(ok)
        upload = SimpleNamespace(file=io.BytesIO(image.tobytes()))

        with (
            patch.object(server, "detect", return_value=[[1, 2, 20, 24]]),
            patch.object(server, "embed", return_value=np.zeros(1)),
            patch.object(server, "identify", return_value=("Alice", 0.95)),
            patch.object(server, "assess_liveness") as assess_liveness,
            patch.object(server, "access_reason", return_value=None),
            patch.object(
                server.attendance_db,
                "employee_by_face_name",
                return_value={
                    "employee_id": "employee-1",
                    "status": "active",
                    "consent_at": "2026-01-01T08:00:00+03:00",
                },
            ),
        ):
            result = server.recognize(upload)["faces"][0]

        self.assertFalse(result["liveness_verified"])
        self.assertIsNone(result["liveness_score"])
        self.assertEqual(result["liveness_status"], "photo_check_required")
        proof = server.RECOGNITION_PROOFS[result["proof"]]
        self.assertEqual(proof[0], "Alice")
        assess_liveness.assert_not_called()


if __name__ == "__main__":
    unittest.main()
