"""HTTP tests with a mocked classifier; no Gemini requests."""

import os
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from google.genai.errors import APIError
import httpx

from main import app
from email_classifier import DEFAULT_PROMPT


class ClassifyAPITests(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, {"GEMINI_API_KEY": "test-secret"}, clear=True)
        env.start()
        self.addCleanup(env.stop)
        mock = patch("main.classify_email")
        self.classifier = mock.start()
        self.addCleanup(mock.stop)
        self.classifier.return_value = {"category": "billing", "summary": "The customer requests a refund."}
        self.client = TestClient(app)
        self.addCleanup(self.client.close)

    def test_success_and_default_prompt(self):
        response = self.client.post("/classify", json={"email": "Refund please."})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), self.classifier.return_value)
        self.classifier.assert_called_once_with("Refund please.", prompt=DEFAULT_PROMPT)

    def test_environment_model(self):
        with patch.dict(os.environ, {"GEMINI_MODEL": "configured-model"}):
            self.assertEqual(self.client.post("/classify", json={"email": "Help"}).status_code, 200)
        self.assertEqual(self.classifier.call_args.kwargs["model"], "configured-model")

    def test_invalid_requests(self):
        for payload in ({}, {"email": ""}, {"email": " \n "}, {"email": None},
                        {"email": 3}, {"email": []}, {"email": "Hi", "extra": True}):
            with self.subTest(payload=payload):
                self.assertEqual(self.client.post("/classify", json=payload).status_code, 422)
        self.assertEqual(self.client.post("/classify", content="bad json",
                                        headers={"Content-Type": "application/json"}).status_code, 422)
        self.classifier.assert_not_called()

    def test_missing_configuration(self):
        for config in ({"GEMINI_API_KEY": ""}, {"GEMINI_MODEL": " "}):
            with self.subTest(config=config), patch.dict(os.environ, config):
                self.assertEqual(self.client.post("/classify", json={"email": "Hi"}).status_code, 503)
        self.classifier.assert_not_called()

    def test_failure_status_codes_and_secret_safety(self):
        for exception, code in (
            (APIError(429, {"message": "test-secret"}), 429),
            (APIError(503, {"message": "test-secret"}), 502),
            (APIError(401, {"message": "test-secret"}), 502),
            (APIError(504, {"message": "test-secret"}), 504),
            (httpx.ReadTimeout("test-secret"), 504),
            (TimeoutError("test-secret"), 504),
            (httpx.ConnectError("test-secret"), 502),
            (ValueError("test-secret"), 502),
            (RuntimeError("test-secret"), 500),
        ):
            with self.subTest(exception=type(exception).__name__, code=code):
                self.classifier.side_effect = exception
                response = self.client.post("/classify", json={"email": "Help"})
                self.assertEqual(response.status_code, code)
                self.assertNotIn("test-secret", response.text)
                self.assertEqual(set(response.json()), {"detail"})

    def test_invalid_classifier_output_is_rejected(self):
        for output in ({"category": "other", "summary": "Help."},
                       {"category": "billing", "summary": " "}, {"category": "billing"}):
            with self.subTest(output=output):
                self.classifier.return_value = output
                self.assertEqual(self.client.post("/classify", json={"email": "Help"}).status_code, 502)

    def test_swagger_and_openapi_are_offline(self):
        self.assertEqual(self.client.get("/docs").status_code, 200)
        schema = self.client.get("/openapi.json").json()
        self.assertIn("post", schema["paths"]["/classify"])
        self.assertIn("ClassifierOutput", schema["components"]["schemas"])
        self.classifier.assert_not_called()
