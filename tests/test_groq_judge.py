"""Groq requests intercepted by MockTransport; never uses real credentials."""

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import IsolatedAsyncioTestCase
from unittest.mock import patch

import httpx

from groq_judge import DEFAULT_MODEL, GroqJudge, GroqJudgeError
from scoring import score_results


class GroqJudgeTests(IsolatedAsyncioTestCase):
    def setUp(self):
        env = patch.dict(os.environ, {}, clear=True)
        env.start()
        self.addCleanup(env.stop)
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.env = Path(directory.name) / '.env'
        self.env.write_text('GROQ_API_KEY=test-secret\n', encoding='utf-8')

    def response(self, content='{"score":5,"reason":"Faithful."}'):
        return {"choices": [{"finish_reason": "stop", "message": {"content": content}}],
                "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30}}

    async def test_request_schema_auth_and_scoring_integration(self):
        calls = []
        def handler(request):
            calls.append(request)
            self.assertEqual(request.headers['Authorization'], 'Bearer test-secret')
            body = json.loads(request.content)
            self.assertEqual(body['model'], DEFAULT_MODEL)
            self.assertTrue(body['response_format']['json_schema']['strict'])
            self.assertEqual(body['response_format']['json_schema']['schema']['required'], ['score', 'reason'])
            return httpx.Response(200, json=self.response())
        judge = GroqJudge(env_file=self.env, transport=httpx.MockTransport(handler))
        self.assertEqual(calls, [])
        dataset = {'dataset_version': 'v1', 'cases': [{'id': 'a', 'input': 'Refund please.',
                   'expected_category': 'billing', 'ideal_summary': 'A refund is requested.'}]}
        report = {'dataset_version': 'v1', 'results': [{'case_id': 'a', 'elapsed_seconds': 1,
                  'error': None, 'output': {'category': 'billing', 'summary': 'A refund is requested.'}}]}
        result = await score_results(report, dataset, judge=judge)
        score = result['results'][0]['scores']
        self.assertEqual(score['summary_relevance'], 5)
        self.assertEqual(score['judge']['model'], DEFAULT_MODEL)
        self.assertEqual(score['judge']['token_usage']['total_tokens'], 30)
        self.assertEqual(len(calls), 1)

    async def test_configuration_and_precedence(self):
        with patch.dict(os.environ, {'GROQ_API_KEY': '', 'GROQ_MODEL': DEFAULT_MODEL}):
            with self.assertRaises(ValueError):
                GroqJudge(env_file=self.env)
        with patch.dict(os.environ, {'GROQ_MODEL': 'custom-model'}):
            self.assertEqual(GroqJudge(env_file=self.env).model, 'custom-model')

    async def test_http_errors_safe_and_not_retried(self):
        for status in (401, 429, 500, 503, 302):
            calls = []
            def handler(request):
                calls.append(request)
                return httpx.Response(status, text='test-secret', headers={'Location': 'https://example.com'})
            judge = GroqJudge(env_file=self.env, transport=httpx.MockTransport(handler))
            with self.subTest(status=status), self.assertRaises(GroqJudgeError) as error:
                await judge('system', '{}')
            self.assertIn(str(status), str(error.exception))
            self.assertNotIn('test-secret', str(error.exception))
            self.assertEqual(len(calls), 1)

    async def test_transport_errors_safe_and_not_retried(self):
        for error_type in (httpx.ReadTimeout, httpx.ConnectError):
            calls = []
            def handler(request):
                calls.append(request)
                raise error_type('test-secret')
            judge = GroqJudge(env_file=self.env, transport=httpx.MockTransport(handler))
            with self.assertRaises(GroqJudgeError) as error:
                await judge('system', '{}')
            self.assertNotIn('test-secret', str(error.exception))
            self.assertEqual(len(calls), 1)

    async def test_invalid_envelopes(self):
        truncated = self.response()
        truncated['choices'][0]['finish_reason'] = 'length'
        for data in ({}, {'choices': []}, self.response(None), truncated):
            judge = GroqJudge(env_file=self.env, transport=httpx.MockTransport(
                lambda request: httpx.Response(200, json=data)))
            with self.assertRaises(GroqJudgeError):
                await judge('system', '{}')

    async def test_raw_invalid_verdict_and_unknown_usage_preserved(self):
        data = self.response('not a verdict')
        del data['usage']
        judge = GroqJudge(env_file=self.env, transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json=data)))
        result = await judge('system', '{}')
        self.assertEqual(result.raw_output, 'not a verdict')
        self.assertIsNone(result.token_usage.total_tokens)
