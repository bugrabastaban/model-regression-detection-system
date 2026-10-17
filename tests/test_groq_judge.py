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

    async def score_response(self, data):
        calls = []

        def handler(request):
            calls.append(request)
            return httpx.Response(200, json=data)

        judge = GroqJudge(env_file=self.env, transport=httpx.MockTransport(handler))
        dataset = {'dataset_version': 'v1', 'cases': [{'id': 'a', 'input': 'Refund please.',
                   'expected_category': 'billing', 'ideal_summary': 'A refund is requested.'}]}
        report = {'dataset_version': 'v1', 'results': [{'case_id': 'a', 'elapsed_seconds': 1,
                  'error': None, 'output': {'category': 'billing', 'summary': 'A refund is requested.'}}]}
        result = await score_results(report, dataset, judge=judge)
        self.assertEqual(len(calls), 1)
        return result['results'][0]['scores']

    async def test_valid_partial_zero_and_missing_usage_preserve_judge_score(self):
        variants = [
            ({'prompt_tokens': 20, 'completion_tokens': 10, 'total_tokens': 37},
             {'input_tokens': 20, 'output_tokens': 10, 'total_tokens': 37}),
            ({'prompt_tokens': 0},
             {'input_tokens': 0, 'output_tokens': None, 'total_tokens': None}),
            ({'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0},
             {'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0}),
            (None, {'input_tokens': None, 'output_tokens': None, 'total_tokens': None}),
        ]
        for usage, expected in variants:
            with self.subTest(usage=usage):
                data = self.response()
                data['usage'] = usage
                score = await self.score_response(data)
                self.assertEqual(score['summary_relevance'], 5)
                self.assertEqual(score['judge']['status'], 'scored')
                self.assertIsNone(score['judge']['error'])
                self.assertEqual(score['judge']['token_usage'], expected)

    async def test_negative_usage_preserves_valid_judge_decision(self):
        for counter in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
            with self.subTest(counter=counter):
                data = self.response()
                data['usage'][counter] = -1
                score = await self.score_response(data)
                self.assertEqual(score['summary_relevance'], 5)
                self.assertEqual(score['judge']['status'], 'scored')
                self.assertEqual(score['judge']['reason'], 'Faithful.')
                self.assertIsNone(score['judge']['error'])
                self.assertEqual(score['judge']['token_usage'],
                                 dict.fromkeys(('input_tokens', 'output_tokens', 'total_tokens')))

    async def test_malformed_usage_preserves_valid_judge_decision(self):
        for usage in ([1], 'unexpected', 7, True, {'prompt_tokens': '20'},
                      {'completion_tokens': False}, {'total_tokens': 1.5},
                      {'prompt_tokens': {'secret': 'test-secret'}}):
            with self.subTest(usage=usage):
                data = self.response()
                data['usage'] = usage
                score = await self.score_response(data)
                self.assertEqual(score['summary_relevance'], 5)
                self.assertEqual(score['judge']['status'], 'scored')
                self.assertIsNone(score['judge']['error'])
                self.assertTrue(all(value is None for value in score['judge']['token_usage'].values()))

    async def test_invalid_verdict_or_envelope_is_not_repaired_by_usage_fallback(self):
        for content in ('not JSON', '{"score":6,"reason":"Invalid score."}'):
            with self.subTest(content=content):
                data = self.response(content)
                data['usage'] = {'prompt_tokens': -1}
                score = await self.score_response(data)
                self.assertIsNone(score['summary_relevance'])
                self.assertEqual(score['judge']['status'], 'error')
                self.assertIsNotNone(score['judge']['error'])
                self.assertEqual(score['judge']['raw_output'], content)
        data = self.response()
        data['choices'][0]['finish_reason'] = 'length'
        data['usage'] = {'prompt_tokens': -1}
        score = await self.score_response(data)
        self.assertIsNone(score['summary_relevance'])
        self.assertEqual(score['judge']['error']['type'], 'GroqJudgeError')

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
