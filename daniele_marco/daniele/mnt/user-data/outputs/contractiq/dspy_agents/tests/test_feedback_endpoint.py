"""
Test 5 — /feedback endpoint + optimization trigger logic.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.mark.asyncio
class TestFeedbackEndpoint:

    async def test_store_feedback(self, client, _patch_redis, mock_redis):
        """Feedback is stored in Redis and counter is returned."""
        mock_redis.llen.return_value = 5

        resp = await client.post("/feedback", json={
            "client_id": "c1",
            "session_id": "s1",
            "question": "Quando scade?",
            "answer": "Il 28/02/2027.",
            "sources": "[]",
            "user_rating": 4,
            "user_comment": "Perfetto",
        })

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "stored"
        assert body["current_examples"] == 5
        assert body["optimizing_now"] is False

        # Verify rpush was called
        mock_redis.rpush.assert_awaited_once()
        pushed = json.loads(mock_redis.rpush.call_args[0][1])
        assert pushed["user_rating"] == 4
        assert pushed["user_comment"] == "Perfetto"

    async def test_optimization_trigger_at_threshold(self, client, _patch_redis, mock_redis):
        """When count hits 20, optimization should trigger."""
        mock_redis.llen.return_value = 20

        resp = await client.post("/feedback", json={
            "client_id": "c1",
            "session_id": "s1",
            "question": "Q",
            "answer": "A",
            "sources": [],
            "user_rating": 5,
        })

        assert resp.status_code == 200
        body = resp.json()
        assert body["optimizing_now"] is True

    async def test_no_optimization_below_threshold(self, client, _patch_redis, mock_redis):
        mock_redis.llen.return_value = 19

        resp = await client.post("/feedback", json={
            "client_id": "c1",
            "session_id": "s1",
            "question": "Q",
            "answer": "A",
            "sources": "[]",
            "user_rating": 3,
        })

        assert resp.status_code == 200
        assert resp.json()["optimizing_now"] is False

    async def test_redis_unavailable(self, client):
        """When redis_client is None, endpoint returns redis_unavailable."""
        import main as m
        original = m.redis_client
        m.redis_client = None

        resp = await client.post("/feedback", json={
            "client_id": "c1",
            "session_id": "s1",
            "question": "Q",
            "answer": "A",
            "sources": "[]",
            "user_rating": 2,
        })

        m.redis_client = original
        assert resp.status_code == 200
        assert resp.json()["status"] == "redis_unavailable"

    async def test_feedback_accepts_sources_as_list_or_string(self, client, _patch_redis, mock_redis):
        """sources field can be either a JSON string or a list."""
        mock_redis.llen.return_value = 1

        # As list
        resp1 = await client.post("/feedback", json={
            "client_id": "c1", "session_id": "s1",
            "question": "Q", "answer": "A",
            "sources": [{"contract": "C1"}],
            "user_rating": 5,
        })
        assert resp1.status_code == 200

        # As string
        resp2 = await client.post("/feedback", json={
            "client_id": "c1", "session_id": "s1",
            "question": "Q", "answer": "A",
            "sources": '[{"contract": "C1"}]',
            "user_rating": 5,
        })
        assert resp2.status_code == 200
