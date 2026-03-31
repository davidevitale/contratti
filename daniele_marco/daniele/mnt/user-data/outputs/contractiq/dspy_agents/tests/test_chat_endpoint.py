"""
Test 4 — /chat/question endpoint.
Mocks RAG service + DSPy orchestrator.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.mark.asyncio
class TestChatQuestionEndpoint:

    async def test_successful_chat(self, client, _patch_redis):
        """Normal flow: RAG context → orchestrator → structured response."""
        mock_prediction = MagicMock()
        mock_prediction.answer = "Il contratto scade il 28/02/2027."
        mock_prediction.confidence = "high"
        mock_prediction.sources = json.dumps([{"contract": "C-001", "clause": "Art. 2", "text": "durata 36 mesi"}])
        mock_prediction.follow_up_questions = json.dumps(["Ci sono clausole di rinnovo?", "Qual è il preavviso?"])

        mock_agent = MagicMock(return_value=mock_prediction)

        with patch("main.optimizer") as mock_optimizer, \
             patch("main.get_rag_context", new_callable=AsyncMock, return_value="Context from RAG"):
            mock_optimizer.load_optimized_orchestrator.return_value = mock_agent

            resp = await client.post("/chat/question", json={
                "client_id": "c1",
                "session_id": "s1",
                "question": "Quando scade il contratto?",
            })

        assert resp.status_code == 200
        body = resp.json()["response"]
        assert "2027" in body["answer"]
        assert body["confidence"] == "high"
        assert isinstance(body["sources"], list)
        assert len(body["sources"]) == 1
        assert isinstance(body["follow_up_questions"], list)

    async def test_chat_with_none_fields(self, client, _patch_redis):
        """If orchestrator returns None for some fields, response defaults apply."""
        mock_prediction = MagicMock()
        mock_prediction.answer = None
        mock_prediction.confidence = None
        mock_prediction.sources = None
        mock_prediction.follow_up_questions = None

        mock_agent = MagicMock(return_value=mock_prediction)

        with patch("main.optimizer") as mock_optimizer, \
             patch("main.get_rag_context", new_callable=AsyncMock, return_value=""):
            mock_optimizer.load_optimized_orchestrator.return_value = mock_agent

            resp = await client.post("/chat/question", json={
                "client_id": "c1",
                "session_id": "s1",
                "question": "Test question",
            })

        assert resp.status_code == 200
        body = resp.json()["response"]
        # getattr fallbacks kick in
        assert body["answer"] is None or isinstance(body["answer"], str)
        assert body["confidence"] is None or isinstance(body["confidence"], str)

    async def test_chat_with_rag_failure(self, client, _patch_redis):
        """RAG failure returns 'N/A' context; orchestrator still runs."""
        mock_prediction = MagicMock()
        mock_prediction.answer = "Non ho trovato informazioni nel contesto."
        mock_prediction.confidence = "low"
        mock_prediction.sources = "[]"
        mock_prediction.follow_up_questions = "[]"

        mock_agent = MagicMock(return_value=mock_prediction)

        with patch("main.optimizer") as mock_optimizer, \
             patch("main.get_rag_context", new_callable=AsyncMock, return_value="N/A"):
            mock_optimizer.load_optimized_orchestrator.return_value = mock_agent

            resp = await client.post("/chat/question", json={
                "client_id": "c1",
                "session_id": "s1",
                "question": "Any question",
            })

        assert resp.status_code == 200
        assert resp.json()["response"]["confidence"] == "low"

    async def test_chat_malformed_sources_json(self, client, _patch_redis):
        """If sources is not valid JSON, parse_blocks returns default []."""
        mock_prediction = MagicMock()
        mock_prediction.answer = "Some answer"
        mock_prediction.confidence = "medium"
        mock_prediction.sources = "not valid json{"
        mock_prediction.follow_up_questions = "also broken["

        mock_agent = MagicMock(return_value=mock_prediction)

        with patch("main.optimizer") as mock_optimizer, \
             patch("main.get_rag_context", new_callable=AsyncMock, return_value="ctx"):
            mock_optimizer.load_optimized_orchestrator.return_value = mock_agent

            resp = await client.post("/chat/question", json={
                "client_id": "c1",
                "session_id": "s1",
                "question": "Test",
            })

        assert resp.status_code == 200
        body = resp.json()["response"]
        assert body["sources"] == []
        assert body["follow_up_questions"] == []
