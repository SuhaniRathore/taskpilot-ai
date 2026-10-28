"""
Test suite for TaskPilot AI – Ollama integration.

All HTTP / external calls are mocked; no live Ollama service is required.
Run with:  pytest tests/ -v
"""
import json
from unittest.mock import MagicMock, patch

import pytest


# ═══════════════════════════════════════════════════════════════════════════════
# config.py
# ═══════════════════════════════════════════════════════════════════════════════

class TestConfig:
    def test_defaults_exist(self):
        from backend.config import (
            AI_PROVIDER, OLLAMA_BASE_URL, OLLAMA_CHAT_MODEL,
            OLLAMA_EMBED_MODEL, GEMINI_MODEL, APP_NAME,
        )
        assert isinstance(AI_PROVIDER, str)
        assert OLLAMA_BASE_URL.startswith("http")
        assert OLLAMA_CHAT_MODEL
        assert OLLAMA_EMBED_MODEL
        assert GEMINI_MODEL
        assert APP_NAME

    def test_paths_exist(self):
        from backend.config import DATABASE_PATH, CHROMA_PATH, UPLOAD_DIR, REPORT_DIR
        for path in (DATABASE_PATH.parent, CHROMA_PATH, UPLOAD_DIR, REPORT_DIR):
            assert path.exists(), f"Expected directory {path} to be created"


# ═══════════════════════════════════════════════════════════════════════════════
# _safe_calculate  (unchanged logic, regression check)
# ═══════════════════════════════════════════════════════════════════════════════

class TestSafeCalculate:
    def test_basic_arithmetic(self):
        from backend.services.agent import _safe_calculate
        assert _safe_calculate("(18 * 4) / 3") == "24.0"
        assert _safe_calculate("2 ** 3") == "8"
        assert _safe_calculate("-5 + 10") == "5"

    def test_rejects_arbitrary_code(self):
        from backend.services.agent import _safe_calculate
        result = _safe_calculate("__import__('os').system('echo bad')")
        assert "error" in result.lower()

    def test_rejects_large_exponent(self):
        from backend.services.agent import _safe_calculate
        result = _safe_calculate("2 ** 100")
        assert "error" in result.lower()


# ═══════════════════════════════════════════════════════════════════════════════
# chunk_text  (regression)
# ═══════════════════════════════════════════════════════════════════════════════

class TestChunkText:
    def test_empty(self):
        from backend.services.rag import chunk_text
        assert chunk_text("") == []

    def test_overlap_preserved(self):
        from backend.services.rag import chunk_text
        chunks = chunk_text("x" * 2000, chunk_size=500, overlap=50)
        assert len(chunks) > 1
        assert chunks[0][-50:] == chunks[1][:50]


# ═══════════════════════════════════════════════════════════════════════════════
# _embed_ollama – mocked HTTP
# ═══════════════════════════════════════════════════════════════════════════════

class TestEmbedOllama:
    def _make_response(self, vector):
        mock = MagicMock()
        mock.raise_for_status = MagicMock()
        mock.json.return_value = {"embedding": vector}
        return mock

    def test_returns_vector(self):
        from backend.services import rag as rag_module
        fake_vector = [0.1, 0.2, 0.3]
        with patch("backend.services.rag.httpx.Client") as MockClient:
            ctx = MockClient.return_value.__enter__.return_value
            ctx.post.return_value = self._make_response(fake_vector)
            result = rag_module._embed_ollama(["hello world"])
        assert result == [fake_vector]
        ctx.post.assert_called_once()
        call_json = ctx.post.call_args.kwargs["json"]
        assert call_json["model"] == rag_module.OLLAMA_EMBED_MODEL
        assert call_json["prompt"] == "hello world"

    def test_multiple_texts(self):
        from backend.services import rag as rag_module
        vectors = [[0.1], [0.2], [0.3]]
        responses = [self._make_response(v) for v in vectors]
        with patch("backend.services.rag.httpx.Client") as MockClient:
            ctx = MockClient.return_value.__enter__.return_value
            ctx.post.side_effect = responses
            result = rag_module._embed_ollama(["a", "b", "c"])
        assert result == vectors
        assert ctx.post.call_count == 3

    def test_missing_embedding_key_raises(self):
        from backend.services import rag as rag_module
        import httpx as real_httpx
        bad_resp = MagicMock()
        bad_resp.raise_for_status = MagicMock()
        bad_resp.json.return_value = {"error": "model not found"}
        with patch("backend.services.rag.httpx.Client") as MockClient:
            ctx = MockClient.return_value.__enter__.return_value
            ctx.post.return_value = bad_resp
            with pytest.raises(RuntimeError, match="embedding"):
                rag_module._embed_ollama(["test"])

    def test_http_error_propagates(self):
        import httpx as real_httpx
        from backend.services import rag as rag_module
        bad_resp = MagicMock()
        bad_resp.raise_for_status.side_effect = real_httpx.HTTPStatusError(
            "500", request=MagicMock(), response=MagicMock()
        )
        with patch("backend.services.rag.httpx.Client") as MockClient:
            ctx = MockClient.return_value.__enter__.return_value
            ctx.post.return_value = bad_resp
            with pytest.raises(real_httpx.HTTPStatusError):
                rag_module._embed_ollama(["fail"])


# ═══════════════════════════════════════════════════════════════════════════════
# _embed dispatcher
# ═══════════════════════════════════════════════════════════════════════════════

class TestEmbedDispatch:
    def test_dispatches_to_ollama(self):
        import backend.services.rag as rag_module
        with patch.object(rag_module, "AI_PROVIDER", "ollama"), \
             patch.object(rag_module, "_embed_ollama", return_value=[[1.0]]) as mock_ol:
            result = rag_module._embed(["x"])
        mock_ol.assert_called_once_with(["x"])
        assert result == [[1.0]]

    def test_dispatches_to_gemini(self):
        import backend.services.rag as rag_module
        with patch.object(rag_module, "AI_PROVIDER", "gemini"), \
             patch.object(rag_module, "_embed_gemini", return_value=[[2.0]]) as mock_ge:
            result = rag_module._embed(["y"])
        mock_ge.assert_called_once_with(["y"])
        assert result == [[2.0]]


# ═══════════════════════════════════════════════════════════════════════════════
# _run_tool
# ═══════════════════════════════════════════════════════════════════════════════

class TestRunTool:
    def test_calculate(self):
        from backend.services.agent import _run_tool
        result = _run_tool("calculate", {"expression": "3 + 4"})
        assert result == {"result": "7"}

    def test_unknown_tool(self):
        from backend.services.agent import _run_tool
        result = _run_tool("nonexistent_tool", {})
        assert "error" in result

    def test_list_csv_files_wrapper(self):
        from backend.services.agent import _run_tool
        with patch("backend.services.agent.list_csv_files", return_value=["a.csv"]):
            result = _run_tool("list_csv_files", {})
        assert result == {"files": ["a.csv"]}

    def test_save_note(self):
        from backend.services.agent import _run_tool
        with patch("backend.services.agent.add_note", return_value=42):
            result = _run_tool("save_note", {"title": "T", "content": "C"})
        assert result == {"saved": True, "note_id": 42}

    def test_create_report(self):
        from backend.services.agent import _run_tool
        with patch("backend.services.agent.create_markdown_report", return_value={"created": True, "content": "# T"}):
            result = _run_tool("create_report", {"title": "T", "content": "C"})
        assert result["created"] is True


# ═══════════════════════════════════════════════════════════════════════════════
# _run_agent_ollama – mocked HTTP
# ═══════════════════════════════════════════════════════════════════════════════

class TestRunAgentOllama:
    def _mock_post(self, messages_sequence):
        """Return a mock httpx.Client that yields successive responses."""
        responses = []
        for msg in messages_sequence:
            r = MagicMock()
            r.raise_for_status = MagicMock()
            r.json.return_value = {"message": msg}
            responses.append(r)
        return responses

    def test_simple_answer_no_tools(self):
        from backend.services import agent as agent_module
        with patch("backend.services.agent.httpx.Client") as MockClient:
            ctx = MockClient.return_value.__enter__.return_value
            ctx.post.return_value = MagicMock(
                raise_for_status=MagicMock(),
                json=MagicMock(return_value={"message": {"role": "assistant", "content": "Hello!", "tool_calls": None}}),
            )
            result = agent_module._run_agent_ollama("hi", [])
        assert result["answer"] == "Hello!"
        assert result["tools_used"] == []
        assert result["tool_results"] == []

    def test_tool_call_then_answer(self):
        from backend.services import agent as agent_module
        tool_call_response = {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"function": {"name": "calculate", "arguments": {"expression": "2+2"}}}],
        }
        final_response = {
            "role": "assistant",
            "content": "The answer is 4.0",
            "tool_calls": None,
        }
        side_effects = [
            MagicMock(raise_for_status=MagicMock(), json=MagicMock(return_value={"message": tool_call_response})),
            MagicMock(raise_for_status=MagicMock(), json=MagicMock(return_value={"message": final_response})),
        ]
        with patch("backend.services.agent.httpx.Client") as MockClient:
            ctx = MockClient.return_value.__enter__.return_value
            ctx.post.side_effect = side_effects
            result = agent_module._run_agent_ollama("what is 2+2", [])
        assert "4" in result["answer"]
        assert "calculate" in result["tools_used"]

    def test_tool_args_as_json_string(self):
        """Ollama sometimes returns arguments as a JSON string – must be handled."""
        from backend.services import agent as agent_module
        tool_call_response = {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"function": {"name": "calculate", "arguments": '{"expression": "5*5"}'}}],
        }
        final_response = {"role": "assistant", "content": "25.0", "tool_calls": None}
        side_effects = [
            MagicMock(raise_for_status=MagicMock(), json=MagicMock(return_value={"message": tool_call_response})),
            MagicMock(raise_for_status=MagicMock(), json=MagicMock(return_value={"message": final_response})),
        ]
        with patch("backend.services.agent.httpx.Client") as MockClient:
            ctx = MockClient.return_value.__enter__.return_value
            ctx.post.side_effect = side_effects
            result = agent_module._run_agent_ollama("5 times 5", [])
        assert "calculate" in result["tools_used"]

    def test_connect_error_returns_friendly_message(self):
        import httpx as real_httpx
        from backend.services import agent as agent_module
        with patch("backend.services.agent.httpx.Client") as MockClient:
            ctx = MockClient.return_value.__enter__.return_value
            ctx.post.side_effect = real_httpx.ConnectError("refused")
            result = agent_module._run_agent_ollama("hello", [])
        assert "ollama" in result["answer"].lower() or "connect" in result["answer"].lower()
        assert result["tools_used"] == []

    def test_run_agent_dispatches_to_ollama(self):
        import backend.services.agent as agent_module
        with patch.object(agent_module, "AI_PROVIDER", "ollama"), \
             patch.object(agent_module, "_run_agent_ollama", return_value={"answer": "ok", "tools_used": [], "tool_results": []}) as mock_ol:
            result = agent_module.run_agent("test")
        mock_ol.assert_called_once()
        assert result["answer"] == "ok"

    def test_run_agent_dispatches_to_gemini(self):
        import backend.services.agent as agent_module
        with patch.object(agent_module, "AI_PROVIDER", "gemini"), \
             patch.object(agent_module, "_run_agent_gemini", return_value={"answer": "gemini", "tools_used": [], "tool_results": []}) as mock_ge:
            result = agent_module.run_agent("test")
        mock_ge.assert_called_once()

    def test_history_trimmed_to_last_8(self):
        from backend.services import agent as agent_module
        long_history = [{"role": "user", "content": f"msg{i}"} for i in range(20)]
        with patch("backend.services.agent.httpx.Client") as MockClient:
            ctx = MockClient.return_value.__enter__.return_value
            ctx.post.return_value = MagicMock(
                raise_for_status=MagicMock(),
                json=MagicMock(return_value={"message": {"role": "assistant", "content": "ok", "tool_calls": None}}),
            )
            agent_module._run_agent_ollama("new message", long_history)
        payload = ctx.post.call_args.kwargs["json"]
        # system(1) + last-8 history + 1 new user = 11 messages
        assert len(payload["messages"]) == 11


# ═══════════════════════════════════════════════════════════════════════════════
# Existing regression tests (kept intact)
# ═══════════════════════════════════════════════════════════════════════════════

class TestDatabaseAndNotes:
    def test_database_and_notes(self):
        from backend.database import save_task, get_recent_tasks, add_note, get_notes
        note_id = add_note("Test Note", "Sample note body")
        assert note_id > 0
        notes = get_notes(5)
        assert any(n["title"] == "Test Note" for n in notes)
        save_task("Test prompt", "Test response", ["test_tool"])
        tasks = get_recent_tasks(5)
        assert any(t["prompt"] == "Test prompt" for t in tasks)


class TestMarkdownReport:
    def test_markdown_report(self):
        from backend.services.reports import create_markdown_report
        res = create_markdown_report("Pytest Report", "Report generated in test")
        assert res["created"] is True
        assert "Pytest Report" in res["content"]


class TestFastAPIHealth:
    def test_health_endpoint(self):
        from fastapi.testclient import TestClient
        from backend.main import app
        with TestClient(app) as client:
            r = client.get("/health")
            assert r.status_code == 200
            assert r.json()["status"] == "ok"
