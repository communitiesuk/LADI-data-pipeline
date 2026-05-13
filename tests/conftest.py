"""Shared pytest fixtures for LADI data pipeline tests."""
import json
from unittest.mock import AsyncMock, MagicMock

import pytest


# ---------------------------------------------------------------------------
# Environment variable fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def apim_env(monkeypatch):
    """Patch all required APIM environment variables."""
    monkeypatch.setenv("LADI_APIM_BASE_URL", "https://apim.example.com/openai")
    monkeypatch.setenv("LADI_APIM_EMBEDDING_URL", "https://apim.example.com/embeddings")
    monkeypatch.setenv("LADI_APIM_SUBSCRIPTION_KEY", "fake-sub-key")
    monkeypatch.setenv("LADI_APIM_TOKEN_SCOPE", "https://cognitiveservices.azure.com/.default")
    monkeypatch.setenv("LADI_APIM_API_VERSION", "2024-02-01")


# ---------------------------------------------------------------------------
# Mock OpenAI client
# ---------------------------------------------------------------------------

def _make_chat_response(content: str):
    """Build a minimal fake ChatCompletion object."""
    message = MagicMock()
    message.content = content

    choice = MagicMock()
    choice.message = message

    response = MagicMock()
    response.choices = [choice]
    return response


@pytest.fixture()
def summarise_json_response():
    return json.dumps({
        "title": "Test Budget 2024",
        "year": 2024,
        "summary": "A budget document for the test council.",
        "themes": ["Local government finance"],
    })


@pytest.fixture()
def mock_openai_client(summarise_json_response):
    """AsyncOpenAI client that returns a canned summarise response."""
    client = MagicMock()
    client.chat = MagicMock()
    client.chat.completions = MagicMock()
    client.chat.completions.create = AsyncMock(
        return_value=_make_chat_response(summarise_json_response)
    )
    return client


# ---------------------------------------------------------------------------
# Sample config
# ---------------------------------------------------------------------------

@pytest.fixture()
def sample_config():
    return {
        "crawl": {"output_dir": "output/crawl"},
        "summarise": {
            "output_file": "output/summaries.jsonl",
            "min_words": 100,
        },
    }


# ---------------------------------------------------------------------------
# Temporary CSV fixture
# ---------------------------------------------------------------------------

LONG_TEXT = " ".join(["word"] * 150)  # 150 words — above the 100-word threshold
SHORT_TEXT = "too short"              # below threshold

@pytest.fixture()
def sample_csv(tmp_path):
    """CSV with 5 rows: 3 long docs (processable) and 2 short docs (skipped)."""
    csv_path = tmp_path / "docs.csv"
    rows = [
        ("http://council.example/doc1.pdf", "Alpha Council", LONG_TEXT),
        ("http://council.example/doc2.pdf", "Beta Council",  LONG_TEXT),
        ("http://council.example/doc3.pdf", "Gamma Council", LONG_TEXT),
        ("http://council.example/short1.pdf", "Delta Council", SHORT_TEXT),
        ("http://council.example/short2.pdf", "Epsilon Council", SHORT_TEXT),
    ]
    lines = ["Document Link,Authority Name,text"]
    for url, auth, text in rows:
        lines.append(f'"{url}","{auth}","{text}"')
    csv_path.write_text("\n".join(lines))
    return csv_path


# ---------------------------------------------------------------------------
# Temporary JSONL checkpoint fixture
# ---------------------------------------------------------------------------

FAKE_EMBEDDING = [0.1, 0.2, 0.3]  # minimal 3-dim vector for tests


def _make_embedding_response(texts):
    """Build a minimal fake CreateEmbeddingResponse for a list of input texts."""
    data = []
    for i, _ in enumerate(texts):
        emb = MagicMock()
        emb.embedding = FAKE_EMBEDDING
        emb.index = i
        data.append(emb)
    response = MagicMock()
    response.data = data
    return response


@pytest.fixture()
def mock_embedding_client():
    """AsyncOpenAI client that returns canned embedding vectors."""
    client = MagicMock()
    client.embeddings = MagicMock()
    client.embeddings.create = AsyncMock(
        side_effect=lambda model, input, **kw: _make_embedding_response(
            input if isinstance(input, list) else [input]
        )
    )
    return client


@pytest.fixture()
def sample_summaries_jsonl(tmp_path):
    """JSONL file with summarise-stage output (title, summary, url, authority, themes)."""
    jsonl_path = tmp_path / "summaries.jsonl"
    entries = [
        {
            "url": "http://council.example/doc1.pdf",
            "authority": "Alpha Council",
            "title": "Budget 2024",
            "year": 2024,
            "summary": "A budget document for Alpha Council.",
            "themes": ["Local government finance"],
        },
        {
            "url": "http://council.example/doc2.pdf",
            "authority": "Beta Council",
            "title": "Housing Strategy",
            "year": 2023,
            "summary": "Beta Council housing strategy.",
            "themes": ["Housing"],
        },
        {
            "url": "http://council.example/doc3.pdf",
            "authority": "Gamma Council",
            "title": "Transport Plan",
            "year": 2022,
            "summary": "Transport plan for Gamma Council.",
            "themes": ["Transport and highways"],
        },
    ]
    jsonl_path.write_text("\n".join(json.dumps(e) for e in entries) + "\n")
    return jsonl_path


@pytest.fixture()
def sample_config_with_embed():
    return {
        "crawl": {"output_dir": "output/crawl"},
        "summarise": {"output_file": "output/summaries.jsonl", "min_words": 100},
        "embed": {
            "model": "text-embedding-3-large",
            "input_field": "summary",
            "batch_size": 2,
            "output_file": "output/embeddings.jsonl",
        },
    }


@pytest.fixture()
def sample_jsonl(tmp_path):
    """JSONL file pre-populated with 2 already-processed URLs."""
    jsonl_path = tmp_path / "summaries.jsonl"
    entries = [
        {"url": "http://council.example/doc1.pdf", "title": "Doc 1", "year": 2023,
         "summary": "Already done.", "themes": ["Housing"], "authority": "Alpha Council"},
        {"url": "http://council.example/doc2.pdf", "title": "Doc 2", "year": 2022,
         "summary": "Also done.",   "themes": ["Education"], "authority": "Beta Council"},
    ]
    jsonl_path.write_text("\n".join(json.dumps(e) for e in entries) + "\n")
    return jsonl_path
