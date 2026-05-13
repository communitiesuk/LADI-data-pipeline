"""Tests for stages/03_embedding.py."""
import asyncio
import importlib.util
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

# Import the module under test (it lives in stages/, not a package)
_spec = importlib.util.spec_from_file_location(
    "embedding",
    Path(__file__).parent.parent / "stages" / "03_embedding.py",
)
embedding = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(embedding)


# ---------------------------------------------------------------------------
# Pure functions: load_summaries
# ---------------------------------------------------------------------------

class TestLoadSummaries:
    def test_loads_valid_records(self, sample_summaries_jsonl):
        records = embedding.load_summaries(sample_summaries_jsonl)
        assert len(records) == 3
        assert records[0]["url"] == "http://council.example/doc1.pdf"
        assert records[1]["summary"] == "Beta Council housing strategy."

    def test_skips_error_records(self, tmp_path):
        """Records with 'error' key and no 'summary' are skipped."""
        jsonl_path = tmp_path / "input.jsonl"
        entries = [
            {"url": "http://good.example/doc.pdf", "summary": "Good doc.", "title": "Good"},
            {"url": "http://bad.example/doc.pdf", "error": "garbled_text"},
            {"url": "http://good2.example/doc.pdf", "summary": "Another good.", "title": "Good2"},
        ]
        jsonl_path.write_text("\n".join(json.dumps(e) for e in entries) + "\n")
        records = embedding.load_summaries(jsonl_path)
        assert len(records) == 2
        assert all("summary" in r for r in records)

    def test_skips_malformed_lines(self, tmp_path):
        jsonl_path = tmp_path / "bad.jsonl"
        jsonl_path.write_text(
            '{"url": "http://ok.example/doc.pdf", "summary": "Ok."}\n'
            'NOT JSON\n'
            '{"url": "http://ok2.example/doc.pdf", "summary": "Ok2."}\n'
        )
        records = embedding.load_summaries(jsonl_path)
        assert len(records) == 2

    def test_skips_blank_lines(self, tmp_path):
        jsonl_path = tmp_path / "blanks.jsonl"
        jsonl_path.write_text(
            '{"url": "http://a.example/doc.pdf", "summary": "A."}\n'
            '\n'
            '{"url": "http://b.example/doc.pdf", "summary": "B."}\n'
            '\n'
        )
        records = embedding.load_summaries(jsonl_path)
        assert len(records) == 2


# ---------------------------------------------------------------------------
# Pure functions: load_checkpoint
# ---------------------------------------------------------------------------

class TestLoadCheckpoint:
    def test_no_file_returns_empty_set(self, tmp_path):
        path = tmp_path / "does_not_exist.jsonl"
        result = embedding.load_checkpoint(path)
        assert result == set()

    def test_with_urls_returns_set(self, tmp_path):
        path = tmp_path / "embeddings.jsonl"
        entries = [
            {"url": "http://a.example/doc.pdf", "embedding": [0.1, 0.2]},
            {"url": "http://b.example/doc.pdf", "embedding": [0.3, 0.4]},
        ]
        path.write_text("\n".join(json.dumps(e) for e in entries) + "\n")
        result = embedding.load_checkpoint(path)
        assert result == {"http://a.example/doc.pdf", "http://b.example/doc.pdf"}

    def test_skips_malformed_lines(self, tmp_path):
        path = tmp_path / "bad.jsonl"
        path.write_text(
            '{"url": "http://good.example/doc.pdf", "embedding": [0.1]}\n'
            'NOT JSON\n'
        )
        result = embedding.load_checkpoint(path)
        assert result == {"http://good.example/doc.pdf"}

    def test_skips_entries_without_url(self, tmp_path):
        path = tmp_path / "no_url.jsonl"
        path.write_text('{"embedding": [0.1]}\n{"url": "http://has.url/", "embedding": [0.2]}\n')
        result = embedding.load_checkpoint(path)
        assert result == {"http://has.url/"}


# ---------------------------------------------------------------------------
# Pure functions: load_config
# ---------------------------------------------------------------------------

class TestLoadConfig:
    def test_load_config_returns_dict(self, tmp_path, sample_config_with_embed):
        cfg_file = tmp_path / "pipeline.yaml"
        cfg_file.write_text(yaml.dump(sample_config_with_embed))
        result = embedding.load_config(str(cfg_file))
        assert result == sample_config_with_embed


# ---------------------------------------------------------------------------
# Async: embed_batch
# ---------------------------------------------------------------------------

class TestEmbedBatch:
    async def test_returns_records_with_embedding(self, mock_embedding_client):
        batch = [
            {"url": "http://a.example/doc.pdf", "summary": "Doc A summary."},
            {"url": "http://b.example/doc.pdf", "summary": "Doc B summary."},
        ]
        results = await embedding.embed_batch(mock_embedding_client, batch, "summary")
        assert len(results) == 2
        assert results[0]["embedding"] == [0.1, 0.2, 0.3]
        assert results[1]["embedding"] == [0.1, 0.2, 0.3]
        # Original fields preserved
        assert results[0]["url"] == "http://a.example/doc.pdf"
        assert results[1]["summary"] == "Doc B summary."

    async def test_passes_correct_texts_to_api(self, mock_embedding_client):
        batch = [
            {"url": "http://x.example/doc.pdf", "summary": "First summary."},
            {"url": "http://y.example/doc.pdf", "summary": "Second summary."},
        ]
        await embedding.embed_batch(mock_embedding_client, batch, "summary")
        call_kwargs = mock_embedding_client.embeddings.create.call_args
        assert call_kwargs[1]["input"] == ["First summary.", "Second summary."]

    async def test_uses_specified_input_field(self, mock_embedding_client):
        batch = [{"url": "http://x.example/doc.pdf", "summary": "S", "title": "The Title"}]
        await embedding.embed_batch(mock_embedding_client, batch, "title")
        call_kwargs = mock_embedding_client.embeddings.create.call_args
        assert call_kwargs[1]["input"] == ["The Title"]

    async def test_handles_missing_field_gracefully(self, mock_embedding_client):
        """If the input_field is missing, passes empty string."""
        batch = [{"url": "http://x.example/doc.pdf"}]
        results = await embedding.embed_batch(mock_embedding_client, batch, "summary")
        call_kwargs = mock_embedding_client.embeddings.create.call_args
        assert call_kwargs[1]["input"] == [""]
        assert results[0]["embedding"] == [0.1, 0.2, 0.3]


# ---------------------------------------------------------------------------
# Async: worker
# ---------------------------------------------------------------------------

class TestWorker:
    async def test_writes_jsonl_on_success(self, mock_embedding_client, tmp_path):
        out_path = tmp_path / "out.jsonl"
        counters = {"done": 0, "errors": 0, "rate_limits": 0}
        queue = asyncio.Queue()

        batch = [
            {"url": "http://a.example/doc.pdf", "summary": "A summary."},
            {"url": "http://b.example/doc.pdf", "summary": "B summary."},
        ]
        await queue.put(batch)
        await queue.put(None)  # sentinel

        with open(out_path, "a") as fh:
            await embedding.worker(queue, mock_embedding_client, fh, counters, "summary")

        lines = [l for l in out_path.read_text().splitlines() if l.strip()]
        assert len(lines) == 2
        assert json.loads(lines[0])["url"] == "http://a.example/doc.pdf"
        assert json.loads(lines[1])["url"] == "http://b.example/doc.pdf"
        assert counters["done"] == 2
        assert counters["errors"] == 0

    async def test_handles_429_with_retry(self, tmp_path):
        """First call raises 429, retry succeeds."""
        from tests.conftest import FAKE_EMBEDDING

        call_count = 0

        async def side_effect(model, input, **kw):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("429 Too Many Requests")
            resp = MagicMock()
            resp.data = []
            for i, _ in enumerate(input):
                emb = MagicMock()
                emb.embedding = FAKE_EMBEDDING
                resp.data.append(emb)
            return resp

        client = MagicMock()
        client.embeddings = MagicMock()
        client.embeddings.create = AsyncMock(side_effect=side_effect)

        out_path = tmp_path / "out.jsonl"
        counters = {"done": 0, "errors": 0, "rate_limits": 0}
        queue = asyncio.Queue()
        batch = [{"url": "http://test.example/doc.pdf", "summary": "Test."}]
        await queue.put(batch)
        await queue.put(None)

        with patch("asyncio.sleep", new_callable=AsyncMock):
            with open(out_path, "a") as fh:
                await embedding.worker(queue, client, fh, counters, "summary")

        assert counters["done"] == 1
        assert counters["rate_limits"] == 1
        assert counters["errors"] == 0

    async def test_non_429_error_increments_error_counter(self, tmp_path):
        """Non-429 exception increments errors by batch size."""
        client = MagicMock()
        client.embeddings = MagicMock()
        client.embeddings.create = AsyncMock(
            side_effect=ValueError("unexpected error")
        )

        out_path = tmp_path / "out.jsonl"
        counters = {"done": 0, "errors": 0, "rate_limits": 0}
        queue = asyncio.Queue()
        batch = [
            {"url": "http://a.example/doc.pdf", "summary": "A."},
            {"url": "http://b.example/doc.pdf", "summary": "B."},
        ]
        await queue.put(batch)
        await queue.put(None)

        with open(out_path, "a") as fh:
            await embedding.worker(queue, client, fh, counters, "summary")

        assert counters["errors"] == 2  # batch size
        assert counters["done"] == 0
        assert out_path.read_text().strip() == ""


# ---------------------------------------------------------------------------
# Integration tests (mocked API, real file I/O)
# ---------------------------------------------------------------------------

class TestRunIntegration:
    async def test_run_end_to_end(
        self, sample_summaries_jsonl, tmp_path, sample_config_with_embed, mock_embedding_client
    ):
        """Summarise JSONL with mocked API produces correct embeddings JSONL."""
        output_file = str(tmp_path / "embeddings.jsonl")
        sample_config_with_embed["embed"]["output_file"] = output_file

        with patch.object(embedding, "build_client", return_value=mock_embedding_client):
            await embedding.run(
                sample_config_with_embed,
                str(sample_summaries_jsonl),
                output_file,
                concurrency=2,
            )

        lines = [l for l in Path(output_file).read_text().splitlines() if l.strip()]
        assert len(lines) == 3
        for line in lines:
            obj = json.loads(line)
            assert "url" in obj
            assert "embedding" in obj
            assert obj["embedding"] == [0.1, 0.2, 0.3]
            # Original fields preserved
            assert "summary" in obj
            assert "authority" in obj

    async def test_run_skips_checkpointed_urls(
        self, sample_summaries_jsonl, tmp_path, sample_config_with_embed, mock_embedding_client
    ):
        """URLs already in output JSONL are skipped."""
        output_file = tmp_path / "embeddings.jsonl"
        # Pre-populate with doc1 already embedded
        output_file.write_text(
            json.dumps({
                "url": "http://council.example/doc1.pdf",
                "summary": "A budget document for Alpha Council.",
                "embedding": [0.9, 0.8, 0.7],
            }) + "\n"
        )
        sample_config_with_embed["embed"]["output_file"] = str(output_file)

        with patch.object(embedding, "build_client", return_value=mock_embedding_client):
            await embedding.run(
                sample_config_with_embed,
                str(sample_summaries_jsonl),
                str(output_file),
                concurrency=1,
            )

        lines = [l for l in output_file.read_text().splitlines() if l.strip()]
        # 1 pre-existing + 2 new
        assert len(lines) == 3
        urls = {json.loads(l)["url"] for l in lines}
        assert "http://council.example/doc1.pdf" in urls
        assert "http://council.example/doc2.pdf" in urls
        assert "http://council.example/doc3.pdf" in urls

    async def test_run_skips_error_records(
        self, tmp_path, sample_config_with_embed, mock_embedding_client
    ):
        """Error records from summarise stage are not embedded."""
        input_path = tmp_path / "summaries.jsonl"
        entries = [
            {"url": "http://a.example/doc.pdf", "summary": "Valid.", "title": "A"},
            {"url": "http://b.example/doc.pdf", "error": "garbled_text"},
        ]
        input_path.write_text("\n".join(json.dumps(e) for e in entries) + "\n")
        output_file = str(tmp_path / "embeddings.jsonl")
        sample_config_with_embed["embed"]["output_file"] = output_file

        with patch.object(embedding, "build_client", return_value=mock_embedding_client):
            await embedding.run(
                sample_config_with_embed,
                str(input_path),
                output_file,
                concurrency=1,
            )

        lines = [l for l in Path(output_file).read_text().splitlines() if l.strip()]
        assert len(lines) == 1
        assert json.loads(lines[0])["url"] == "http://a.example/doc.pdf"

    async def test_run_nothing_to_do(
        self, tmp_path, sample_config_with_embed, mock_embedding_client
    ):
        """All URLs already checkpointed — exits cleanly."""
        input_path = tmp_path / "summaries.jsonl"
        input_path.write_text(
            json.dumps({"url": "http://a.example/doc.pdf", "summary": "A."}) + "\n"
        )
        output_file = tmp_path / "embeddings.jsonl"
        output_file.write_text(
            json.dumps({"url": "http://a.example/doc.pdf", "embedding": [0.1]}) + "\n"
        )
        sample_config_with_embed["embed"]["output_file"] = str(output_file)

        with patch.object(embedding, "build_client", return_value=mock_embedding_client):
            await embedding.run(
                sample_config_with_embed,
                str(input_path),
                str(output_file),
                concurrency=1,
            )

        # Output unchanged (nothing new appended)
        lines = [l for l in output_file.read_text().splitlines() if l.strip()]
        assert len(lines) == 1
