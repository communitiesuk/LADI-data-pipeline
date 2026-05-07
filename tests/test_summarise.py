"""Tests for stages/02_summarise.py."""
import asyncio
import importlib.util
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

# Import the module under test (it lives in stages/, not a package)
_spec = importlib.util.spec_from_file_location(
    "summarise",
    Path(__file__).parent.parent / "stages" / "02_summarise.py",
)
summarise = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(summarise)


# ---------------------------------------------------------------------------
# Pure functions: clean_text
# ---------------------------------------------------------------------------

class TestCleanText:
    def test_strips_control_chars(self):
        """Control chars \x00–\x08, \x0b, \x0c, \x0e–\x1f are removed."""
        dirty = "hello\x00\x01\x02\x03\x04\x05\x06\x07\x08 world\x0b\x0c\x0e\x1f"
        result = summarise.clean_text(dirty)
        assert result == "hello world"

    def test_preserves_normal_chars(self):
        """Normal printable chars, £, newline, tab are preserved."""
        text = "Hello £10\nworld\ttabs"
        assert summarise.clean_text(text) == text

    def test_empty_string(self):
        assert summarise.clean_text("") == ""

    def test_only_control_chars_becomes_empty(self):
        assert summarise.clean_text("\x00\x01\x1f") == ""


# ---------------------------------------------------------------------------
# Pure functions: truncate_words
# ---------------------------------------------------------------------------

class TestTruncateWords:
    def test_under_limit_returns_unchanged(self):
        text = " ".join(["word"] * 50)
        assert summarise.truncate_words(text, max_words=600) == text

    def test_at_limit_returns_unchanged(self):
        text = " ".join(["word"] * 600)
        assert summarise.truncate_words(text, max_words=600) == text

    def test_over_limit_truncates(self):
        text = " ".join(["word"] * 700)
        result = summarise.truncate_words(text, max_words=600)
        assert len(result.split()) == 600

    def test_truncated_text_is_prefix(self):
        words = [str(i) for i in range(800)]
        text = " ".join(words)
        result = summarise.truncate_words(text, max_words=100)
        assert result == " ".join(words[:100])


# ---------------------------------------------------------------------------
# File I/O: load_config
# ---------------------------------------------------------------------------

class TestLoadConfig:
    def test_load_config_returns_dict(self, tmp_path, sample_config):
        cfg_file = tmp_path / "pipeline.yaml"
        cfg_file.write_text(yaml.dump(sample_config))
        result = summarise.load_config(str(cfg_file))
        assert result == sample_config

    def test_load_config_nested_keys(self, tmp_path):
        data = {"summarise": {"min_words": 200, "output_file": "out.jsonl"}}
        cfg_file = tmp_path / "pipeline.yaml"
        cfg_file.write_text(yaml.dump(data))
        result = summarise.load_config(str(cfg_file))
        assert result["summarise"]["min_words"] == 200


# ---------------------------------------------------------------------------
# File I/O: load_checkpoint
# ---------------------------------------------------------------------------

class TestLoadCheckpoint:
    def test_no_file_returns_empty_set(self, tmp_path):
        path = tmp_path / "does_not_exist.jsonl"
        result = summarise.load_checkpoint(path)
        assert result == set()

    def test_with_urls_returns_set(self, sample_jsonl):
        result = summarise.load_checkpoint(sample_jsonl)
        assert result == {
            "http://council.example/doc1.pdf",
            "http://council.example/doc2.pdf",
        }

    def test_skips_malformed_lines(self, tmp_path):
        path = tmp_path / "bad.jsonl"
        path.write_text(
            '{"url": "http://good.example/doc.pdf"}\n'
            'NOT JSON AT ALL\n'
            '{"url": "http://good2.example/doc.pdf"}\n'
        )
        result = summarise.load_checkpoint(path)
        assert result == {
            "http://good.example/doc.pdf",
            "http://good2.example/doc.pdf",
        }

    def test_skips_entries_without_url(self, tmp_path):
        path = tmp_path / "no_url.jsonl"
        path.write_text('{"title": "no url here"}\n{"url": "http://has.url/"}\n')
        result = summarise.load_checkpoint(path)
        assert result == {"http://has.url/"}


# ---------------------------------------------------------------------------
# File I/O: _find_latest_csv
# ---------------------------------------------------------------------------

class TestFindLatestCsv:
    def test_returns_most_recent(self, tmp_path):
        older = tmp_path / "a.csv"
        newer = tmp_path / "b.csv"
        older.write_text("data")
        newer.write_text("data")
        # Touch newer to ensure it has a later mtime
        import time
        time.sleep(0.01)
        newer.touch()
        result = summarise._find_latest_csv(str(tmp_path))
        assert Path(result) == newer

    def test_no_csv_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            summarise._find_latest_csv(str(tmp_path))


# ---------------------------------------------------------------------------
# Async: call_api
# ---------------------------------------------------------------------------

class TestCallApi:
    async def test_returns_parsed_json(self, mock_openai_client):
        result = await summarise.call_api(
            mock_openai_client,
            url="http://test.example/doc.pdf",
            authority="Test Council",
            text=" ".join(["word"] * 150),
        )
        assert result["title"] == "Test Budget 2024"
        assert result["year"] == 2024
        assert "summary" in result
        assert "themes" in result
        assert result["url"] == "http://test.example/doc.pdf"
        assert result["authority"] == "Test Council"

    async def test_injects_url_and_authority(self, mock_openai_client):
        result = await summarise.call_api(
            mock_openai_client,
            url="http://example.com/unique.pdf",
            authority="Special Council",
            text=" ".join(["word"] * 150),
        )
        assert result["url"] == "http://example.com/unique.pdf"
        assert result["authority"] == "Special Council"

    async def test_cleans_and_truncates_input(self, mock_openai_client):
        """Verify control chars are stripped and text is truncated before sending."""
        dirty_long = "\x00" + " ".join(["word"] * 700)
        await summarise.call_api(
            mock_openai_client,
            url="http://test.example/doc.pdf",
            authority="Test",
            text=dirty_long,
        )
        call_kwargs = mock_openai_client.chat.completions.create.call_args
        user_content = call_kwargs[1]["messages"][1]["content"]
        # Control char should be stripped
        assert "\x00" not in user_content
        # Word count in the text portion should be ≤ default max_words (600)
        text_portion = user_content.split("\n\n", 1)[1]
        assert len(text_portion.split()) <= 600


# ---------------------------------------------------------------------------
# Async: worker
# ---------------------------------------------------------------------------

class TestWorker:
    async def test_writes_jsonl_on_success(self, mock_openai_client, tmp_path):
        out_path = tmp_path / "out.jsonl"
        counters = {"done": 0, "errors": 0, "rate_limits": 0}
        queue = asyncio.Queue()
        await queue.put(("http://test.example/doc.pdf", "Council A", " ".join(["word"] * 150)))
        await queue.put(None)  # sentinel

        with open(out_path, "a") as fh:
            await summarise.worker(queue, mock_openai_client, fh, counters)

        lines = [l for l in out_path.read_text().splitlines() if l.strip()]
        assert len(lines) == 1
        obj = json.loads(lines[0])
        assert obj["url"] == "http://test.example/doc.pdf"
        assert counters["done"] == 1
        assert counters["errors"] == 0

    async def test_handles_429_with_retry(self, tmp_path):
        """First call raises 429, second call succeeds."""
        good_response = MagicMock()
        good_response.choices = [MagicMock()]
        good_response.choices[0].message.content = json.dumps({
            "title": "Retry Doc", "year": 2023,
            "summary": "Retried successfully.", "themes": ["Housing"],
        })

        call_count = 0

        async def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("429 Too Many Requests")
            return good_response

        client = MagicMock()
        client.chat.completions.create = AsyncMock(side_effect=side_effect)

        out_path = tmp_path / "out.jsonl"
        counters = {"done": 0, "errors": 0, "rate_limits": 0}
        queue = asyncio.Queue()
        await queue.put(("http://test.example/retry.pdf", "Council B", " ".join(["word"] * 150)))
        await queue.put(None)

        with patch("asyncio.sleep", new_callable=AsyncMock):
            with open(out_path, "a") as fh:
                await summarise.worker(queue, client, fh, counters)

        assert counters["done"] == 1
        assert counters["rate_limits"] == 1
        assert counters["errors"] == 0

    async def test_non_429_error_increments_error_counter(self, tmp_path):
        """Non-429 exception increments errors, doesn't crash."""
        client = MagicMock()
        client.chat.completions.create = AsyncMock(
            side_effect=ValueError("some unexpected error")
        )

        out_path = tmp_path / "out.jsonl"
        counters = {"done": 0, "errors": 0, "rate_limits": 0}
        queue = asyncio.Queue()
        await queue.put(("http://test.example/fail.pdf", "Council C", " ".join(["word"] * 150)))
        await queue.put(None)

        with open(out_path, "a") as fh:
            await summarise.worker(queue, client, fh, counters)

        assert counters["errors"] == 1
        assert counters["done"] == 0
        # File should be empty (no successful writes)
        assert out_path.read_text().strip() == ""


# ---------------------------------------------------------------------------
# Integration tests (mocked API, real file I/O)
# ---------------------------------------------------------------------------

class TestRunIntegration:
    async def test_run_end_to_end(self, sample_csv, tmp_path, sample_config, mock_openai_client):
        """Small CSV with mocked API produces correct JSONL output."""
        output_file = str(tmp_path / "output.jsonl")
        sample_config["summarise"]["output_file"] = output_file
        sample_config["summarise"]["min_words"] = 100

        with patch.object(summarise, "build_client", return_value=mock_openai_client):
            await summarise.run(sample_config, str(sample_csv), output_file, concurrency=2)

        lines = [l for l in Path(output_file).read_text().splitlines() if l.strip()]
        # 3 long docs should be processed (2 short ones skipped)
        assert len(lines) == 3
        for line in lines:
            obj = json.loads(line)
            assert "url" in obj
            assert "authority" in obj
            assert "title" in obj

    async def test_run_skips_checkpointed_urls(
        self, sample_csv, sample_jsonl, sample_config, mock_openai_client
    ):
        """URLs already in output JSONL are skipped."""
        # sample_jsonl has doc1 and doc2 pre-done; only doc3 should be processed
        with patch.object(summarise, "build_client", return_value=mock_openai_client):
            await summarise.run(
                sample_config,
                str(sample_csv),
                str(sample_jsonl),
                concurrency=1,
            )

        all_lines = [l for l in sample_jsonl.read_text().splitlines() if l.strip()]
        # 2 pre-existing + 1 new
        assert len(all_lines) == 3
        urls = {json.loads(l)["url"] for l in all_lines}
        assert "http://council.example/doc1.pdf" in urls
        assert "http://council.example/doc2.pdf" in urls
        assert "http://council.example/doc3.pdf" in urls

    async def test_run_skips_short_docs(self, tmp_path, sample_config, mock_openai_client):
        """Documents below min_words threshold are not processed."""
        csv_path = tmp_path / "short.csv"
        csv_path.write_text(
            'Document Link,Authority Name,text\n'
            '"http://test.example/short.pdf","Test Council","just five words here"\n'
        )
        output_file = str(tmp_path / "output.jsonl")
        sample_config["summarise"]["min_words"] = 100

        with patch.object(summarise, "build_client", return_value=mock_openai_client):
            await summarise.run(sample_config, str(csv_path), output_file, concurrency=1)

        output_path = Path(output_file)
        if output_path.exists():
            lines = [l for l in output_path.read_text().splitlines() if l.strip()]
            assert len(lines) == 0
        # If file doesn't exist at all, that's also correct (nothing to write)
