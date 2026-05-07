"""Tests for eval_themes.py."""
import importlib.util
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

# Load eval_themes as a module (not a package)
_spec = importlib.util.spec_from_file_location(
    "eval_themes",
    Path(__file__).parent.parent / "eval_themes.py",
)
eval_themes = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(eval_themes)

# Also load summarise for cross-module THEMES comparison
_s_spec = importlib.util.spec_from_file_location(
    "summarise",
    Path(__file__).parent.parent / "stages" / "02_summarise.py",
)
summarise = importlib.util.module_from_spec(_s_spec)
_s_spec.loader.exec_module(summarise)


# ---------------------------------------------------------------------------
# clean_text
# ---------------------------------------------------------------------------

class TestCleanText:
    def test_strips_control_chars(self):
        dirty = "hello\x00\x01\x02\x03\x04\x05\x06\x07\x08 world\x0b\x0c\x0e\x1f"
        assert eval_themes.clean_text(dirty) == "hello world"

    def test_preserves_normal_chars(self):
        text = "Hello £10\nworld\ttabs"
        assert eval_themes.clean_text(text) == text

    def test_empty_string(self):
        assert eval_themes.clean_text("") == ""

    def test_carriage_return_preserved(self):
        """\\r (0x0d) is NOT in the stripped range and should be kept."""
        text = "line1\r\nline2"
        assert eval_themes.clean_text(text) == text


# ---------------------------------------------------------------------------
# truncate_words
# ---------------------------------------------------------------------------

class TestTruncateWords:
    def test_under_limit_returns_unchanged(self):
        text = " ".join(["word"] * 50)
        assert eval_themes.truncate_words(text, max_words=1000) == text

    def test_at_limit_returns_unchanged(self):
        text = " ".join(["word"] * 1000)
        assert eval_themes.truncate_words(text, max_words=1000) == text

    def test_over_limit_truncates(self):
        text = " ".join(["word"] * 1500)
        result = eval_themes.truncate_words(text, max_words=1000)
        assert len(result.split()) == 1000

    def test_default_max_is_1000(self):
        """eval_themes default max_words is 1000 (differs from summarise's 600)."""
        text = " ".join(["word"] * 1200)
        result = eval_themes.truncate_words(text)
        assert len(result.split()) == 1000


# ---------------------------------------------------------------------------
# classify (async)
# ---------------------------------------------------------------------------

class TestClassify:
    def _make_client(self, themes: list):
        content = json.dumps({"themes": themes})
        message = MagicMock()
        message.content = content
        choice = MagicMock()
        choice.message = message
        response = MagicMock()
        response.choices = [choice]

        client = MagicMock()
        client.chat = MagicMock()
        client.chat.completions = MagicMock()
        client.chat.completions.create = AsyncMock(return_value=response)
        return client

    async def test_classify_returns_themes(self):
        themes = ["Housing", "Local government finance"]
        client = self._make_client(themes)
        result = await eval_themes.classify(client, " ".join(["word"] * 200))
        assert result["themes"] == themes

    async def test_classify_handles_single_theme(self):
        client = self._make_client(["Education"])
        result = await eval_themes.classify(client, " ".join(["word"] * 200))
        assert result["themes"] == ["Education"]

    async def test_classify_passes_text_to_api(self):
        client = self._make_client(["Housing"])
        sample_text = " ".join(["unique_word"] * 100)
        await eval_themes.classify(client, sample_text)
        call_kwargs = client.chat.completions.create.call_args[1]
        user_content = call_kwargs["messages"][1]["content"]
        assert "unique_word" in user_content

    async def test_classify_strips_control_chars(self):
        client = self._make_client(["Housing"])
        dirty_text = "\x00" + " ".join(["word"] * 100)
        await eval_themes.classify(client, dirty_text)
        call_kwargs = client.chat.completions.create.call_args[1]
        user_content = call_kwargs["messages"][1]["content"]
        assert "\x00" not in user_content


# ---------------------------------------------------------------------------
# Cross-module THEMES consistency
# ---------------------------------------------------------------------------

class TestThemesConsistency:
    def test_themes_list_identical_in_both_files(self):
        """THEMES constant must be the same in eval_themes and 02_summarise."""
        assert eval_themes.THEMES == summarise.THEMES, (
            "THEMES lists differ between eval_themes.py and stages/02_summarise.py.\n"
            f"eval_themes: {eval_themes.THEMES}\n"
            f"summarise:   {summarise.THEMES}"
        )

    def test_themes_contains_uncategorised(self):
        assert "Uncategorised" in eval_themes.THEMES

    def test_themes_has_no_duplicates(self):
        assert len(eval_themes.THEMES) == len(set(eval_themes.THEMES))

    def test_themes_non_empty(self):
        assert len(eval_themes.THEMES) > 0
