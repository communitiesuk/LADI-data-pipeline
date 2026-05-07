"""Tests for ladi/apim.py."""
import json
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from ladi.apim import _get_token, build_client


# ---------------------------------------------------------------------------
# _get_token
# ---------------------------------------------------------------------------

def test_get_token_success(apim_env):
    """mock subprocess returning valid JSON — token extracted correctly."""
    fake_output = json.dumps({"accessToken": "test-bearer-token-abc"})
    mock_result = MagicMock()
    mock_result.stdout = fake_output

    with patch("ladi.apim.subprocess.run", return_value=mock_result) as mock_run:
        token = _get_token()

    assert token == "test-bearer-token-abc"
    mock_run.assert_called_once()
    call_args = mock_run.call_args
    assert "az" in call_args[0][0]
    assert "get-access-token" in call_args[0][0]


def test_get_token_az_not_logged_in(apim_env):
    """subprocess raises CalledProcessError when az CLI fails."""
    with patch(
        "ladi.apim.subprocess.run",
        side_effect=subprocess.CalledProcessError(1, "az"),
    ):
        with pytest.raises(subprocess.CalledProcessError):
            _get_token()


def test_get_token_uses_scope_env_var(apim_env, monkeypatch):
    """Token scope env var is forwarded to the az CLI command."""
    monkeypatch.setenv("LADI_APIM_TOKEN_SCOPE", "https://custom.scope/.default")
    fake_output = json.dumps({"accessToken": "scope-test-token"})
    mock_result = MagicMock()
    mock_result.stdout = fake_output

    with patch("ladi.apim.subprocess.run", return_value=mock_result) as mock_run:
        token = _get_token()

    assert token == "scope-test-token"
    cmd = mock_run.call_args[0][0]
    assert "https://custom.scope/.default" in cmd


# ---------------------------------------------------------------------------
# build_client
# ---------------------------------------------------------------------------

def test_build_client_returns_async_openai(apim_env):
    """build_client returns an AsyncOpenAI configured with APIM env vars."""
    from openai import AsyncOpenAI

    with patch("ladi.apim._get_token", return_value="fake-token"):
        client = build_client()

    assert isinstance(client, AsyncOpenAI)
    assert str(client.base_url).rstrip("/") == "https://apim.example.com/openai"


def test_build_client_sets_subscription_header(apim_env):
    """build_client sets Ocp-Apim-Subscription-Key in default headers."""
    with patch("ladi.apim._get_token", return_value="fake-token"):
        client = build_client()

    assert client.default_headers.get("Ocp-Apim-Subscription-Key") == "fake-sub-key"
