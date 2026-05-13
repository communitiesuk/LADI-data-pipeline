"""APIM client. Config lives in .env — see .env.example."""
import json
import os
import subprocess

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()


def _get_token() -> str:
    """Get Azure AD bearer token via az CLI."""
    result = subprocess.run(
        ["az", "account", "get-access-token",
         "--scope", os.environ["LADI_APIM_TOKEN_SCOPE"],
         "--output", "json"],
        check=True, capture_output=True, text=True,
    )
    return json.loads(result.stdout)["accessToken"]


def build_client() -> AsyncOpenAI:
    """Build client for GPT chat completions."""
    return AsyncOpenAI(
        base_url=os.environ["LADI_APIM_BASE_URL"],
        api_key=_get_token(),
        default_headers={"Ocp-Apim-Subscription-Key": os.environ["LADI_APIM_SUBSCRIPTION_KEY"]},
        default_query={"api-version": os.environ["LADI_APIM_API_VERSION"]},
    )


def build_embedding_client() -> AsyncOpenAI:
    """Build client for text-embedding-3-large."""
    return AsyncOpenAI(
        base_url=os.environ["LADI_APIM_EMBEDDING_URL"],
        api_key=_get_token(),
        default_headers={"Ocp-Apim-Subscription-Key": os.environ["LADI_APIM_SUBSCRIPTION_KEY"]},
        default_query={"api-version": os.environ["LADI_APIM_API_VERSION"]},
    )
