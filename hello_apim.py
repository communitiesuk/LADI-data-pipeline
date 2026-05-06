"""Quick smoke test — sends 'hello' to the APIM LLM and prints the response."""
import asyncio
import os
from ladi.apim import build_client

async def main():
    client = build_client()
    resp = await client.chat.completions.create(
        model=os.environ["LADI_APIM_DEPLOYMENT_ID"],
        messages=[{"role": "user", "content": "Say hello in one sentence."}],
        max_tokens=50,
    )
    print(resp.choices[0].message.content)

asyncio.run(main())
