"""
ACMD vanilla-cli.py BufWritePost Term ./app.py
A lightweight, zero‑bloat wrapper for OpenAI‑compatible LLM APIs.
"""

import json
import sys
import time

from prompt_toolkit import prompt

import httpx


BASE_URL = "http://127.0.0.1:8080/v1"
MODEL = "Llama-3.2-3B-Instruct-Q4_K_M"


class VanillaLLM:
    def __init__(self, base_url: str = BASE_URL, api_key: str = "dummy", timeout: int = 120, retries: int = 3):
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}", "user-agent": "hej cli"},
            timeout=httpx.Timeout(timeout, connect=5.0),
        )
        self._retries = retries

    def _post(self, endpoint: str, data: dict) -> httpx.Response:
        err = None
        for attempt in range(self._retries):
            try:
                r = self._client.post(endpoint, json=data)
                r.raise_for_status()
                return r
            except (httpx.ConnectError, httpx.ReadTimeout, httpx.HTTPStatusError) as e:
                err = e
                time.sleep(2 ** attempt)
                print(f"Attempt {attempt + 1} failed. Retrying...")
        raise err

    def ask(self, messages, *, model: str = MODEL, temperature: float = 0.7, max_tokens: int = 1024, verbose: bool = True, **kw) -> str:
        payload = {
            "model": model, "messages": messages, "temperature": temperature,
            "max_tokens": max_tokens, "stream": True, **kw,
        }
        full = []
        if verbose:
            print("Assistant: ", end="", flush=True)
        with self._client.stream("POST", "/chat/completions", json=payload) as s:
            s.raise_for_status()
            for line in s.iter_lines():
                if not line.startswith("data: "):
                    continue
                chunk = line[6:]
                if chunk == "[DONE]":
                    break
                try:
                    delta = json.loads(chunk)["choices"][0].get("delta", {})
                    if delta and "content" in delta:
                        token = delta["content"]
                        if token is not None:
                            print(token, end="", flush=True)
                            full.append(token)
                except (json.JSONDecodeError, KeyError, IndexError, TypeError):
                    continue
        print()
        return "".join(full)

    def close(self):
        self._client.close()


llm = VanillaLLM(BASE_URL)

if __name__ == "__main__":
    try:
        # prompt = " ".join(sys.argv[1:]) or input("> ")
        prompt = " ".join(sys.argv[1:]) or prompt("> ", vi_mode=True)
        llm.ask([{"role": "user", "content": prompt}])
    finally:
        llm.close()
