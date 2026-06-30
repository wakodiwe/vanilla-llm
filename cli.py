"""
ACMD vanilla-cli.py BufWritePost Term ./app.py
A lightweight, zero‑bloat wrapper for OpenAI‑compatible LLM APIs.
"""

import json
import sys
import time
from typing import List, Dict, Iterator

import httpx

VERSION = "0.8.15"
BASE_URL = "http://127.0.0.1:8080/v1"
MODEL = "Llama-3.2-3B-Instruct-Q4_K_M"


class VanillaLLM:
    def __init__(self, base_url: str = BASE_URL, api_key: str = "dummy", timeout: int = 120, retries: int = 3):
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}", "user-agent": f"hej {VERSION}"},
            timeout=httpx.Timeout(timeout, connect=5.0),
        )
        self._retries = retries

    def _post(self, endpoint: str, data: dict) -> httpx.Response:
        last_err = None
        for attempt in range(self._retries):
            try:
                r = self._client.post(endpoint, json=data)
                r.raise_for_status()
                return r
            except (httpx.ConnectError, httpx.ReadTimeout, httpx.HTTPStatusError) as e:
                last_err = e
                time.sleep(2 ** attempt)
                print(f"Attempt {attempt + 1} failed. Retrying...")
        raise last_err

    def _build_payload(self, messages, model, temperature, max_tokens, stream, **kw):
        return {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
            **kw,
        }

    def chat(self, messages: List[Dict[str, str]], model: str = MODEL, **kw) -> str:
        payload = self._build_payload(messages, model, 0.7, 1024, False, **kw)
        data = self._post("/chat/completions", payload).json()
        return data["choices"][0].get("message", {}).get("content", "")

    def stream(self, messages: List[Dict[str, str]], model: str = MODEL, **kw) -> Iterator[str]:
        payload = self._build_payload(messages, model, 0.7, 1024, True, **kw)
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
                    if "content" in delta:
                        yield delta["content"]
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue

    def stream_print(self, messages, *, model: str = MODEL, **kw) -> str:
        print("Assistant: ", end="", flush=True)
        full = []
        for token in self.stream(messages, model, **kw):
            if token:
                print(token, end="", flush=True)
                full.append(token)
        print()
        return "".join(full)

    def close(self):
        self._client.close()

    __enter__ = lambda self, *a: self
    __exit__ = lambda self, *a: self.close() or None


llm = VanillaLLM(BASE_URL)

if __name__ == "__main__":
    try:
        prompt = " ".join(sys.argv[1:]) or input("> ")
        llm.stream_print([{"role": "user", "content": prompt}])
    finally:
        llm.close()
