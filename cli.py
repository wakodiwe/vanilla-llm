"""
A lightweight, zero‑bloat wrapper for OpenAI‑compatible LLM APIs.
"""

from pprint import pprint

import json
import time
from typing import List, Dict, Optional, Generator, Any

import httpx

VERSION = "0.8.15"
BASE_URL = "http://127.0.0.1:8080/v1"

# MODEL = "smollm"
# MODEL = "qwen3-coder"
# MODEL = "llama3.2:latest"
MODEL = "Llama-3.2-3B-Instruct-Q4_K_M"
# models.ini
# nomic-embed-text-latest.gguf
# Phi-3.5-mini-instruct.Q5_K_M.gguf
# Qwen2.5-Coder-0.5B-Q4_K_M.gguf
# qwen3-0.6b.gguf
# qwen3-coder.gguf
# session-ses_0f26.md
# smollm-135m.gguf
# SmolLM2-360M-Instruct-Q4_K_M.gguf
# smollm.gguf


class VanillaLLM:
    """A lightweight, zero‑bloat wrapper for OpenAI‑compatible LLM APIs."""

    def __init__(
        self,
        base_url: str,
        api_key: str = "dummy",
        timeout: int = 120,
        max_retries: int = 3,
    ):
        self.base_url = base_url.rstrip("/")
        self.max_retries = max_retries

        self.client = httpx.Client(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "user-agent": f"hej {VERSION}",
            },
            timeout=httpx.Timeout(timeout, connect=5.0),
        )

    def _post_with_retry(
        self, endpoint: str, json_data: Dict[str, Any]
    ) -> httpx.Response:
        last_exception = None
        for attempt in range(self.max_retries):
            try:
                response = self.client.post(endpoint, json=json_data)
                response.raise_for_status()  # raises on 4xx/5xx
                return response
            except (httpx.ConnectError, httpx.ReadTimeout, httpx.HTTPStatusError) as e:
                last_exception = e
                wait_time = 2**attempt  # 1, 2, 4, 8... seconds
                print(f"Attempt {attempt + 1} failed. Retrying in {wait_time}s...")
                time.sleep(wait_time)
        raise last_exception  # if all retries fail, re-raise the last error

    def chat(
        self,
        messages: List[Dict[str, str]],
        model: str = MODEL,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs,
    ) -> str:
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,  # we'll handle streaming separately
            **kwargs,  # allow extra params like top_p, stop, etc.
        }
        response = self._post_with_retry("/chat/completions", payload)
        data = response.json()
        choice = data["choices"][0]
        if "message" in choice and "content" in choice["message"]:
            return choice["message"]["content"]
        return ""  # fallback in case of unexpected structure

    def stream(
        self,
        messages: List[Dict[str, str]],
        model: str = MODEL,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs,
    ) -> Generator[str, None, None]:
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
            **kwargs,
        }
        with self.client.stream("POST", "/chat/completions", json=payload) as stream:
            stream.raise_for_status()
            for line in stream.iter_lines():
                if not line.startswith("data: "):
                    continue
                chunk_data = line[6:]  # remove "data: " prefix
                if chunk_data == "[DONE]":
                    break
                try:
                    parsed = json.loads(chunk_data)
                    delta = parsed["choices"][0].get("delta", {})
                    if "content" in delta:
                        yield delta["content"]
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue

    def stream_print(
        self,
        messages: List[Dict[str, str]],
        model: str = MODEL,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs,
    ) -> str:
        """Stream and print tokens in real-time, return full response."""
        full = []
        for token in self.stream(messages, model, temperature, max_tokens, **kwargs):
            if token:
                print(token, end="", flush=True)
                full.append(token)
        print()  # newline at end
        return "".join(full)

    def close(self) -> None:
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

llm = VanillaLLM(BASE_URL)

def run(prompt):
    reply = llm.chat([{"role": "user", "content": prompt}])
    print(reply)

def stream(prompt):
    reply = llm.stream_print([{"role": "user", "content": prompt}])

