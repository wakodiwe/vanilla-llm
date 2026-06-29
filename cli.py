"""
A lightweight, zero‑bloat wrapper for OpenAI‑compatible LLM APIs.
"""

from pprint import pprint

import json
import time
from typing import List, Dict, Optional, Generator, Any

import httpx

VERSION = "0.8.15"
MODEL = "smollm-135m"
BASE_URL = "http://127.0.0.1:8080/v1"


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
                print(attempt)
                response = self.client.post(endpoint, json=json_data)
                response.raise_for_status()  # raises on 4xx/5xx
                return response
            except (httpx.ConnectError, httpx.ReadTimeout, httpx.HTTPStatusError) as e:
                print(attempt)
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


llm = VanillaLLM(BASE_URL)
reply = llm.chat([{"role": "user", "content": "Hi, how are you?"}])
print(reply)
