"""
A lightweight, zero‑bloat wrapper for OpenAI‑compatible LLM APIs.
"""

import json
import logging
import sys
import time
from typing import Any, Dict, List, Optional

from prompt_toolkit import prompt

import httpx


logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


BASE_URL = "http://127.0.0.1:8080/v1"
MODEL = "Llama-3.2-3B-Instruct-Q4_K_M"


class VanillaLLM:
    """A lightweight, zero‑bloat wrapper for OpenAI‑compatible LLM APIs."""

    def __init__(
        self,
        base_url: str = BASE_URL,
        api_key: str = "dummy",
        timeout: int = 120,
        max_retries: int = 3,
    ):
        if api_key == "dummy":
            logger.warning(
                "Using dummy API key. This works for Ollama/llama.cpp but "
                "will fail for OpenAI. Pass a real API key for other services."
            )
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}", "user-agent": "vanilla-llm"},
            timeout=httpx.Timeout(timeout, connect=5.0),
        )
        self._max_retries = max_retries
        self._closed = False

    def __enter__(self) -> "VanillaLLM":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def close(self) -> None:
        """Close the HTTP client and release resources."""
        if not self._closed:
            self._client.close()
            self._closed = True

    def _post(self, endpoint: str, data: Dict[str, Any]) -> httpx.Response:
        last_exception: Optional[Exception] = None
        for attempt in range(self._max_retries):
            try:
                response = self._client.post(endpoint, json=data)
                response.raise_for_status()
                return response
            except (httpx.ConnectError, httpx.ReadTimeout, httpx.HTTPStatusError) as e:
                last_exception = e
                wait_time = 2 ** attempt
                logger.debug(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)

        raise RuntimeError(
            f"All {self._max_retries} attempts failed. Last error: {last_exception}"
        ) from last_exception

    def ask(
        self,
        messages: List[Dict[str, str]],
        *,
        model: str = MODEL,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        verbose: bool = True,
        **kw: Any,
    ) -> str:
        """Send a chat completion request and return the assistant's response."""
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
            **kw,
        }
        full: List[str] = []
        if verbose:
            print("Assistant: ", end="", flush=True)

        with self._client.stream("POST", "/chat/completions", json=payload) as stream_resp:
            stream_resp.raise_for_status()
            for line in stream_resp.iter_lines():
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
                except (json.JSONDecodeError, KeyError, IndexError, TypeError) as e:
                    logger.debug(f"Skipped malformed chunk: {e}")
                    continue

        print()
        return "".join(full)


def main() -> None:
    """CLI entry point."""
    with VanillaLLM(BASE_URL) as llm:
        prompt_text = " ".join(sys.argv[1:]) or prompt("> ", vi_mode=True)
        llm.ask([{"role": "user", "content": prompt_text}])


def get_llm(base_url: str = BASE_URL, api_key: str = "dummy", **kwargs: Any) -> VanillaLLM:
    """Factory function to create VanillaLLM instances with proper resource management."""
    return VanillaLLM(base_url=base_url, api_key=api_key, **kwargs)


llm: VanillaLLM = VanillaLLM(BASE_URL)
logger.warning(
    "Module-level 'llm' singleton is deprecated. Use 'with VanillaLLM(...) as llm:' "
    "or get_llm() for proper resource management."
)


if __name__ == "__main__":
    main()
