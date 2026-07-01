#!/usr/bin/env python
"""
VanillaLLM - A lightweight, zero‑bloat wrapper for OpenAI‑compatible LLM APIs.
"""

import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional

import httpx
from prompt_toolkit import PromptSession, prompt
from prompt_toolkit.history import FileHistory


logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


BASE_URL = "http://127.0.0.1:8080/v1"
MODEL = "Llama-3.2-3B-Instruct-Q4_K_M"
HISTORY_FILE = os.path.expanduser("~/.vanillachat_history")


class VanillaLLM:
    """A lightweight, zero‑bloat wrapper for OpenAI‑compatible LLM APIs."""

    def __init__(
        self,
        base_url: str = BASE_URL,
        api_key: str = "dummy",
        timeout: int = 120,
        max_retries: int = 3,
        warn_api_key: bool = True,
    ):
        if api_key == "dummy" and warn_api_key and not base_url.startswith("http://127.0.0.1") and not base_url.startswith("http://localhost"):
            logger.warning(
                "Using dummy API key. This may fail for non-local servers. "
                "Pass a real API key if needed."
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


def chat(base_url: str = BASE_URL, model: str = MODEL) -> None:
    """Interactive chatbot REPL."""
    conv: List[Dict[str, str]] = [{"role": "system", "content": "You are a helpful assistant."}]
    session = PromptSession(history=FileHistory(HISTORY_FILE), vi_mode=True)

    with VanillaLLM(base_url) as llm:
        print("Welcome to VanillaChat! Type 'exit' or press Ctrl+D to quit.")
        while True:
            try:
                user = session.prompt("You: ")
            except EOFError:
                print("\nBye.")
                break
            if user.lower() in ("exit", "quit"):
                print("Bye.")
                break
            conv.append({"role": "user", "content": user})
            conv.append({"role": "assistant", "content": llm.ask(conv, model=model)})


def main() -> None:
    """CLI entry point - interactive if no args, one-shot otherwise."""
    base_url = BASE_URL
    model = MODEL

    args = sys.argv[1:]
    if "--help" in args or "-h" in args:
        print("Usage: vllm.py [options] [message]")
        print("       vllm.py                 # Interactive chat")
        print("       vllm.py 'Hello!'       # One-shot")
        print("       echo 'Hi' | vllm.py     # Piped input")
        print("  --url URL    Base URL (default: http://127.0.0.1:8080/v1)")
        print("  --model M   Model name")
        sys.exit(0)

    i = 0
    while i < len(args):
        if args[i] == "--url" and i + 1 < len(args):
            base_url = args[i + 1]
            i += 2
        elif args[i] == "--model" and i + 1 < len(args):
            model = args[i + 1]
            i += 2
        else:
            break

    message = " ".join(args[i:])
    piped = ""

    if not sys.stdin.isatty():
        piped = sys.stdin.read().strip()

    with VanillaLLM(base_url) as llm:
        if message or piped:
            prompt = piped or message
            llm.ask([{"role": "user", "content": prompt}], model=model)
        else:
            chat(base_url, model)


def get_llm(base_url: str = BASE_URL, api_key: str = "dummy", **kwargs: Any) -> VanillaLLM:
    """Factory function to create VanillaLLM instances."""
    return VanillaLLM(base_url=base_url, api_key=api_key, **kwargs)


def __getattr__(name: str) -> Any:
    if name == "llm":
        logger.warning(
            "Module-level 'llm' is deprecated. Use 'with VanillaLLM(...) as llm:' "
            "or get_llm() for proper resource management."
        )
        return VanillaLLM(BASE_URL)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if __name__ == "__main__":
    main()
