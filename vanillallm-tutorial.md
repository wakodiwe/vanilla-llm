# VanillaLLM – A Minimalist httpx Wrapper for LLMs

Build your own no-bloat LLM client, learn the hidden magic of HTTP, and never trust a framework blindly again.

---

# Table of Contents

1. [The Why – A Tale of Frameworks and Frustration](#the-why)
2. [Step 0: What You'll Need](#step-0)
3. [Step 1: The Skeleton – A Class That Does Nothing (Yet)](#step-1)
4. [Step 2: The Client – Making Friends with httpx](#step-2)
5. [Step 3: Retry – Because the Internet is Unreliable](#step-3)
6. [Step 4: Ask – The Heartbeat](#step-4)
7. [Step 5: Streaming – Token by Token, Like a Typewriter](#step-5)
8. [Step 6: Cleanup – Don't Leave Connections Hanging](#step-6)
9. [Putting It All Together: Your First Vanilla Chatbot](#putting-it-all-together)
10. [Final Thoughts – Embrace the Vanilla](#final-thoughts)

---

<a name="the-why"></a>

# The Why – A Tale of Frameworks and Frustration

Imagine this: you want to talk to an LLM. You could fire up LangChain, import 15 modules, and watch your code turn into a tangled mess of chains, runnables, and callbacks. Debugging? Good luck tracing through 20 layers of abstraction. Latency? You just added 300 ms of overhead, and your users are starting to yawn.

But what if you could call the API directly? No magic, no hidden retries, no opinionated prompt templates. Just you, `httpx`, and the raw JSON flowing over the wire.

That's **VanillaLLM**.

It's a tiny class (about 80 lines in `cli.py`) that wraps the OpenAI-compatible chat endpoint with:

- **Zero latency overhead** – direct HTTP, no abstractions
- **Streaming support** – watch tokens appear in real time
- **Automatic retries** – exponential backoff for flaky connections
- **Full control** – every request parameter is yours to tweak

And the best part? You're going to build it yourself, step by step. By the end, you'll understand every line and have a superpower: the ability to talk to any LLM without relying on a framework that might break tomorrow.

Ready? Let's roll up our sleeves.

---

<a name="step-0"></a>

# Step 0: What You'll Need

- Python 3.8+ installed (you probably have it)
- `httpx` – the only dependency:
  ```bash
  pip install -r requirements.txt
  ```
- A local LLM server (see [`models.md`](models.md) for setup with [llama.cpp](https://github.com/ggerganov/llama.cpp) or [llamafile](https://github.com/Mozilla-Ocho/llamafile))
- Your favourite code editor – we'll be typing a lot.

We'll use `http://127.0.0.1:8080/v1` as our base URL (llama.cpp / llamafile). If you're using **Ollama**, swap to `http://localhost:11434/v1`. OpenAI, Groq, or any other provider works too – just change the URL and add your API key.

---

<a name="step-1"></a>

# Step 1: The Skeleton – A Class That Does Nothing (Yet)

Open a new file, say `cli.py`, and start with the bare bones:

```python
import json
import sys
import time

import httpx


BASE_URL = "http://127.0.0.1:8080/v1"
MODEL = "Llama-3.2-3B-Instruct-Q4_K_M"


class VanillaLLM:
    def __init__(self, base_url: str = BASE_URL, api_key: str = "dummy",
                 timeout: int = 120, retries: int = 3):
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}", "user-agent": "hej cli"},
            timeout=httpx.Timeout(timeout, connect=5. throttle your patience.0),
        )
        self._retries = retries
```

What's happening?

- `base_url` – the root endpoint of your LLM server. Secret tip: Always strip trailing slashes – some servers are picky.
- `api_key` – for local servers, put `"dummy"`; for real ones, load it from an environment variable.
- `timeout` – total timeout for a request. The `connect=5.0` sub-timeout means "if the server is down, tell me quickly."
- `retries` – how many times to try again on failure.
- `_client` – notice the leading underscore? It's a Python convention for "this is internal, don't touch."

Now let's add the meat.

---

<a name="step-2"></a>

# Step 2: The Client – Making Friends with httpx

In Step 1 we already created the `httpx.Client`, so let's talk about *why*.

**Why `httpx.Client`?**
It reuses a connection pool (keep-alive), which speeds up multiple requests to the same server. Under the hood, it manages HTTP/1.1 and HTTP/2 connections efficiently. This is a huge win over plain `requests` when you're making many calls.

The `connect=5.0` – we separate the connection timeout from the overall read timeout. If the server is down, you'll know in 5 seconds instead of hanging for 120.

We also set the `Authorization` header – for local servers it's ignored, but it's there for compatibility with real providers (OpenAI, Groq, etc.).

---

<a name="step-3"></a>

# Step 3: Retry – Because the Internet is Unreliable

Inside `VanillaLLM`, add the private method that performs a POST request with exponential backoff:

```python
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
```

Secret tips:

- We only retry on connection errors, timeouts, and HTTP status errors. A `400 Bad Request` won't get fixed by retrying, so `raise_for_status()` lets it bubble up immediately.
- Exponential backoff prevents hammering a struggling server. The sleep doubles each attempt (`1s`, `2s`, `4s`...).
- We store `err = None` and assign it in the `except` block so the final `raise err` always has something to throw.

---

<a name="step-4"></a>

# Step 4: Ask – The Heartbeat

This is the main method you'll actually use. It takes a list of messages, streams the response, and returns the assistant's reply.

```python
    def ask(self, messages, *, model: str = MODEL,
            temperature: float = 0.7, max_tokens: int = 1024,
            verbose: bool = True, **kw) -> str:
        payload = {
            "model": model, "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens, "stream": True, **kw,
        }
        full = []
        if verbose:
            print("Assistant: ", end="", flush=True)
        with self._client.stream("POST", "/chat/completions",
                                   json=payload) as s:
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
                        token = delta["content"]
                        print(token, end="", flush=True)
                        full.append(token)
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
        print()
        return "".join(full)
```

Observations:

- `*` in the signature forces `model`, `temperature`, etc. to be keyword-only.
- `stream=True` tells the server to send Server-Sent Events (SSE).
- `**kw` lets you pass extra parameters (`top_p`, `stop`, etc.) without changing the method signature.
- We iterate over SSE lines with `iter_lines()`. Each line looks like: `data: {"choices": [{"delta": {"content": "Hello"}}]}`
- We guard parsing with a broad `except` because malformed chunks happen – better to skip one bad line than crash the whole chat.

Test it! Add this at the bottom of `cli.py`:

```python
llm = VanillaLLM(BASE_URL)

if __name__ == "__main__":
    try:
        prompt = " ".join(sys.argv[1:]) or input("> ")
        llm.ask([{"role": "user", "content": prompt}])
    finally:
        llm.close()
```

Run it:

```bash
python cli.py "What is the meaning of life?"
```

You should see tokens appear one by one – instant gratification!

---

<a name="step-5"></a>

# Step 5: Streaming – Token by Token, Like a Typewriter

Wait – we *already* did streaming in Step 4! That's the beauty of VanillaLLM: the core `ask` method streams by default. No separate `stream()` generator to manage – the tokens print as they arrive, and the final string is returned for you to store, log, or ignore.

If you ever need **silent** mode (no printing), just flip the switch:

```python
reply = llm.ask(messages, verbose=False)
```

Then `reply` contains the full text, quietly.

---

<a name="step-6"></a>

# Step 6: Cleanup – Don't Leave Connections Hanging

Add a `close` method to shut down the HTTP client:

```python
    def close(self):
        self._client.close()
```

And use it safely with a `finally` block (as shown in the test code above), or wrap it in `__enter__`/`__exit__` if you prefer `with` statements.

---

<a name="putting-it-all-together"></a>

# Putting It All Together: Your First Vanilla Chatbot

Create `app.py` in the same directory:

```python
import sys
import os
from cli import llm, MODEL
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory

SYSTEM = "You are a helpful assistant."
HISTORY_FILE = os.path.expanduser("~/.vanillachat_history")

if __name__ == "__main__":
    conv = [{"role": "system", "content": SYSTEM}]
    session = PromptSession(history=FileHistory(HISTORY_FILE), vi_mode=True)

    if len(sys.argv) > 1:
        conv.append({"role": "user", "content": " ".join(sys.argv[1:])})
        llm.ask(conv, model=MODEL)
    else:
        print("Welcome to VanillaChat! Type 'exit' or press Ctrl+D to quit.")
        while True:
            try:
                user = session.prompt("You: ")
            except EOFError:
                print("\nBye."); break
            if user.lower() in ("exit", "quit"):
                print("Bye."); break
            conv.append({"role": "user", "content": user})
            conv.append({"role": "assistant",
                         "content": llm.ask(conv, model=MODEL)})
        llm.close()
```

The `PromptSession` with `FileHistory` gives us arrow-key recall and a persistent history file at `~/.vanillachat_history` — so your previous prompts survive across sessions.

Run the interactive chat:

```bash
python app.py
```

Or fire a one-liner:

```bash
python app.py "Explain quantum computing in one sentence"
```

---

<a name="final-thoughts"></a>

# Final Thoughts – Embrace the Vanilla

You now have a complete, production-ish LLM client in under 80 lines of Python. No LangChain, no OpenAI SDK, no hidden magic. Just `httpx`, JSON, and the raw OpenAI-compatible API.

What you learned:

1. How `httpx.Client` reuses connections for speed.
2. How exponential backoff protects against flaky networks.
3. How Server-Sent Events (SSE) work for streaming LLM responses.
4. How to build a tiny, composable class that does one thing well.

What to try next:

- Add `__enter__` / `__exit__` for `with` statement support.
- Parse `usage` tokens from the final SSE chunk for cost tracking.
- Swap `BASE_URL` and `MODEL` to point at OpenAI, Groq, or any other compatible provider.
- Add a `--system-prompt` CLI flag.

The framework can't break what you understand. Now go build something vanilla.
