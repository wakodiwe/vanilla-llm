# VanillaLLM

A lightweight, zero-bloat Python wrapper for OpenAI-compatible LLM APIs. Built with `httpx`, no heavy frameworks required.

## What is this?

VanillaLLM is a tiny (~80 lines) class that wraps the OpenAI-compatible chat completion endpoint. It gives you:

- **Zero latency overhead** – direct HTTP, no abstractions
- **Streaming support** – watch tokens appear in real time
- **Automatic retries** – exponential backoff for flaky connections
- **Persistent history** – up-arrow recall in the interactive shell (prompt_toolkit)
- **Full control** – every request parameter is yours to tweak

## Quick start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run a quick chat:**
   ```bash
   python cli.py "What is the meaning of life?"
   ```

3. **Or start the interactive chatbot:**
   ```bash
   python app.py
   ```

## Files

| File | Purpose |
|------|---------|
| `cli.py` | `VanillaLLM` class + one-shot CLI |
| `app.py` | Interactive chatbot shell |
| `vllm/` | Alternate implementation package |
| `models.md` | GGUF model recommendations for 8GB RAM |

## Configuration

The `VanillaLLM` class defaults connect to a local server:

```python
BASE_URL = "http://127.0.0.1:8080/v1"   # llama.cpp / llamafile
# BASE_URL = "http://localhost:11434/v1"  # Ollama
MODEL = "Llama-3.2-3B-Instruct-Q4_K_M"
```

Override when instantiating:

```python
from cli import VanillaLLM

llm = VanillaLLM(base_url="http://your-server:8080/v1", api_key="sk-...")
```

## Project structure

```
vanilla-llm/
├── app.py              # Interactive chatbot (REPL)
├── cli.py              # One-shot CLI + VanillaLLM class
├── vllm/
│   ├── __init__.py     # Package version
│   ├── cli.py          # Alternate implementation (fuller features)
│   └── utils.py        # Utility helpers (reserved)
├── models.md           # GGUF model recommendations
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## Usage

### As context manager (recommended)

```python
from cli import VanillaLLM

with VanillaLLM(base_url="http://localhost:8080/v1", api_key="sk-...") as llm:
    response = llm.ask([{"role": "user", "content": "Hello!"}])
```

### Legacy module-level singleton

```python
from cli import llm
response = llm.ask([{"role": "user", "content": "Hello!"}])
llm.close()
```

**Note:** The module-level `llm` singleton will print a deprecation warning. Use the context manager pattern for proper resource cleanup.

## API Key

The default API key is `"dummy"` which works for Ollama/llama.cpp but will fail for OpenAI. A warning is printed at initialization when using the dummy key. Pass a real API key for services that require authentication.

## Dependencies

See `requirements.txt`. Dependencies: `httpx`, `prompt-toolkit`.

## License

BSD License – see the [`LICENSE`](LICENSE) file for details.
