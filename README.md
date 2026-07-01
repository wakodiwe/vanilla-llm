# VanillaLLM

A lightweight, zero-bloat Python wrapper for OpenAI-compatible LLM APIs. Built with `httpx`, no heavy frameworks required.

## What is this?

VanillaLLM is a single-file (~200 lines) module that wraps the OpenAI-compatible chat completion endpoint:

- **Zero latency overhead** – direct HTTP, no abstractions
- **Streaming support** – watch tokens appear in real time
- **Automatic retries** – exponential backoff for flaky connections
- **Interactive REPL** – persistent history with up-arrow recall
- **Full control** – every request parameter is yours to tweak

## Quick start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run a quick chat:**
   ```bash
   python vllm.py "What is the meaning of life?"
   ```

3. **Or start the interactive chatbot:**
   ```bash
   python vllm.py
   ```

## Usage

### One-shot

```bash
python vllm.py "Hello!"
```

### Interactive REPL

```bash
python vllm.py
```

### Custom server

```bash
python vllm.py --url http://localhost:11434/v1 --model llama3 "Hi"
```

### Piped input

```bash
echo "Hello" | vllm.py
cat file.txt | vllm.py
```

### As library

```python
from vllm import VanillaLLM

with VanillaLLM(base_url="http://localhost:8080/v1", api_key="sk-...") as llm:
    response = llm.ask([{"role": "user", "content": "Hello!"}])
```

## Configuration

Defaults connect to a local server:

```python
BASE_URL = "http://127.0.0.1:8080/v1"   # llama.cpp / llamafile
MODEL = "Llama-3.2-3B-Instruct-Q4_K_M"
```

## API Key

The default API key is `"dummy"` which works for Ollama/llama.cpp. Pass a real API key for OpenAI or other services that require authentication.

## Project structure

```
vanilla-llm/
├── vllm.py              # Single file: class + CLI + REPL
├── models.md            # GGUF model recommendations
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

## Dependencies

See `requirements.txt`: `httpx`, `prompt-toolkit`.

## License

BSD License – see [`LICENSE`](LICENSE) for details.