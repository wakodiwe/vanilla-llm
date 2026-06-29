from pprint import pprint

import json
import time
from typing import List, Dict, Optional, Generator, Any

import httpx

MODEL = "smollm-135m"
BASE_URL = "http://localhost:11434"


class VanillaLLM:
    """A lightweight, zero‑bloat wrapper for OpenAI‑compatible LLM APIs."""

    def __init__(self, base_url: str, api_key: str = "dummy", 
                 timeout: int = 120, max_retries: int = 3):
        self.base_url = base_url.rstrip("/")
        self.max_retries = max_retries
        # we'll add the client here in a moment...
