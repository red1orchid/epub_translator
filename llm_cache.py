import hashlib
import json
import math
import os
import time
from typing import Callable, Dict, List, Optional

from llm_provider import LLMProvider

__all__ = ["LLMResponseCache", "CachedLLMProvider", "estimate_tokens", "format_tokens"]


def estimate_tokens(text: str) -> int:
    """Rough char-to-token estimate (same heuristic as batching)."""
    return len(text) // 3


def format_tokens(n_tokens: int) -> str:
    """Round up to the nearest 1k for display, e.g. '~20k tokens'."""
    return f"~{max(1, math.ceil(n_tokens / 1000))}k tokens"


class LLMResponseCache:
    """Disk cache of raw LLM responses, keyed by the exact request.

    Because batching is deterministic (same input file, same settings produce
    the same requests), a retried chapter replays already-received responses
    from disk instead of calling the LLM again. Entries can be edited on disk
    (e.g. to fix a broken delimiter) and the retry will use the edited text.
    """

    def __init__(self, cache_dir: str, namespace: str):
        self.cache_dir = cache_dir
        self.namespace = namespace  # e.g. "openai:gpt-5.1" — model change = fresh cache
        os.makedirs(cache_dir, exist_ok=True)

    def key_for(self, messages: List[Dict[str, str]]) -> str:
        payload = json.dumps(
            {"ns": self.namespace, "messages": messages},
            ensure_ascii=False, sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]

    def _path(self, key: str) -> str:
        return os.path.join(self.cache_dir, f"resp_{key}.json")

    def load(self, key: str) -> Optional[dict]:
        p = self._path(key)
        if not os.path.exists(p):
            return None
        try:
            with open(p) as f:
                return json.load(f)
        except Exception:
            return None

    def save(self, key: str, messages: List[Dict[str, str]], response: str) -> None:
        entry = {
            "namespace": self.namespace,
            "messages": messages,
            "response": response,
            "saved_at": time.time(),
        }
        with open(self._path(key), "w") as f:
            json.dump(entry, f, ensure_ascii=False)

    def update_response(self, key: str, new_response: str) -> bool:
        entry = self.load(key)
        if entry is None:
            return False
        entry["response"] = new_response
        entry["edited_at"] = time.time()
        with open(self._path(key), "w") as f:
            json.dump(entry, f, ensure_ascii=False)
        return True

    def delete(self, key: str) -> None:
        try:
            os.remove(self._path(key))
        except FileNotFoundError:
            pass


class CachedLLMProvider(LLMProvider):
    """Wraps any provider: serves identical requests from disk, saves new
    responses the instant they arrive (before any parsing can fail).

    on_event(kind, key, est_tokens) is called with kind one of:
      "cache_hit" — request served from disk, no API cost (tokens = prompt size)
      "llm_call"  — request is being sent to the LLM (tokens = prompt size)
      "llm_done"  — response arrived and was saved (tokens = response size)
    """

    def __init__(self, inner: LLMProvider, cache: LLMResponseCache,
                 on_event: Optional[Callable[[str, str, int], None]] = None):
        self.inner = inner
        self.cache = cache
        self.on_event = on_event

    def chat(self, messages: List[Dict[str, str]]) -> str:
        key = self.cache.key_for(messages)
        prompt_tokens = estimate_tokens("".join(m.get("content", "") for m in messages))

        cached = self.cache.load(key)
        if cached is not None:
            self._emit("cache_hit", key, prompt_tokens)
            return cached["response"]

        self._emit("llm_call", key, prompt_tokens)
        response = self.inner.chat(messages)
        self.cache.save(key, messages, response)
        self._emit("llm_done", key, estimate_tokens(response))
        return response

    def _emit(self, kind: str, key: str, tokens: int) -> None:
        if self.on_event:
            self.on_event(kind, key, tokens)
