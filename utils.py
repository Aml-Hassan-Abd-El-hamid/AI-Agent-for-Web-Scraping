"""
Shared helpers used across the scraping agents and orchestrator.

Keep this module dependency-light (standard library only) so every entry point
can import it without pulling in heavy packages or risking circular imports.
"""
import time

# Transient server-side errors worth retrying (Gemini 500/503, rate limits, etc.)
_TRANSIENT_LLM_MARKERS = (
    "500", "503", "internal error", "internal server", "overloaded",
    "unavailable", "deadline", "timeout", "429", "rate limit", "resource exhausted",
)


def _is_transient_llm_error(err: Exception) -> bool:
    """Return True if the error looks like a transient, retryable LLM failure."""
    msg = str(err).lower()
    return any(marker in msg for marker in _TRANSIENT_LLM_MARKERS)


# ─────────────────────────────────────────────────────────────
# Token usage accounting
# ─────────────────────────────────────────────────────────────
# Per-process accumulator. Each agent runs in its own subprocess, so this
# naturally scopes to a single agent invocation; the orchestrator reads each
# agent's totals back out of the ORCH_RESULT payload and re-aggregates them.
_TOKEN_USAGE = {"prompt": 0, "candidates": 0, "total": 0, "calls": 0}


def token_usage_from_response(response) -> dict:
    """Extract {prompt, candidates, total} token counts from an LLM response."""
    um = getattr(response, "usage_metadata", None)
    if um is None:
        return {"prompt": 0, "candidates": 0, "total": 0}
    return {
        "prompt": getattr(um, "prompt_token_count", 0) or 0,
        "candidates": getattr(um, "candidates_token_count", 0) or 0,
        "total": getattr(um, "total_token_count", 0) or 0,
    }


def reset_token_usage():
    global _TOKEN_USAGE
    _TOKEN_USAGE = {"prompt": 0, "candidates": 0, "total": 0, "calls": 0}


def record_token_usage(response):
    """Add one response's token counts to the per-process accumulator."""
    u = token_usage_from_response(response)
    _TOKEN_USAGE["prompt"] += u["prompt"]
    _TOKEN_USAGE["candidates"] += u["candidates"]
    _TOKEN_USAGE["total"] += u["total"]
    _TOKEN_USAGE["calls"] += 1


def get_token_usage() -> dict:
    return dict(_TOKEN_USAGE)


# ─────────────────────────────────────────────────────────────
# Missing / N/A value detection (shared by the orchestrator + evaluator)
# ─────────────────────────────────────────────────────────────
_NA_TOKENS = {"", "n/a", "na", "none", "null", "-", "--"}


def is_na_value(v) -> bool:
    """Return True if a field value is effectively empty / not-available."""
    if v is None:
        return True
    if isinstance(v, str):
        return v.strip().lower() in _NA_TOKENS
    if isinstance(v, (list, dict, tuple)):
        return len(v) == 0
    return False


def _generate_with_retry(model, prompt, max_attempts: int = 4):
    """Call model.generate_content, retrying transient server errors with backoff.

    Transient failures (e.g. 500 Internal error, 503 overloaded, rate limits)
    are retried with exponential backoff. Non-transient errors are raised
    immediately so the caller can surface the real problem. Token usage from the
    successful response is recorded in the per-process accumulator.
    """
    last_err = None
    for attempt in range(max_attempts):
        try:
            resp = model.generate_content(prompt)
            record_token_usage(resp)
            return resp
        except Exception as e:
            last_err = e
            if not _is_transient_llm_error(e) or attempt == max_attempts - 1:
                raise
            wait = 2 ** attempt  # 1s, 2s, 4s, ...
            print(f"  ⏳ Transient LLM error ({str(e)[:120]}). "
                  f"Retry {attempt + 1}/{max_attempts - 1} in {wait}s...")
            time.sleep(wait)
    raise last_err

