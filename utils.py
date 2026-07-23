"""
Shared helpers used across the scraping agents and orchestrator.

This is the common core both agents build on, so neither agent depends on the
other. It owns the Cloudflare-resistant page fetch (browser + requests
fallback), challenge detection, LLM retry/token accounting, and small value
helpers. Each agent supplies its own `create_structural_map` to
`fetch_page_structure` via the `map_fn` argument, since the two agents build
structural maps differently.
"""
import json
import ast
import os
import re
import subprocess
import sys
import time
import traceback
from typing import Callable, Dict, List, Optional, Tuple

import requests
import google.generativeai as genai
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright
from urllib.parse import urlparse


# ─────────────────────────────────────────────────────────────
# Prompt-size / token budgeting
# ─────────────────────────────────────────────────────────────
# Room left for the model's own output when comparing against the context
# window.
OUTPUT_TOKEN_RESERVE = 1024

# Default per-request input-token budget. Google's free tier caps input tokens
# at ~16k per model *per minute* and does NOT expose that quota through the API,
# so this stays a configurable number (with headroom). Raise it on paid tiers
# or larger models.
DEFAULT_INPUT_TOKEN_BUDGET = 15000


def estimate_tokens(text: str) -> int:
    """Cheap, dependency-free token estimate used only as a fallback for
    count_tokens(). Conservative for non-ASCII scripts (e.g. Arabic), where the
    usual ~4-chars/token rule underestimates: each non-ASCII char counts as ~1
    token so we never under-budget."""
    if not text:
        return 1
    ascii_n = sum(1 for c in text if ord(c) < 128)
    return max(1, ascii_n // 4 + (len(text) - ascii_n))


def count_tokens(model, text: str) -> int:
    """Exact prompt token count via the model's own tokenizer.

    Uses the separate countTokens endpoint, which does NOT consume the
    generateContent input-token quota, so it is safe to call before generating.
    Falls back to estimate_tokens() if the call fails (offline, quota, etc.).
    """
    try:
        return int(model.count_tokens(text).total_tokens)
    except Exception:
        return estimate_tokens(text)


def model_context_limit(model_name: str):
    """Best-effort input context-window size for *model_name* from the API's
    model metadata, or None if it can't be determined (so the caller can fall
    back to the configured per-request budget instead of over-shrinking)."""
    try:
        name = model_name if model_name.startswith("models/") else "models/" + model_name
        info = genai.get_model(name)
        limit = getattr(info, "input_token_limit", None)
        return int(limit) if limit else None
    except Exception:
        return None


def input_token_budget(model_name: str, max_input_tokens: int = DEFAULT_INPUT_TOKEN_BUDGET) -> int:
    """Effective per-request input-token budget: the smaller of the configured
    rate-quota budget and the model's context window (minus an output reserve).
    When the context window is unknown, the configured budget rules."""
    ctx = model_context_limit(model_name)
    if not ctx:
        return max_input_tokens
    return min(max_input_tokens, max(1000, ctx - OUTPUT_TOKEN_RESERVE))

# Transient server-side errors worth retrying (Gemini 500/503, rate limits, etc.)
_TRANSIENT_LLM_MARKERS = (
    "500", "503", "internal error", "internal server", "overloaded",
    "unavailable", "deadline", "timeout", "429", "rate limit", "resource exhausted",
)


def _is_transient_llm_error(err: Exception) -> bool:
    """Return True if the error looks like a transient, retryable LLM failure."""
    msg = str(err).lower()
    return any(marker in msg for marker in _TRANSIENT_LLM_MARKERS)


async def list_available_models(api_key: str) -> List[str]:
    """List all Gemini model names that support generateContent."""
    try:
        genai.configure(api_key=api_key)
        models = genai.list_models()
        available = []
        for model in models:
            if 'generateContent' in model.supported_generation_methods:
                available.append(model.name)
        return available
    except Exception as e:
        print(f"⚠️  Could not list models: {e}")
        return []


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


# ─────────────────────────────────────────────────────────────
# Generated-code sandboxing
# ─────────────────────────────────────────────────────────────
DEFAULT_SANDBOX_TIMEOUT_SECONDS = 30
DEFAULT_SANDBOX_MAX_STDOUT = 1_000_000


def validate_generated_code_safety(code: str) -> Tuple[bool, str]:
    """Cheap text-level guardrail before AST validation and subprocess execution."""
    code_no_comments = re.sub(r'#.*$', '', code, flags=re.MULTILINE)
    dangerous_patterns = [
        r'import\s+os',
        r'import\s+sys',
        r'import\s+subprocess',
        r'import\s+requests',
        r'import\s+socket',
        r'import\s+pickle',
        r'__import__',
        r'eval\s*\(',
        r'exec\s*\(',
        r'(?<!re\.)compile\s*\(',
        r'open\s*\(',
        r'file\s*\(',
    ]

    for pattern in dangerous_patterns:
        if re.search(pattern, code_no_comments, re.IGNORECASE):
            return False, f"Dangerous pattern detected: {pattern}"

    return True, "Code appears safe"


_FORBIDDEN_AST_NODES = (
    ast.Import, ast.ImportFrom, ast.ClassDef, ast.Lambda, ast.Global,
    ast.Nonlocal, ast.With, ast.AsyncWith, ast.AsyncFunctionDef, ast.Await,
    ast.Yield, ast.YieldFrom, ast.Try, ast.Raise, ast.Delete, ast.While,
)

_FORBIDDEN_NAMES = {
    '__builtins__', '__import__', 'eval', 'exec', 'compile', 'open', 'input',
    'globals', 'locals', 'vars', 'dir', 'type', 'super', 'object', 'memoryview',
    'getattr', 'setattr', 'delattr', 'hasattr', 'breakpoint', 'help',
}


def validate_generated_code_ast(code: str) -> Tuple[bool, str]:
    """Allow only a small, extraction-oriented subset of Python syntax."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"Syntax error: {e}"

    function_defs = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if len(function_defs) != 1 or function_defs[0].name != 'extract_data':
        return False, "Generated code must define exactly one function named extract_data"
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            return False, "Only the extract_data function may appear at module scope"

    for node in ast.walk(tree):
        if isinstance(node, _FORBIDDEN_AST_NODES):
            return False, f"Disallowed syntax: {type(node).__name__}"
        if isinstance(node, ast.Name) and node.id in _FORBIDDEN_NAMES:
            return False, f"Disallowed name: {node.id}"
        if isinstance(node, ast.Attribute) and node.attr.startswith('__'):
            return False, f"Disallowed private/introspection attribute: {node.attr}"
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in _FORBIDDEN_NAMES:
                return False, f"Disallowed call: {func.id}"
            if isinstance(func, ast.Attribute) and func.attr.startswith('__'):
                return False, f"Disallowed call attribute: {func.attr}"

    return True, "AST appears safe"


_SANDBOX_WORKER_CODE = r'''
import json
import re
import sys
import traceback
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse


for stream in (sys.stdin, sys.stdout, sys.stderr):
    if hasattr(stream, 'reconfigure'):
        stream.reconfigure(encoding='utf-8', errors='replace')


def safe_urljoin(base, url, *args, **kwargs):
    if isinstance(base, list):
        base = base[0] if base else ''
    if isinstance(url, list):
        url = url[0] if url else ''
    return urljoin(str(base), str(url), *args, **kwargs)


def safe_urlparse(url, *args, **kwargs):
    if isinstance(url, list):
        url = url[0] if url else ''
    return urlparse(str(url), *args, **kwargs)


def main():
    payload = json.loads(sys.stdin.read())
    restricted_globals = {
        '__builtins__': {
            'print': lambda *args, **kwargs: None,
            'len': len, 'str': str, 'int': int, 'float': float,
            'bool': bool, 'list': list, 'dict': dict, 'set': set, 'tuple': tuple,
            'range': range, 'enumerate': enumerate, 'zip': zip, 'filter': filter,
            'map': map, 'sorted': sorted, 'any': any, 'all': all, 'max': max,
            'min': min, 'sum': sum, 'None': None, 'True': True, 'False': False,
            'isinstance': isinstance,
        },
        'BeautifulSoup': BeautifulSoup,
        're': re,
        'json': json,
        'urljoin': safe_urljoin,
        'urlparse': safe_urlparse,
    }

    exec(payload['code'], restricted_globals)
    user_func = restricted_globals.get('extract_data')
    if not callable(user_func):
        raise RuntimeError('Generated code does not define callable extract_data')
    result = user_func(payload['html_content'])
    print(json.dumps({'ok': True, 'result': result}, ensure_ascii=False))


try:
    main()
except Exception as e:
    print(json.dumps({'ok': False, 'error': str(e), 'traceback': traceback.format_exc()}, ensure_ascii=False))
'''


def _sandbox_env() -> Dict[str, str]:
    env = {'PYTHONIOENCODING': 'utf-8', 'PYTHONUTF8': '1'}
    for key in ('SystemRoot', 'TEMP', 'TMP'):
        if key in os.environ:
            env[key] = os.environ[key]
    return env


def execute_generated_code_sandboxed(
    code: str,
    html_content: str,
    output_validator: Callable,
    timeout_seconds: int = DEFAULT_SANDBOX_TIMEOUT_SECONDS,
    max_stdout: int = DEFAULT_SANDBOX_MAX_STDOUT,
) -> Tuple[bool, object]:
    """Validate and execute generated extract_data code in a stripped subprocess."""
    if all(line.strip() == '' or line.strip().startswith('#') for line in code.splitlines()):
        return False, f"LLM ERROR: {code.replace('# ', '').strip()}"

    is_safe, safety_msg = validate_generated_code_safety(code)
    if not is_safe:
        return False, f"SAFETY ERROR: {safety_msg}"

    ast_safe, ast_msg = validate_generated_code_ast(code)
    if not ast_safe:
        return False, f"AST SAFETY ERROR: {ast_msg}"

    try:
        payload = json.dumps({'code': code, 'html_content': html_content}, ensure_ascii=False)
        completed = subprocess.run(
            [sys.executable, '-I', '-c', _SANDBOX_WORKER_CODE],
            input=payload,
            text=True,
            encoding='utf-8',
            errors='replace',
            capture_output=True,
            timeout=timeout_seconds,
            env=_sandbox_env(),
        )
        stdout = completed.stdout.strip()
        if len(stdout) > max_stdout:
            return False, "SANDBOX ERROR: Output exceeded maximum size"
        if completed.returncode != 0:
            return False, f"SANDBOX ERROR: Worker exited with {completed.returncode}: {completed.stderr.strip()}"
        if not stdout:
            return False, "SANDBOX ERROR: Worker produced no output"
        envelope = json.loads(stdout.splitlines()[-1])
        if not envelope.get('ok'):
            return False, f"EXECUTION ERROR: {envelope.get('error', 'unknown error')}\n\n{envelope.get('traceback', '')}"

        schema_ok, schema_msg, clean_result = output_validator(envelope.get('result'))
        if not schema_ok:
            return False, f"OUTPUT SCHEMA ERROR: {schema_msg}"
        return True, clean_result
    except subprocess.TimeoutExpired:
        return False, f"SANDBOX ERROR: Execution timed out after {timeout_seconds} seconds"
    except json.JSONDecodeError as e:
        return False, f"SANDBOX ERROR: Invalid JSON from worker: {e}"
    except Exception as e:
        error_detail = traceback.format_exc()
        return False, f"EXECUTION ERROR: {str(e)}\n\n{error_detail}"


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


# ─────────────────────────────────────────────────────────────
# Cloudflare-resistant page fetch (shared by both agents + orch)
# ─────────────────────────────────────────────────────────────
# Populated whenever a fetch fully fails, so callers can report the real cause.
_LAST_FETCH_ERROR = None


def get_last_fetch_error():
    """Return the aggregated error from the most recent failed fetch (or None)."""
    return _LAST_FETCH_ERROR


CONTENT_READY_SELECTORS = [
    'article', '.article', 'table', '.listing', 'ul li a',
    '.item', '.post', '.card', '.entry', '.news-item',
    'h2 a', 'h3 a', '.title a',
]

# Strong markers: if present, the page is almost certainly an interstitial
# challenge regardless of size.
STRONG_CHALLENGE_MARKERS = [
    'cf-browser-verification',
    'cf_chl_opt',
    'Just a moment',
    'Checking your browser',
    'Performing security verification',
    'Verifying you are human',
    'Enable JavaScript and cookies to continue',
]

# Weak markers: Cloudflare injects these (e.g. /cdn-cgi/challenge-platform/...
# scripts, Turnstile widgets) into NORMAL served pages too, so they only
# indicate a challenge when the page has no real content yet.
WEAK_CHALLENGE_MARKERS = [
    'challenge-platform',
    'turnstile',
]


def _has_real_content(html: str) -> bool:
    """Heuristic: does the HTML contain enough article/body markup to be a
    real content page (as opposed to a tiny interstitial challenge page)?"""
    if len(html) < 20000:
        return False
    lowered = html.lower()
    content_signals = (
        lowered.count('<article') + lowered.count('<p') + lowered.count('<h1')
        + lowered.count('<h2') + lowered.count('<li')
    )
    return content_signals >= 3


def _is_challenge_page(html: str) -> bool:
    """Return True if the HTML looks like a Cloudflare/bot challenge, not real content.

    Strong markers always count. Weak markers (which Cloudflare also injects
    into normally-served pages) only count when the page lacks real content —
    this avoids the false positive where a fully-loaded page is mistaken for a
    challenge just because it carries an injected challenge-platform script.
    """
    if any(m in html for m in STRONG_CHALLENGE_MARKERS):
        return True
    if any(m in html for m in WEAK_CHALLENGE_MARKERS):
        return not _has_real_content(html)
    return False


async def _wait_for_content(page, timeout_ms: int = 15000) -> bool:
    """After a challenge clears, wait until a known content selector appears."""
    combined_selector = ', '.join(CONTENT_READY_SELECTORS)
    try:
        await page.wait_for_selector(combined_selector, timeout=timeout_ms)
        print(f"  ✓ Content selector found — DOM is ready")
        return True
    except Exception:
        print(f"  ⚠  Content selector not found within {timeout_ms // 1000}s — using DOM as-is")
        return False


async def _launch_and_fetch(p, url: str, headless: bool):
    """Launch browser, navigate, handle challenges.
    Returns (html_or_None, status, error_or_None)."""
    mode = "headless" if headless else "headed"
    browser = None
    try:
        browser = await p.chromium.launch(
            headless=headless,
            args=['--disable-blink-features=AutomationControlled', '--no-sandbox'],
        )
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                       "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
            viewport={'width': 1920, 'height': 1080},
            locale='en-US',
            timezone_id='America/New_York',
        )
        page = await context.new_page()
        await page.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
            Object.defineProperty(navigator, 'plugins', { get: () => [1, 2, 3, 4, 5] });
            Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'] });
        """)

        await page.goto(url, wait_until="domcontentloaded", timeout=60000)
        try:
            await page.wait_for_load_state("networkidle", timeout=15000)
        except Exception:
            pass

        max_checks = 12 if headless else 18
        content_waited = False
        for attempt in range(max_checks):
            try:
                html_snapshot = await page.content()
            except Exception:
                live_pages = context.pages
                if live_pages:
                    page = live_pages[-1]
                    await page.wait_for_load_state("domcontentloaded", timeout=10000)
                    html_snapshot = await page.content()
                else:
                    raise RuntimeError("Browser has no open pages")

            if not _is_challenge_page(html_snapshot):
                print(f"  ✓ Page loaded [{mode}] (after {(attempt + 1) * 5}s)")
                if not content_waited:
                    content_waited = True
                    await _wait_for_content(page, timeout_ms=15000)
                break

            if 'Verification successful' in html_snapshot:
                print(f"  ✓ Challenge solved, waiting for content... ({(attempt + 1) * 5}s)")
                await page.wait_for_timeout(5000)
                continue

            print(f"  ⏳ Waiting for challenge [{mode}]... ({(attempt + 1) * 5}s)")
            await page.wait_for_timeout(5000)

        # Scroll for lazy-loaded content
        try:
            await page.evaluate("""async () => {
                const delay = ms => new Promise(r => setTimeout(r, ms));
                for (let i = 0; i < 3; i++) {
                    window.scrollBy(0, window.innerHeight);
                    await delay(800);
                }
                window.scrollTo(0, 0);
            }""")
            await page.wait_for_timeout(2000)
        except Exception:
            page = context.pages[-1] if context.pages else page

        html_content = await page.content()
        status = "challenge" if _is_challenge_page(html_content) else "ok"
        return html_content, status, None

    except Exception as e:
        err = f"[{mode}] {type(e).__name__}: {e}\n{traceback.format_exc()}"
        print(f"  ❌ Failed [{mode}]: {e}")
        return None, "error", err
    finally:
        if browser is not None:
            try:
                await browser.close()
            except Exception:
                pass


def _fetch_with_requests(url: str):
    """Fallback: fetch via requests. Returns (html_or_None, error_or_None)."""
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                      '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'ar,en-US;q=0.9,en;q=0.8',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Referer': urlparse(url)._replace(path='/', params='', query='', fragment='').geturl(),
    }
    )
    last_err = None
    for attempt in range(3):
        try:
            print(f"  [requests] Attempt {attempt + 1}/3...")
            resp = session.get(url, timeout=30)
            resp.raise_for_status()
            html = resp.text
            if _is_challenge_page(html):
                last_err = "requests received a Cloudflare/bot challenge page"
                print(f"  [requests] Still got a challenge page")
                return None, last_err
            if any(tag in html.lower() for tag in ['<html', '<body', '<div', '<!doctype']):
                print(f"  ✓ Page fetched via requests ({len(html)} bytes)")
                return html, None
            last_err = "requests response did not look like HTML"
            print(f"  [requests] {last_err}")
            return None, last_err
        except requests.RequestException as e:
            last_err = f"{type(e).__name__}: {e}"
            print(f"  [requests] Error: {e}")
            if attempt < 2:
                time.sleep((attempt + 1) * 3)
    return None, last_err


async def fetch_page_structure(
    url: str,
    map_fn: Callable[[BeautifulSoup], List[Dict]],
) -> Tuple[Optional[str], Optional[List[Dict]]]:
    """Fetch a page (headless → headed browser → plain HTTP) and build its map.

    *map_fn* is the caller's structural-map builder (each agent has its own),
    called as ``map_fn(soup.body or soup)``. Returns (html, structural_map),
    (html, []) when the map is empty, or (None, None) when every fetch strategy
    fails (in which case get_last_fetch_error() holds the aggregated cause).
    """
    global _LAST_FETCH_ERROR
    _LAST_FETCH_ERROR = None
    errors = []

    # Attempt 1: Playwright headless
    async with async_playwright() as p:
        html_content, status, err = await _launch_and_fetch(p, url, headless=True)
    if err:
        errors.append("── Headless browser attempt ──\n" + err)

    # Attempt 2: headed browser (on challenge OR crash)
    if status in ("challenge", "error"):
        why = "blocked by Cloudflare" if status == "challenge" else "crashed"
        print(f"  ⚠  Headless browser {why} — trying headed browser...")
        async with async_playwright() as p:
            html_content, status, err = await _launch_and_fetch(p, url, headless=False)
        if err:
            errors.append("── Headed browser attempt ──\n" + err)

    # Attempt 3: plain HTTP (also runs when the browser errored, not only on challenge)
    if status in ("challenge", "error") or html_content is None:
        print("  ⚠  Browser attempts unusable — trying plain HTTP (requests)...")
        req_html, req_err = _fetch_with_requests(url)
        if req_html:
            html_content, status = req_html, "ok"
        elif req_err:
            errors.append("── requests attempt ──\n" + req_err)

    if html_content is None:
        _LAST_FETCH_ERROR = "\n\n".join(errors) if errors else \
            "No HTML returned and no underlying error was captured."
        print("  ❌ Could not fetch page via headless, headed, or HTTP requests.")
        print(_LAST_FETCH_ERROR)
        return None, None

    soup = BeautifulSoup(html_content, 'lxml')
    structural_map = map_fn(soup.body if soup.body else soup)

    if not structural_map:
        print(f"  ⚠  Structural map is empty (no matching tags found).")
        print(f"  ⚠  HTML length: {len(html_content)} bytes")
        print(f"  ⚠  First 500 chars of HTML:\n{html_content[:500]}\n")
        print(f"  ⚠  Proceeding with empty map — LLM will work from raw HTML.")
        return html_content, []

    return html_content, structural_map

