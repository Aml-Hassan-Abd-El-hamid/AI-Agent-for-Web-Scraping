"""
Shared helpers used across the scraping agents and orchestrator.

This is the common core both agents build on, so neither agent depends on the
other. It owns the Cloudflare-resistant page fetch (browser + requests
fallback), challenge detection, LLM retry/token accounting, and small value
helpers. Each agent supplies its own `create_structural_map` to
`fetch_page_structure` via the `map_fn` argument, since the two agents build
structural maps differently.
"""
import time
import traceback
from typing import Callable, Dict, List, Optional, Tuple

import requests
import google.generativeai as genai
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright
from urllib.parse import urlparse

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

