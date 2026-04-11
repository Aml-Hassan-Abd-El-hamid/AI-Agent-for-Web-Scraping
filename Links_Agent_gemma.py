"""
LLM:  Gemma (gemma-3-27b-it / gemma-3-12b-it via Google AI API)

Gemma-targeted version of Links_Agent_m2.py with:
- Prompts formatted using Gemma's <start_of_turn> / <end_of_turn> control tokens
- Few-shot examples to guide code generation
- System-style instructions prepended to user turn
- All other functionality identical to Links_Agent_m2.py


Tested on the following links:
- ❌ fields extraction from this page -failed- tried everything and decided to move on to other pages:
    - https://www.palestine-studies.org/ar/blogs/explorer?f%5B0%5D=field_blog_series%3A19943
- ✅ youm7 -success :)-:
    - https://www.youm7.com/Section/%D8%A3%D8%AE%D8%A8%D8%A7%D8%B1-%D8%B9%D8%A7%D8%AC%D9%84%D8%A9/65/1
- 🔄 Cloudflare challenge page -failed-:
    - https://www.almasryalyoum.com/section/index/8
- ✅ Successfully extracted article links from this page -success-:  
    - https://pchrgaza.org/ar/category/genocide-on-gaza-ar/testimonies-from-the-war-ar/

"""
import asyncio
import json
import re
import random
import time
import requests
from typing import Dict, List, Optional, Tuple
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
import google.generativeai as genai
from urllib.parse import urljoin, urlparse

random_num = random.randint(10000, 99999)

# --- Configuration ---
TARGET_TAGS = ['div', 'section', 'article', 'main', 'header', 'footer', 'nav',
               'ul', 'ol', 'li', 'a', 'h1', 'h2', 'h3', 'h4', 'p', 'span',
               'table', 'tr', 'td', 'th', 'figure', 'figcaption', 'time', 'img']
MAX_DEPTH = 10
MAX_RETRIES = 3

# Allowed imports for generated code
ALLOWED_IMPORTS = {
    'BeautifulSoup': BeautifulSoup,
    're': re,
    'json': json,
    'urljoin': urljoin,
    'urlparse': urlparse,
}

# --- Structural Map Generation ---
def create_structural_map(soup: BeautifulSoup, depth: int = 0) -> List[Dict]:
    """Recursively generates a simplified, nested structural map of the HTML."""
    if depth >= MAX_DEPTH:
        return []

    structure = []
    
    for child in soup.children:
        if child.name and child.name.lower() in TARGET_TAGS:
            attributes = {}
            if child.get('class'):
                attributes['class'] = " ".join(child.get('class')[:4])
            if child.get('id'):
                attributes['id'] = child.get('id')
            if child.get('role'):
                attributes['role'] = child.get('role')
            if child.name.lower() == 'a' and child.get('href'):
                attributes['href'] = child.get('href')
            # Capture data-* attributes (common in SPAs)
            for attr_name, attr_val in child.attrs.items():
                if attr_name.startswith('data-') and isinstance(attr_val, str):
                    attributes[attr_name] = attr_val[:80]

            node = {
                'tag': child.name.lower(),
                'attributes': attributes,
                'children': create_structural_map(child, depth + 1)
            }
            
            if not node['children'] and child.text and len(child.text.strip()) > 5:
                text_content = child.text.strip()
                node['text_snippet'] = text_content[:120].replace('\n', ' ') + ('...' if len(text_content) > 120 else '')

            structure.append(node)
            
    return structure


CHALLENGE_MARKERS = [
    'cf-browser-verification',
    'challenge-platform',
    'Just a moment',
    'Checking your browser',
    'cf_chl_opt',
    'turnstile',
    'Performing security verification',
]

def _is_challenge_page(html: str) -> bool:
    """Return True if the HTML looks like a Cloudflare/bot challenge, not real content."""
    return any(m in html for m in CHALLENGE_MARKERS)


async def _launch_and_fetch(p, url: str, headless: bool) -> Tuple[Optional[str], Optional[str]]:
    """Low-level: launch browser, navigate, handle challenges, return (html, status)."""
    mode = "headless" if headless else "headed"
    browser = await p.chromium.launch(
        headless=headless,
        args=[
            '--disable-blink-features=AutomationControlled',
            '--no-sandbox',
        ]
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

    try:
        await page.goto(url, wait_until="domcontentloaded", timeout=60000)
        try:
            await page.wait_for_load_state("networkidle", timeout=15000)
        except Exception:
            pass

        # Wait for challenge to resolve (up to 60s headless, 90s headed)
        max_checks = 12 if headless else 18
        for attempt in range(max_checks):
            # The page object may become stale if Cloudflare redirects;
            # recover by grabbing the latest page from the context.
            try:
                html_snapshot = await page.content()
            except Exception:
                try:
                    live_pages = context.pages
                    if live_pages:
                        page = live_pages[-1]
                        await page.wait_for_load_state("domcontentloaded", timeout=10000)
                        html_snapshot = await page.content()
                    else:
                        print(f"  ❌ Browser has no open pages [{mode}]")
                        break
                except Exception as recover_err:
                    print(f"  ❌ Could not recover page [{mode}]: {recover_err}")
                    break

            if not _is_challenge_page(html_snapshot):
                print(f"  ✓ Page loaded [{mode}] (after {(attempt + 1) * 5}s)")
                break

            if 'Verification successful' in html_snapshot:
                print(f"  ✓ Challenge solved, waiting for content... ({(attempt + 1) * 5}s)")
                # Just wait — don't use wait_for_event which misses same-URL reloads
                await page.wait_for_timeout(5000)
                continue

            print(f"  ⏳ Waiting for challenge [{mode}]... ({(attempt + 1) * 5}s)")
            await page.wait_for_timeout(5000)

        # Scroll for lazy-loaded content (page may have been swapped above)
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
            html_content = await page.content()
        except Exception:
            # Recover page one more time if needed
            try:
                page = context.pages[-1] if context.pages else page
                html_content = await page.content()
            except Exception as e:
                print(f"  ❌ Failed to get final content [{mode}]: {e}")
                await browser.close()
                return None, "error"

    except Exception as e:
        print(f"  ❌ Failed [{mode}]: {e}")
        await browser.close()
        return None, "error"

    await browser.close()

    if _is_challenge_page(html_content):
        return html_content, "challenge"
    return html_content, "ok"


def _fetch_with_requests(url: str) -> Optional[str]:
    """Fallback: fetch via requests.Session with browser-like headers.
    Some Cloudflare setups block headless browsers but allow plain HTTP."""
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
    })
    for attempt in range(3):
        try:
            print(f"  [requests] Attempt {attempt + 1}/3...")
            resp = session.get(url, timeout=30)
            resp.raise_for_status()
            html = resp.text
            if _is_challenge_page(html):
                print(f"  [requests] Still got a challenge page")
                return None
            # Quick sanity check
            if any(tag in html.lower() for tag in ['<html', '<body', '<div', '<!doctype']):
                print(f"  ✓ Page fetched via requests ({len(html)} bytes)")
                return html
            print(f"  [requests] Response doesn't look like HTML")
            return None
        except requests.RequestException as e:
            print(f"  [requests] Error: {e}")
            if attempt < 2:
                time.sleep((attempt + 1) * 3)
    return None


async def fetch_page_structure(url: str) -> Tuple[Optional[str], Optional[List[Dict]]]:
    # Attempt 1: Playwright headless (fast, works for most sites)
    async with async_playwright() as p:
        html_content, status = await _launch_and_fetch(p, url, headless=True)

    if status == "challenge":
        # Attempt 2: Playwright headed — can solve interactive Cloudflare challenges
        print("  ⚠  Cloudflare blocked headless browser — trying headed browser...")
        async with async_playwright() as p:
            html_content, status = await _launch_and_fetch(p, url, headless=False)

    if status == "challenge":
        # Attempt 3: plain requests (works when Cloudflare blocks browsers
        # but allows normal HTTP with good headers)
        print("  ⚠  Headed browser also blocked — trying plain HTTP...")
        html_content = _fetch_with_requests(url)
        if html_content:
            status = "ok"

    if status != "ok" or not html_content:
        if status == "challenge":
            print("  ❌ Could not bypass Cloudflare via browser, headed browser, or HTTP requests.")
        return None, None
    
    soup = BeautifulSoup(html_content, 'lxml')
    structural_map = create_structural_map(soup.body if soup.body else soup)
    
    return html_content, structural_map


# --- LLM Integration ---
async def list_available_models(api_key: str) -> List[str]:
    """List all available Gemini models."""
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


# ═══════════════════════════════════════════════════════════
# FEW-SHOT EXAMPLE (teaches Gemma the exact input→output pattern)
# ═══════════════════════════════════════════════════════════
_FEW_SHOT_MAP = '''[
  {"tag": "div", "attributes": {"class": "post-list"}, "children": [
    {"tag": "article", "attributes": {"class": "post-card"}, "children": [
      {"tag": "h2", "attributes": {}, "children": [
        {"tag": "a", "attributes": {"href": "/2024/03/story-one", "class": "post-title"}, "text_snippet": "Story One Title"}
      ]},
      {"tag": "span", "attributes": {"class": "date"}, "text_snippet": "March 1, 2024"}
    ]},
    {"tag": "article", "attributes": {"class": "post-card"}, "children": [
      {"tag": "h2", "attributes": {}, "children": [
        {"tag": "a", "attributes": {"href": "/2024/03/story-two", "class": "post-title"}, "text_snippet": "Story Two Title"}
      ]},
      {"tag": "span", "attributes": {"class": "date"}, "text_snippet": "March 2, 2024"}
    ]}
  ]}
]'''

_FEW_SHOT_CODE = '''def extract_data(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    base_url = '/'.join('https://example.com/news'.split('/')[:3])

    seen = set()
    article_links = []

    for article in soup.select('article.post-card'):
        a_tag = article.select_one('h2 a.post-title')
        if not a_tag or not a_tag.get('href'):
            continue
        href = a_tag['href']
        if href.startswith('javascript:') or href == '#':
            continue
        url = urljoin(base_url, href)
        if url in seen:
            continue
        seen.add(url)
        title = a_tag.get_text(strip=True)
        article_links.append({"url": url, "title": title})

    return {"article_links": article_links}'''


class GemmaAgent:
    """LLM agent using Gemma's prompt format with few-shot examples."""

    def __init__(self, api_key: str, model_name: str = 'gemma-3-27b-it'):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.conversation_history = []
        self.model_name = model_name
    
    def generate_extraction_code(self, structural_map: str, user_requirements: str = "", page_url: str = "", error_context: Optional[str] = None) -> str:
        """Generate Python extraction code using Gemma's prompt format with few-shot learning."""

        if not error_context:
            # ── Few-shot turn 1: teach the pattern ──────────────
            prompt = f"""<start_of_turn>user
You are a Python web-scraping expert. You will receive an HTML structural map (JSON) and a target URL. Your job is to write a single Python function called `extract_data(html_content)` that extracts all article links from the page.

Rules you MUST follow:
- The function signature is exactly: def extract_data(html_content):
- Use BeautifulSoup to parse the HTML.
- Return a dict: {{"article_links": [  {{"url": "...", "title": "..."}}, ...  ]}}
- Use urljoin(base_url, href) to resolve relative URLs. Derive base_url from the target URL.
- Deduplicate by URL.
- Only use selectors (tags, classes, ids) that appear in the structural map. Do NOT invent selectors.
- Exclude: nav/header/footer links, javascript: hrefs, anchor-only (#) links, social media domains.
- If no links exist, return {{"article_links": []}}
- Do NOT include import statements. Only output the function body.
- Available in scope: BeautifulSoup, re, json, urljoin, urlparse.

Think step-by-step:
1. Identify the repeating container pattern in the structural map that holds article links.
2. Pick the CSS selector for that container and the <a> tag inside it.
3. Write the function.

Here is an example:

TARGET URL: https://example.com/news

STRUCTURAL MAP:
{_FEW_SHOT_MAP}
<end_of_turn>
<start_of_turn>model
{_FEW_SHOT_CODE}
<end_of_turn>
<start_of_turn>user
Good. Now do the same for this real page.

TARGET URL: {page_url}

STRUCTURAL MAP:
{structural_map}

Write only the Python function. No explanation, no imports.
<end_of_turn>
<start_of_turn>model
"""
        else:
            # ── Retry turn: include the error ──────────────────
            prompt = f"""<start_of_turn>user
You are a Python web-scraping expert. Your previous code failed. Fix it.

Here is a working example for reference:

TARGET URL: https://example.com/news
STRUCTURAL MAP:
{_FEW_SHOT_MAP}

Working code:
{_FEW_SHOT_CODE}

Now here is the REAL task that failed:

TARGET URL: {page_url}

STRUCTURAL MAP:
{structural_map}

ERROR from the previous attempt:
{error_context}

Fix the code. Same rules:
- Function signature: def extract_data(html_content):
- Return dict with key "article_links" as list of {{"url": ..., "title": ...}}
- Use urljoin(base_url, href) for relative URLs.  base_url = '/'.join('{page_url}'.split('/')[:3])
- Only use selectors from the structural map.
- Available: BeautifulSoup, re, json, urljoin, urlparse
- Output ONLY the corrected function. No explanation.
<end_of_turn>
<start_of_turn>model
"""

        try:
            response = self.model.generate_content(prompt)
            code = response.text

            # ── Extract just the Python function from Gemma's verbose output ──
            # Strategy 1: pull code from the LAST ```python ... ``` block
            fenced = re.findall(r'```python\s*\n(.*?)```', code, re.DOTALL)
            if fenced:
                code = fenced[-1].strip()
            else:
                # Strategy 2: find the last "def extract_data(" or any "def " line
                # and keep everything from there
                lines = code.split('\n')
                func_start = None
                for i, line in enumerate(lines):
                    if re.match(r'^def\s+\w+\s*\(', line):
                        func_start = i
                if func_start is not None:
                    code = '\n'.join(lines[func_start:])
                # else: leave code as-is and let exec() report the error

            # Remove any trailing Gemma turn tokens the model might echo
            code = re.sub(r'<end_of_turn>\s*$', '', code)
            # Remove any stray markdown leftovers
            code = re.sub(r'^```\w*\s*$', '', code, flags=re.MULTILINE)
            code = code.strip()
            
            # Save generated code to file
            
            code_filename = f"code/gemma_generated_code_{random_num}.py"
            with open(code_filename, 'w', encoding='utf-8') as f:
                f.write(code)
            print(f"💾 Code saved to: {code_filename}")
            
            self.conversation_history.append({
                'prompt': prompt,
                'response': code,
                'error': error_context,
                'saved_file': code_filename
            })
            
            return code
            
        except Exception as e:
            error_msg = str(e)
            error_code = '\n'.join(f'# {line}' for line in f'Error generating code: {error_msg}'.splitlines())
            #random_num = random.randint(10000, 99999)
            code_filename = f"code/gemma_generated_code_{random_num}_ERROR.py"
            with open(code_filename, 'w', encoding='utf-8') as f:
                f.write(error_code)
            print(f"❌ LLM call failed: {error_msg[:200]}")
            return error_code


# --- Safe Code Execution ---
def validate_code_safety(code: str) -> Tuple[bool, str]:
    """Basic safety validation for generated code."""
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


def execute_extraction_code(code: str, html_content: str) -> Tuple[bool, any]:
    """Execute the generated extraction code in a restricted environment."""

    # If the code is just comments (LLM call failed), don't bother exec-ing
    if all(line.strip() == '' or line.strip().startswith('#') for line in code.splitlines()):
        return False, f"LLM ERROR: {code.replace('# ', '').strip()}"
    
    # Validate safety first
    is_safe, safety_msg = validate_code_safety(code)
    if not is_safe:
        return False, f"SAFETY ERROR: {safety_msg}"

    # Wrap urljoin/urlparse to handle non-string args
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

    # Create restricted namespace with proper imports
    restricted_globals = {
        '__builtins__': {
            'print': print,
            'len': len,
            'str': str,
            'int': int,
            'float': float,
            'bool': bool,
            'list': list,
            'dict': dict,
            'set': set,
            'tuple': tuple,
            'range': range,
            'enumerate': enumerate,
            'zip': zip,
            'filter': filter,
            'map': map,
            'sorted': sorted,
            'any': any,
            'all': all,
            'max': max,
            'min': min,
            'sum': sum,
            'None': None,
            'True': True,
            'False': False,
            'isinstance': isinstance,
            'hasattr': hasattr,
            'getattr': getattr,
        },
        'BeautifulSoup': BeautifulSoup,
        're': re,
        'json': json,
        'urljoin': safe_urljoin,
        'urlparse': safe_urlparse,
        'bs4': type('Module', (), {'BeautifulSoup': BeautifulSoup})(),
    }
    
    def safe_import(name, *args, **kwargs):
        allowed = {
            'bs4': type('Module', (), {'BeautifulSoup': BeautifulSoup})(),
            're': re,
            'json': json,
            'urllib.parse': type('Module', (), {'urljoin': safe_urljoin, 'urlparse': safe_urlparse})(),
        }
        if name in allowed:
            return allowed[name]
        raise ImportError(f"Import of '{name}' is not allowed")
    
    restricted_globals['__builtins__']['__import__'] = safe_import
    
    try:
        # Snapshot keys before exec so we only pick up LLM-defined functions
        pre_exec_keys = set(restricted_globals.keys())

        exec(code, restricted_globals)
        
        # Auto-detect the user-defined function (Gemma may name it anything)
        user_func = None
        for name in restricted_globals:
            if name in pre_exec_keys or name.startswith('__'):
                continue
            obj = restricted_globals[name]
            if callable(obj):
                user_func = obj
                break

        if user_func is None:
            return False, "ERROR: Generated code does not contain any callable function"
        
        result = user_func(html_content)

        # Normalize list-of-dicts → single dict
        if isinstance(result, list) and len(result) == 1 and isinstance(result[0], dict):
            result = result[0]
        
        return True, result
        
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        return False, f"EXECUTION ERROR: {str(e)}\n\n{error_detail}"


# --- Output Analysis ---
def analyze_output(data: Dict) -> Dict:
    """Generate statistics about the extracted data."""
    stats = {
        'total_fields': len(data),
        'fields': list(data.keys()),
        'field_details': {}
    }
    
    for key, value in data.items():
        if isinstance(value, str):
            stats['field_details'][key] = {
                'type': 'string',
                'length': len(value),
                'preview': value[:100] + '...' if len(value) > 100 else value,
                'is_empty': len(value.strip()) == 0,
                'starts_with_error': value.startswith(('N/A', 'ERROR', 'Error', 'Extraction Error'))
            }
        elif isinstance(value, (list, tuple)):
            stats['field_details'][key] = {
                'type': 'list',
                'count': len(value),
                'preview': str(value[:3]) + '...' if len(value) > 3 else str(value)
            }
        elif isinstance(value, dict):
            stats['field_details'][key] = {
                'type': 'dict',
                'keys': list(value.keys()),
                'preview': str(value)[:100] + '...'
            }
        else:
            stats['field_details'][key] = {
                'type': type(value).__name__,
                'value': str(value)
            }
    
    return stats


def display_sample_output(data: Dict, stats: Dict):
    """Display sample output and statistics to the user."""
    print("\n" + "="*60)
    print("📊 EXTRACTION RESULTS")
    print("="*60)
    
    print(f"\n✓ Total fields extracted: {stats['total_fields']}")
    print(f"✓ Fields: {', '.join(stats['fields'])}")
    
    print("\n" + "-"*60)
    print("SAMPLE OUTPUT:")
    print("-"*60)
    
    for field, details in stats['field_details'].items():
        print(f"\n[{field}]")
        print(f"  Type: {details['type']}")
        
        if details['type'] == 'string':
            status = "❌ EMPTY" if details['is_empty'] else ("⚠️  ERROR" if details['starts_with_error'] else "✓")
            print(f"  Status: {status}")
            print(f"  Length: {details['length']} characters")
            print(f"  Preview: {details['preview']}")
        elif details['type'] == 'list':
            print(f"  Count: {details['count']} items")
            print(f"  Preview: {details['preview']}")
        else:
            print(f"  Preview: {details.get('preview', details.get('value', 'N/A'))}")
    
    print("\n" + "="*60)


# --- Main Agent Logic ---
async def main():
    print("="*60)
    print("🤖 AI WEB SCRAPING AGENT (Gemma Edition)")
    print("="*60)
    
    api_key = input("\n🔑 Enter your Gemini API key: ").strip()
    if not api_key:
        print("❌ API key is required!")
        return
    
    print("\n⏳ Checking available models...")
    available_models = await list_available_models(api_key)
    
    # Filter to show Gemma models first, then others
    gemma_models = [m for m in available_models if 'gemma' in m.lower()]
    other_models = [m for m in available_models if 'gemma' not in m.lower()]
    sorted_models = gemma_models + other_models

    if sorted_models:
        print(f"\n Found {len(sorted_models)} available models:")
        if gemma_models:
            print(f"  ── Gemma models ──")
        for i, model in enumerate(sorted_models, 1):
            marker = " ★" if 'gemma' in model.lower() else ""
            print(f"   {i}. {model}{marker}")
        
        print(f"\n Select a model:")
        print(f"   [Enter number] Choose from list above")
        print(f"   [Press Enter] Use default (gemma-3-27b-it)")
        
        choice = input("   → ").strip()
        
        if choice.isdigit() and 1 <= int(choice) <= len(sorted_models):
            selected_model = sorted_models[int(choice) - 1]
            if selected_model.startswith('models/'):
                selected_model = selected_model[7:]
            print(f"✓ Selected: {selected_model}")
        else:
            selected_model = 'gemma-3-27b-it'
            print(f"✓ Using default: {selected_model}")
    else:
        selected_model = 'gemma-3-27b-it'
        print(f"⚠️  Could not list models, using default: {selected_model}")
    
    # Get URL
    url = input("\n🌐 Enter the URL to scrape: ").strip()
    if not url:
        print("❌ URL is required!")
        return
    
    # Initialize agent
    agent = GemmaAgent(api_key, selected_model)
    
    # Step 1: Fetch page structure
    print(f"\n⏳ Fetching page structure from {url}...")
    html_content, structural_map = await fetch_page_structure(url)
    
    if not html_content or not structural_map:
        print("❌ Failed to fetch page structure!")
        return

    rand_id = random_num#random.randint(10000, 99999)

    html_filename = f"html_files/gemma_html_content_{rand_id}.html"
    with open(html_filename, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"💾 Saved HTML to: {html_filename}")

    structural_map_filename = f"structural_maps/gemma_structural_map_{rand_id}.json"
    with open(structural_map_filename, "w", encoding="utf-8") as f:
        json.dump(structural_map, f, indent=2, ensure_ascii=False)
    print(f"💾 Saved structural map to: {structural_map_filename}")

    print("✓ Page structure fetched successfully!")
    
    structural_map_json = json.dumps(structural_map, indent=2)
    
    # Step 2: Generate extraction code
    print("\n⏳ Generating extraction code with Gemma...")
    
    retry_count = 0
    error_context = None
    
    while retry_count <= MAX_RETRIES:
        extraction_code = agent.generate_extraction_code(
            structural_map_json, 
            page_url=url,
            error_context=error_context
        )
        
        print("✓ Code generated!")
        
        gen_code_filename = f"code/gemma_gen_code_{rand_id}.py"
        with open(gen_code_filename, "w", encoding="utf-8") as f:
            f.write(extraction_code)
        print(f"💾 Saved generated code to: {gen_code_filename}")
        
        print("\n" + "-"*60)
        print("GENERATED CODE:")
        print("-"*60)
        print(extraction_code)
        print("-"*60)
        
        # Step 3: Execute code
        print("\n⏳ Executing extraction code...")
        success, result = execute_extraction_code(extraction_code, html_content)
        
        if success:
            stats = analyze_output(result)
            display_sample_output(result, stats)
            
            print("\n❓ Are you satisfied with the results?")
            print("   [s] Save to JSON")
            print("   [r] Retry (generate new code)")
            print("   [q] Quit")
            
            decision = input("   → ").strip().lower()
            
            if decision == 's':
                output_filename = f"Gemma_extracted_data_{url.split('//')[1].split('/')[0].replace('.', '_')}.json"
                with open(output_filename, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                print(f"\n✅ Data saved to: {output_filename}")
                break
                
            elif decision == 'r':
                if retry_count >= MAX_RETRIES:
                    print(f"\n❌ Maximum retries ({MAX_RETRIES}) reached.")
                    break
                print("\n🔄 Retrying...")
                error_context = "User requested retry. Improve the extraction logic."
                retry_count += 1
                
            else:
                print("\n👋 Exiting without saving.")
                break
                
        else:
            print(f"\n❌ EXECUTION FAILED!")
            print(f"Error: {result}")
            
            if retry_count >= MAX_RETRIES:
                print(f"\nMaximum retries ({MAX_RETRIES}) reached.")
                break
            
            print("\n❓ Would you like to retry?")
            print("   [y] Yes, retry with error context")
            print("   [n] No, quit")
            
            decision = input("   → ").strip().lower()
            
            if decision == 'y':
                print("\n🔄 Retrying with error context...")
                error_context = result
                retry_count += 1
            else:
                print("\n👋 Exiting.")
                break
    
    print("\n" + "="*60)
    print("🏁 Agent finished!")
    print("="*60)


# --- CLI (non-interactive) mode for orchestration ---
async def main_cli(url: str, api_key: str, model: str = 'gemma-3-27b-it', max_retries: int = 0):
    """Non-interactive entry point. Returns paths via JSON line on stdout.

    Prints a JSON object on success:
      {"status": "ok", "code_file": "...", "output_file": "...", "data": {...}}
    Or on failure:
      {"status": "error", "error": "..."}
    """
    agent = GemmaAgent(api_key, model)

    print(f"⏳ Fetching page structure from {url}...", flush=True)
    html_content, structural_map = await fetch_page_structure(url)

    if not html_content or not structural_map:
        print("ORCH_RESULT:" + json.dumps({"status": "error", "error": "Failed to fetch page structure"}), flush=True)
        return

    rand_id = random_num

    html_filename = f"html_files/gemma_html_content_{rand_id}.html"
    with open(html_filename, "w", encoding="utf-8") as f:
        f.write(html_content)

    structural_map_filename = f"structural_maps/gemma_structural_map_{rand_id}.json"
    with open(structural_map_filename, "w", encoding="utf-8") as f:
        json.dump(structural_map, f, indent=2, ensure_ascii=False)

    structural_map_json = json.dumps(structural_map, indent=2)

    retry_count = 0
    error_context = None

    while retry_count <= max_retries:
        extraction_code = agent.generate_extraction_code(
            structural_map_json,
            page_url=url,
            error_context=error_context
        )

        gen_code_filename = f"code/gemma_gen_code_{rand_id}.py"
        with open(gen_code_filename, "w", encoding="utf-8") as f:
            f.write(extraction_code)

        success, result = execute_extraction_code(extraction_code, html_content)

        if success:
            output_filename = f"Gemma_extracted_links_{rand_id}.json"
            with open(output_filename, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            print("ORCH_RESULT:" + json.dumps({
                "status": "ok",
                "code_file": gen_code_filename,
                "output_file": output_filename,
                "data": result
            }), flush=True)
            return
        else:
            print(f"  ❌ Attempt {retry_count + 1} failed: {str(result)}", flush=True)
            error_context = result
            retry_count += 1

    print("ORCH_RESULT:" + json.dumps({
        "status": "error",
        "error": f"All {max_retries + 1} attempt(s) failed. Last error: {str(error_context)}"
    }), flush=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Links Agent (Gemma Edition)")
    parser.add_argument("--url", type=str, help="URL to extract links from (CLI mode)")
    parser.add_argument("--api-key", type=str, help="Gemini API key (CLI mode)")
    parser.add_argument("--model", type=str, default="gemma-3-27b-it", help="Model name")
    parser.add_argument("--max-retries", type=int, default=0, help="Max retries in CLI mode (default: 0 = single attempt)")
    args = parser.parse_args()

    if args.url and args.api_key:
        # Non-interactive CLI mode
        asyncio.run(main_cli(args.url, args.api_key, args.model, args.max_retries))
    else:
        # Interactive mode (original behavior)
        asyncio.run(main())
