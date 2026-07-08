"""
LLM:  Gemma (gemma-3-27b-it / gemma-3-12b-it via Google AI API)

Gemma-targeted Links Agent with error-propagation fixes:
- _launch_and_fetch now returns (html, status, error) and always closes the browser
- _fetch_with_requests returns (html, error)
- fetch_page_structure aggregates every attempt's error into _LAST_FETCH_ERROR
  and now falls back to plain HTTP on *crash* too, not only on Cloudflare challenge
- main_cli / main surface the real error instead of a generic message
- output directories are created up-front
"""
import asyncio
import json
import re
import os
import random
import time
import traceback
import requests
from typing import Dict, List, Optional, Tuple
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
import google.generativeai as genai
from urllib.parse import urljoin, urlparse

# Shared core: LLM retry/token helpers + the Cloudflare-resistant page fetch.
from utils import (
    _generate_with_retry, get_token_usage, reset_token_usage,
    list_available_models,
    fetch_page_structure as _utils_fetch_page_structure,
    _is_challenge_page, get_last_fetch_error,
)

random_num = random.randint(10000, 99999)

# --- Configuration ---
TARGET_TAGS = ['div', 'section', 'article', 'main', 'header', 'footer', 'nav',
               'ul', 'ol', 'li', 'a', 'h1', 'h2', 'h3', 'h4', 'p', 'span',
               'table', 'tr', 'td', 'th', 'figure', 'figcaption', 'time', 'img']
MAX_DEPTH = 10
MAX_RETRIES = 3

ALLOWED_IMPORTS = {
    'BeautifulSoup': BeautifulSoup,
    're': re,
    'json': json,
    'urljoin': urljoin,
    'urlparse': urlparse,
}


def _ensure_dirs():
    """Make sure output folders exist before we try to write into them."""
    for d in ("html_files", "structural_maps", "code"):
        os.makedirs(d, exist_ok=True)


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


async def fetch_page_structure(url: str) -> Tuple[Optional[str], Optional[List[Dict]]]:
    """Fetch a listing page (Cloudflare-resistant) and build the links map.

    Thin wrapper over utils.fetch_page_structure that passes this module's
    link-oriented create_structural_map. On total failure returns (None, None)
    and get_last_fetch_error() holds the aggregated cause.
    """
    return await _utils_fetch_page_structure(url, create_structural_map)


# ═══════════════════════════════════════════════════════════
# FEW-SHOT EXAMPLE
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
            response = _generate_with_retry(self.model, prompt)
            code = response.text

            fenced = re.findall(r'```python\s*\n(.*?)```', code, re.DOTALL)
            if fenced:
                code = fenced[-1].strip()
            else:
                lines = code.split('\n')
                func_start = None
                for i, line in enumerate(lines):
                    if re.match(r'^def\s+\w+\s*\(', line):
                        func_start = i
                if func_start is not None:
                    code = '\n'.join(lines[func_start:])

            code = re.sub(r'<end_of_turn>\s*$', '', code)
            code = re.sub(r'^```\w*\s*$', '', code, flags=re.MULTILINE)
            code = code.strip()

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

    if all(line.strip() == '' or line.strip().startswith('#') for line in code.splitlines()):
        return False, f"LLM ERROR: {code.replace('# ', '').strip()}"

    is_safe, safety_msg = validate_code_safety(code)
    if not is_safe:
        return False, f"SAFETY ERROR: {safety_msg}"

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

    restricted_globals = {
        '__builtins__': {
            'print': print, 'len': len, 'str': str, 'int': int, 'float': float,
            'bool': bool, 'list': list, 'dict': dict, 'set': set, 'tuple': tuple,
            'range': range, 'enumerate': enumerate, 'zip': zip, 'filter': filter,
            'map': map, 'sorted': sorted, 'any': any, 'all': all, 'max': max,
            'min': min, 'sum': sum, 'None': None, 'True': True, 'False': False,
            'isinstance': isinstance, 'hasattr': hasattr, 'getattr': getattr,
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
        pre_exec_keys = set(restricted_globals.keys())

        exec(code, restricted_globals)

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

        if isinstance(result, list) and len(result) == 1 and isinstance(result[0], dict):
            result = result[0]

        return True, result

    except Exception as e:
        error_detail = traceback.format_exc()
        return False, f"EXECUTION ERROR: {str(e)}\n\n{error_detail}"


# --- Output Analysis ---
def analyze_output(data: Dict) -> Dict:
    """Generate statistics about the extracted data."""
    stats = {'total_fields': len(data), 'fields': list(data.keys()), 'field_details': {}}

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
            stats['field_details'][key] = {'type': type(value).__name__, 'value': str(value)}

    return stats


def display_sample_output(data: Dict, stats: Dict):
    """Display sample output and statistics to the user."""
    print("\n" + "=" * 60)
    print("📊 EXTRACTION RESULTS")
    print("=" * 60)
    print(f"\n✓ Total fields extracted: {stats['total_fields']}")
    print(f"✓ Fields: {', '.join(stats['fields'])}")
    print("\n" + "-" * 60)
    print("SAMPLE OUTPUT:")
    print("-" * 60)

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

    print("\n" + "=" * 60)


# --- Main Agent Logic ---
async def main():
    _ensure_dirs()
    print("=" * 60)
    print("🤖 AI WEB SCRAPING AGENT (Gemma Edition)")
    print("=" * 60)

    api_key = input("\n🔑 Enter your Gemini API key: ").strip()
    if not api_key:
        print("❌ API key is required!")
        return

    print("\n⏳ Checking available models...")
    available_models = await list_available_models(api_key)

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

    url = input("\n🌐 Enter the URL to scrape: ").strip()
    if not url:
        print("❌ URL is required!")
        return

    agent = GemmaAgent(api_key, selected_model)

    print(f"\n⏳ Fetching page structure from {url}...")
    html_content, structural_map = await fetch_page_structure(url)

    if html_content is None:
        print("❌ Failed to fetch page structure!")
        if get_last_fetch_error():
            print(get_last_fetch_error())
        return

    if not structural_map:
        print("⚠  Structural map is empty — will pass raw HTML to LLM directly.")

    rand_id = random_num

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

    print("\n⏳ Generating extraction code with Gemma...")

    retry_count = 0
    error_context = None

    while retry_count <= MAX_RETRIES:
        extraction_code = agent.generate_extraction_code(
            structural_map_json, page_url=url, error_context=error_context
        )
        print("✓ Code generated!")

        gen_code_filename = f"code/gemma_gen_code_{rand_id}.py"
        with open(gen_code_filename, "w", encoding="utf-8") as f:
            f.write(extraction_code)
        print(f"💾 Saved generated code to: {gen_code_filename}")

        print("\n" + "-" * 60)
        print("GENERATED CODE:")
        print("-" * 60)
        print(extraction_code)
        print("-" * 60)

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

    print("\n" + "=" * 60)
    print("🏁 Agent finished!")
    print("=" * 60)


# --- CLI (non-interactive) mode for orchestration ---
async def main_cli(url: str, api_key: str, model: str = 'gemma-3-27b-it', max_retries: int = 0):
    """Non-interactive entry point. Returns paths via JSON line on stdout."""
    _ensure_dirs()
    agent = GemmaAgent(api_key, model)

    reset_token_usage()
    print(f"⏳ Fetching page structure from {url}...", flush=True)
    html_content, structural_map = await fetch_page_structure(url)

    if html_content is None:
        detail = get_last_fetch_error() or "no further detail captured"
        print("ORCH_RESULT:" + json.dumps({
            "status": "error",
            "error": f"Failed to fetch page structure for {url}\n\n{detail}"
        }), flush=True)
        return

    if not structural_map:
        print("  ⚠  Structural map is empty — proceeding with empty map (LLM will use raw HTML).", flush=True)

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
            structural_map_json, page_url=url, error_context=error_context
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
                "data": result,
                "token_usage": get_token_usage()
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
    parser.add_argument("--max-retries", type=int, default=0, help="Max retries in CLI mode (default: 0)")
    args = parser.parse_args()

    if args.url and args.api_key:
        asyncio.run(main_cli(args.url, args.api_key, args.model, args.max_retries))
    else:
        asyncio.run(main())