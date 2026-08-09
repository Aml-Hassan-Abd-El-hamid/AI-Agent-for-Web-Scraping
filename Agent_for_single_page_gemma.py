import asyncio
import json
import os
import re
import time
import random
from typing import Dict, List, Optional, Tuple
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
import google.generativeai as genai
from urllib.parse import urljoin, urlparse

# Cloudflare-resistant page fetch lives in the shared core (utils), so this
# agent no longer depends on the Links agent.
from utils import (
    _generate_with_retry, get_token_usage, reset_token_usage,
    list_available_models,
    fetch_page_structure as _utils_fetch_page_structure,
    count_tokens, input_token_budget, DEFAULT_INPUT_TOKEN_BUDGET,
    execute_generated_code_sandboxed,
    analyze_output, display_sample_output,
    fit_structural_map_to_budget,
)

random_num = random.randint(10000, 99999)

# --- Configuration ---
TARGET_TAGS = ['div', 'section', 'article', 'main', 'header', 'footer', 'nav',
               'ul', 'ol', 'li', 'a', 'h1', 'h2', 'h3', 'h4', 'p', 'span',
               'table', 'tr', 'td', 'th', 'figure', 'figcaption', 'time', 'img']
MAX_DEPTH = 16   # modern article pages nest the byline/date deep behind CSS-in-JS
                 # wrapper divs (e.g. arageek's author link sits at DOM depth ~17),
                 # so a shallow map hid them and the agent returned author/date =
                 # N/A. A shallow page's map is unchanged by a higher cap (recursion
                 # stops at leaves), and _fit_map_to_budget shrinks any page whose
                 # deeper map would exceed the input-token budget.
MIN_MAP_DEPTH = 3      # floor when shrinking a too-large map to fit the token budget
MAX_RETRIES = 3
SANDBOX_TIMEOUT_SECONDS = 30
SANDBOX_MAX_STDOUT = 1_000_000
SANDBOX_MAX_STRING_LENGTH = 500_000
SANDBOX_MAX_LIST_ITEMS = 5_000
SANDBOX_MAX_DICT_KEYS = 200

# Output folders
CODE_DIR = "Agent_for_single_page_gemma_code"
HTML_DIR = "Agent_for_single_page_gemma_html"
MAP_DIR = "Agent_for_single_page_gemma_structural_maps"
OUTPUT_DIR = "Agent_for_single_page_gemma_output"
for _d in (CODE_DIR, HTML_DIR, MAP_DIR, OUTPUT_DIR):
    os.makedirs(_d, exist_ok=True)

# Allowed imports for generated code
ALLOWED_IMPORTS = {
    'BeautifulSoup': BeautifulSoup,
    're': re,
    'json': json,
    'urljoin': urljoin,
    'urlparse': urlparse,
}

# ═══════════════════════════════════════════════════════════
# FEW-SHOT EXAMPLE (teaches Gemma the exact input→output pattern)
# ═══════════════════════════════════════════════════════════
_FEW_SHOT_MAP = '''[
  {"tag": "article", "attributes": {"class": "post-detail"}, "children": [
    {"tag": "h1", "attributes": {"class": "post-title"}, "text_snippet": "Breaking News: Major Event Unfolds"},
    {"tag": "div", "attributes": {"class": "post-meta"}, "children": [
      {"tag": "span", "attributes": {"class": "author"}, "text_snippet": "John Doe"},
      {"tag": "time", "attributes": {"class": "date"}, "text_snippet": "March 15, 2024"}
    ]},
    {"tag": "div", "attributes": {"class": "post-body"}, "children": [
      {"tag": "p", "attributes": {}, "text_snippet": "First paragraph of the article..."},
      {"tag": "p", "attributes": {}, "text_snippet": "Second paragraph continues here..."}
    ]}
  ]}
]'''

_FEW_SHOT_CODE = '''def extract_data(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    data = {}

    # Title
    title_tag = soup.select_one('article.post-detail h1.post-title')
    data['title'] = title_tag.get_text(strip=True) if title_tag else 'N/A'

    # Author
    author_tag = soup.select_one('.post-meta span.author')
    data['author'] = author_tag.get_text(strip=True) if author_tag else 'N/A'

    # Date
    date_tag = soup.select_one('.post-meta time.date')
    data['date'] = date_tag.get_text(strip=True) if date_tag else 'N/A'

    # Body text
    body_div = soup.select_one('div.post-body')
    if body_div:
        paragraphs = body_div.find_all(['p', 'li'])
        data['body_text'] = '\\n'.join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))
    else:
        data['body_text'] = 'N/A'

    return data'''

# --- Structural Map Generation ---
# Byline/date live deep inside aside wrappers on modern news pages, so when a
# large page's map is shrunk to fit the token budget those nodes get cut and the
# agent returns author/date = N/A. These signals let us *pin* such nodes: even
# past the depth cutoff we keep the few elements that look like an author or a
# date, so the agent can always see a selector for them.
_SEMANTIC_SIGNALS = (
    'author', 'byline', 'writer', 'contributor',
    'date', 'published', 'pubdate', 'timestamp', 'time-details',
)
_SEMANTIC_ITEMPROPS = ('author', 'datepublished', 'datemodified', 'datecreated')
_MAX_PINNED_NODES = 12   # cap pinned nodes so the map stays small
_MAX_PINNED_SCAN = 2000  # cap subtree scan so huge pages stay fast
_METADATA_KEYS = {
    'article:published_time', 'article:modified_time', 'author',
    'date', 'datepublished', 'datemodified', 'og:title',
}


def _structural_attributes(el) -> Dict:
    attributes = {}
    if el.get('class'):
        attributes['class'] = " ".join(el.get('class')[:2])
    if el.get('id'):
        attributes['id'] = el.get('id')
    for name in ('datetime', 'itemprop', 'property', 'name'):
        if el.get(name):
            attributes[name] = el.get(name)
    if el.name in {'meta', 'time'} and el.get('content'):
        attributes['content'] = str(el.get('content'))[:200]
    return attributes


def _collect_document_metadata(soup) -> List[Dict]:
    metadata = []
    for element in soup.select('meta[property], meta[name], meta[itemprop], time[datetime]'):
        key = str(
            element.get('property') or element.get('name') or element.get('itemprop') or ''
        ).lower()
        if element.name != 'time' and key not in _METADATA_KEYS:
            continue
        metadata.append(_make_pinned_node(element))
        if len(metadata) >= _MAX_PINNED_NODES:
            break
    return metadata


def _is_semantic_node(el) -> bool:
    """True if *el* looks like an author or date field worth pinning."""
    name = getattr(el, 'name', None)
    if not name:
        return False
    name = name.lower()
    if name == 'time':
        return True
    if el.get('datetime'):
        return True
    itemprop = (el.get('itemprop') or '').lower()
    if itemprop in _SEMANTIC_ITEMPROPS:
        return True
    rel = el.get('rel')
    if rel and any(r.lower() == 'author' for r in rel):
        return True
    ident = (" ".join(el.get('class') or []) + " " + (el.get('id') or "")).lower()
    return any(sig in ident for sig in _SEMANTIC_SIGNALS)


def _make_pinned_node(el) -> Dict:
    """Build a compact map node for a pinned author/date element."""
    node = {'tag': el.name.lower(), 'attributes': _structural_attributes(el)}
    text_content = el.get_text(strip=True)
    if text_content:
        node['text_snippet'] = (
            text_content[:50].replace('\n', ' ') + ('...' if len(text_content) > 50 else '')
        )
    return node


def _collect_pinned_nodes(element) -> List[Dict]:
    """Scan below a truncated element for author/date nodes and pin them.

    Returns compact nodes (deduped by tag+class+id) so a depth-shrunk map still
    exposes byline/date selectors that would otherwise be cut off.
    """
    pinned: List[Dict] = []
    seen_keys = set()
    scanned = 0
    for desc in element.descendants:
        scanned += 1
        if scanned > _MAX_PINNED_SCAN or len(pinned) >= _MAX_PINNED_NODES:
            break
        name = getattr(desc, 'name', None)
        if not name or name.lower() not in TARGET_TAGS:
            continue
        if not _is_semantic_node(desc):
            continue
        key = (name.lower(), " ".join(desc.get('class') or []), desc.get('id') or "")
        if key in seen_keys:
            continue
        seen_keys.add(key)
        pinned.append(_make_pinned_node(desc))
    return pinned


def create_structural_map(soup: BeautifulSoup, depth: int = 0, max_depth: int = None) -> List[Dict]:
    """Recursively generates a simplified, nested structural map of the HTML.

    *max_depth* defaults to MAX_DEPTH; callers pass a smaller value to shrink an
    over-large map so the prompt fits the input-token budget. At the depth
    cutoff, author/date nodes deeper in the subtree are still *pinned* so a
    shrunk map keeps byline/date selectors visible to the agent.
    """
    if max_depth is None:
        max_depth = MAX_DEPTH
    if depth >= max_depth:
        return _collect_pinned_nodes(soup)

    structure = _collect_document_metadata(soup) if depth == 0 else []
    
    for child in soup.children:
        if child.name and child.name.lower() in TARGET_TAGS:
            node = {
                'tag': child.name.lower(),
                'attributes': _structural_attributes(child),
                'children': create_structural_map(child, depth + 1, max_depth)
            }
            
            if not node['children'] and child.text and len(child.text.strip()) > 5:
                text_content = child.text.strip()
                node['text_snippet'] = text_content[:50].replace('\n', ' ') + ('...' if len(text_content) > 50 else '')

            structure.append(node)
            
    return structure


async def fetch_page_structure(url: str) -> Tuple[Optional[str], Optional[List[Dict]]]:
    """Fetch an article page (Cloudflare-resistant) and build its structural map.

    Thin wrapper over utils.fetch_page_structure that passes this module's
    content-tuned create_structural_map, so the single-page extraction prompt
    is unchanged.
    """
    return await _utils_fetch_page_structure(url, create_structural_map)


# --- LLM Integration ---
class GemmaAgent:
    """LLM agent using Gemma's prompt format with few-shot examples."""

    def __init__(self, api_key: str, model_name: str = 'gemma-3-27b-it'):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.conversation_history = []
        self.model_name = model_name
    
    def _build_prompt(self, structural_map: str, user_requirements: str = "", page_url: str = "", error_context: Optional[str] = None) -> str:
        """Assemble the Gemma prompt (shared by code generation and token counting)."""
        if not error_context:
            # ── Few-shot turn 1: teach the pattern ──────────────
            return f"""<start_of_turn>user
You are a Python web-scraping expert. You will receive an HTML structural map (JSON), a target URL, and user requirements describing what data to extract. Your job is to write a single Python function called `extract_data(html_content)` that extracts the requested data from the page.

Rules you MUST follow:
- The function signature is exactly: def extract_data(html_content):
- Use BeautifulSoup to parse the HTML.
- Return a dict with keys matching the requested fields.
- Use urljoin(base_url, href) to resolve relative URLs when extracting links. Derive base_url from the target URL.
- Only use selectors (tags, classes, ids) that appear in the structural map. Do NOT invent selectors.
- For long/body text, select the CONTAINER by its class or id, then collect text from ALL of its block children with a class-agnostic call like container.find_all(['p', 'li', 'h2', 'h3', 'blockquote']). Do NOT filter those children by their own class (e.g. avoid find_all('p', class_='wp-block-paragraph')) — other pages built from the same template often use plain <p>/<li> or a different class, so a class-specific selector returns nothing for them.
- Return 'N/A' ONLY if no matching tag exists in the structural map AND all fallbacks fail.
- Do NOT include import statements. Only output the function body.
- Available in scope: BeautifulSoup, re, json, urljoin, urlparse.

Think step-by-step:
1. Read the user requirements to understand what fields are needed.
2. Identify which node(s) in the structural map correspond to each requested field.
3. Pick CSS selectors directly derived from those nodes.
4. Write the function.

Here is an example:

USER REQUIREMENTS: Extract the title, author, date, and body text.
TARGET URL: https://example.com/article/123

STRUCTURAL MAP:
{_FEW_SHOT_MAP}
<end_of_turn>
<start_of_turn>model
{_FEW_SHOT_CODE}
<end_of_turn>
<start_of_turn>user
Good. Now do the same for this real page.

USER REQUIREMENTS: {user_requirements}
TARGET URL: {page_url}

STRUCTURAL MAP:
{structural_map}

Write only the Python function. No explanation, no imports.
<end_of_turn>
<start_of_turn>model
"""
        else:
            # ── Retry turn: include the error ──────────────────
            return f"""<start_of_turn>user
You are a Python web-scraping expert. Your previous code failed. Fix it.

Here is a working example for reference:

USER REQUIREMENTS: Extract the title, author, date, and body text.
TARGET URL: https://example.com/article/123
STRUCTURAL MAP:
{_FEW_SHOT_MAP}

Working code:
{_FEW_SHOT_CODE}

Now here is the REAL task that failed:

USER REQUIREMENTS: {user_requirements}
TARGET URL: {page_url}

STRUCTURAL MAP:
{structural_map}

ERROR from the previous attempt:
{error_context}

Fix the code. Same rules:
- Function signature: def extract_data(html_content):
- Return a dict with keys matching the requested fields.
- Use urljoin(base_url, href) for relative URLs.  base_url = '/'.join('{page_url}'.split('/')[:3])
- Only use selectors from the structural map.
- For body text, gather ALL block children of the container with find_all(['p', 'li', 'h2', 'h3', 'blockquote']) WITHOUT filtering by their class.
- Available: BeautifulSoup, re, json, urljoin, urlparse
- Output ONLY the corrected function. No explanation.
<end_of_turn>
<start_of_turn>model
"""

    def generate_extraction_code(self, structural_map: str, user_requirements: str = "", page_url: str = "", error_context: Optional[str] = None) -> str:
        """Generate Python extraction code using Gemma's prompt format with few-shot learning."""
        prompt = self._build_prompt(structural_map, user_requirements, page_url, error_context)
        try:
            response = _generate_with_retry(self.model, prompt)
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
            code_filename = f"{CODE_DIR}/generated_code_{random_num}.py"
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
            code_filename = f"{CODE_DIR}/generated_code_{random_num}_ERROR.py"
            with open(code_filename, 'w', encoding='utf-8') as f:
                f.write(error_code)
            print(f"❌ LLM call failed: {error_msg[:200]}")
            return error_code


# --- Safe Code Execution ---
def _validate_json_value(value, depth: int = 0) -> Tuple[bool, str]:
    if depth > 10:
        return False, "JSON output is nested too deeply"
    if value is None or isinstance(value, (bool, int, float)):
        return True, "ok"
    if isinstance(value, str):
        if len(value) > SANDBOX_MAX_STRING_LENGTH:
            return False, "String output exceeds maximum length"
        return True, "ok"
    if isinstance(value, list):
        if len(value) > SANDBOX_MAX_LIST_ITEMS:
            return False, "List output exceeds maximum length"
        for item in value:
            ok, reason = _validate_json_value(item, depth + 1)
            if not ok:
                return ok, reason
        return True, "ok"
    if isinstance(value, dict):
        if len(value) > SANDBOX_MAX_DICT_KEYS:
            return False, "Object output has too many keys"
        for key, item in value.items():
            if not isinstance(key, str) or not key or len(key) > 100 or key.startswith('__'):
                return False, "Object output contains an invalid key"
            ok, reason = _validate_json_value(item, depth + 1)
            if not ok:
                return ok, reason
        return True, "ok"
    return False, f"Output contains non-JSON value: {type(value).__name__}"


def _validate_extraction_output(result) -> Tuple[bool, str, Dict]:
    """Validate the generated extractor's result against the single-page contract."""
    if isinstance(result, list) and len(result) == 1 and isinstance(result[0], dict):
        result = result[0]
    if not isinstance(result, dict):
        return False, "Output must be a JSON object", {}
    ok, reason = _validate_json_value(result)
    if not ok:
        return False, reason, {}
    return True, "Output schema is valid", result


def _space_around_tags(html_content: str) -> str:
    """Insert a space around every tag boundary so text separated by any tag
    (a, span, strong, bdi, time, ...) does not get glued into one word when
    extracted. Runs of whitespace are collapsed again by _normalize_whitespace,
    so this only ever adds missing word separators, it never changes real words."""
    return re.sub(r"(<[^>]+>)", r" \1 ", html_content)


def _normalize_whitespace(text: str) -> str:
    """Collapse runs of spaces/tabs (keeping line breaks) and trim each line."""
    text = re.sub(r"[^\S\n]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    return text.strip()


def execute_extraction_code(code: str, html_content: str) -> Tuple[bool, any]:
    """Execute generated code in a subprocess with validation and time limits."""
    html_content = _space_around_tags(html_content)
    ok, result = execute_generated_code_sandboxed(
        code,
        html_content,
        _validate_extraction_output,
        timeout_seconds=SANDBOX_TIMEOUT_SECONDS,
        max_stdout=SANDBOX_MAX_STDOUT,
    )
    if ok and isinstance(result, dict):
        result = {k: (_normalize_whitespace(v) if isinstance(v, str) else v) for k, v in result.items()}
    return ok, result

# --- Main Agent Logic ---
def _fit_map_to_budget(agent, html_content, structural_map, page_url, requirements, budget):
    """Fit the article map while retaining article-specific prompt arguments."""
    body = None

    def rebuild_map(depth):
        nonlocal body
        if body is None:
            soup = BeautifulSoup(html_content, 'lxml')
            body = soup.body if soup.body else soup
        return create_structural_map(body, max_depth=depth)

    return fit_structural_map_to_budget(
        agent.model,
        structural_map,
        MAX_DEPTH,
        MIN_MAP_DEPTH,
        budget,
        lambda map_json: agent._build_prompt(map_json, requirements, page_url),
        rebuild_map,
    )


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
    
    # Get extraction requirements
    print("\n What data do you want to extract?")
    print("   (e.g., 'title, date, article text, author name, links to related articles')")
    requirements = input("   → ").strip()
    if not requirements:
        print("❌ Requirements are required!")
        return
    
    # Initialize agent
    agent = GemmaAgent(api_key, selected_model)
    
    # Step 1: Fetch page structure
    print(f"\n⏳ Fetching page structure from {url}...")
    html_content, structural_map = await fetch_page_structure(url)
    
    if not html_content or not structural_map:
        print("❌ Failed to fetch page structure!")
        return

    rand_id = random_num

    html_filename = f"{HTML_DIR}/html_content_{rand_id}.html"
    with open(html_filename, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"💾 Saved HTML to: {html_filename}")

    structural_map_filename = f"{MAP_DIR}/structural_map_{rand_id}.json"
    with open(structural_map_filename, "w", encoding="utf-8") as f:
        json.dump(structural_map, f, indent=2, ensure_ascii=False)
    print(f"💾 Saved structural map to: {structural_map_filename}")

    print("✓ Page structure fetched successfully!")
    
    budget = input_token_budget(selected_model)
    structural_map, structural_map_json, depth_used, ntok = _fit_map_to_budget(
        agent, html_content, structural_map, url, requirements, budget)
    if depth_used < MAX_DEPTH:
        print(f"⚠  Large page (~{ntok} prompt tokens, budget {budget}): reduced "
              f"structural-map depth to {depth_used} to fit the input-token quota.")
    
    # Step 2: Generate extraction code
    print("\n⏳ Generating extraction code with Gemma...")
    
    retry_count = 0
    error_context = None
    
    while retry_count <= MAX_RETRIES:
        extraction_code = agent.generate_extraction_code(
            structural_map_json, 
            user_requirements=requirements,
            page_url=url,
            error_context=error_context
        )
        
        print("✓ Code generated!")
        
        gen_code_filename = f"{CODE_DIR}/gen_code_{rand_id}.py"
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
                output_filename = f"{OUTPUT_DIR}/Gemma_extracted_data_{url.split('//')[1].split('/')[0].replace('.', '_')}.json"
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
def build_structure_from_html(html_content: str) -> Tuple[Optional[str], Optional[List[Dict]]]:
    """Build a structural map from already-fetched HTML (no network/browser).

    Mirrors utils.fetch_page_structure's mapping step so the map matches the
    exact HTML the extraction code will later run against. This avoids a subtle
    failure mode where the map is built from a JS-rendered DOM (with dynamic
    classes like ``active`` that JavaScript adds at runtime) while extraction
    runs on static HTML that lacks those classes — making the LLM pick selectors
    that match nothing.
    """
    if not html_content:
        return None, None
    soup = BeautifulSoup(html_content, 'lxml')
    structural_map = create_structural_map(soup.body if soup.body else soup)
    return html_content, structural_map


async def main_cli(url: str, api_key: str, requirements: str, model: str = 'gemma-3-27b-it', max_retries: int = 0,
                   max_input_tokens: int = DEFAULT_INPUT_TOKEN_BUDGET, html_file: Optional[str] = None):
    """Non-interactive entry point. Returns paths via JSON line on stdout.

    Prints a JSON object on success:
      {"status": "ok", "code_file": "...", "output_file": "...", "data": {...}}
    Or on failure:
      {"status": "error", "error": "..."}

    When *html_file* is given, the structural map is built from that exact HTML
    instead of re-fetching the page, so the map and the extraction target are
    the same DOM.
    """
    agent = GemmaAgent(api_key, model)

    reset_token_usage()
    if html_file:
        print(f"⏳ Building page structure from provided HTML: {html_file}...", flush=True)
        try:
            with open(html_file, "r", encoding="utf-8") as f:
                provided_html = f.read()
        except OSError as exc:
            print(json.dumps({"status": "error", "error": f"Failed to read HTML file: {exc}"}))
            return
        html_content, structural_map = build_structure_from_html(provided_html)
    else:
        print(f"⏳ Fetching page structure from {url}...", flush=True)
        html_content, structural_map = await fetch_page_structure(url)

    if not html_content or not structural_map:
        print(json.dumps({"status": "error", "error": "Failed to fetch page structure"}))
        return

    rand_id = random_num

    html_filename = f"{HTML_DIR}/html_content_{rand_id}.html"
    with open(html_filename, "w", encoding="utf-8") as f:
        f.write(html_content)

    structural_map_filename = f"{MAP_DIR}/structural_map_{rand_id}.json"
    with open(structural_map_filename, "w", encoding="utf-8") as f:
        json.dump(structural_map, f, indent=2, ensure_ascii=False)

    budget = input_token_budget(model, max_input_tokens)
    structural_map, structural_map_json, depth_used, ntok = _fit_map_to_budget(
        agent, html_content, structural_map, url, requirements, budget)
    if depth_used < MAX_DEPTH:
        print(f"  ⚠  Large page (~{ntok} prompt tokens, budget {budget}): reduced "
              f"structural-map depth to {depth_used} to fit the input-token quota.", flush=True)

    retry_count = 0
    error_context = None

    while retry_count <= max_retries:
        extraction_code = agent.generate_extraction_code(
            structural_map_json,
            user_requirements=requirements,
            page_url=url,
            error_context=error_context
        )

        gen_code_filename = f"{CODE_DIR}/gen_code_{rand_id}.py"
        with open(gen_code_filename, "w", encoding="utf-8") as f:
            f.write(extraction_code)

        success, result = execute_extraction_code(extraction_code, html_content)

        if success:
            output_filename = f"{OUTPUT_DIR}/Gemma_extracted_data_{rand_id}.json"
            with open(output_filename, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            # Emit structured result as the LAST line
            print("ORCH_RESULT:" + json.dumps({
                "status": "ok",
                "code_file": gen_code_filename,
                "output_file": output_filename,
                "data": result,
                "map_depth": depth_used,
                "input_tokens": ntok,
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

    parser = argparse.ArgumentParser(description="AI Web Scraping Agent (Gemma Edition)")
    parser.add_argument("--url", type=str, help="URL to scrape (CLI mode)")
    parser.add_argument("--api-key", type=str, help="Gemini API key (CLI mode)")
    parser.add_argument("--requirements", type=str, help="Extraction requirements (CLI mode)")
    parser.add_argument("--model", type=str, default="gemma-3-27b-it", help="Model name")
    parser.add_argument("--max-retries", type=int, default=0, help="Max retries in CLI mode (default: 0 = single attempt)")
    parser.add_argument("--max-input-tokens", type=int, default=DEFAULT_INPUT_TOKEN_BUDGET,
                        help=f"Per-request input-token budget (default {DEFAULT_INPUT_TOKEN_BUDGET}, sized for the free-tier per-minute cap; raise for paid tiers/larger models)")
    parser.add_argument("--html-file", type=str, default=None,
                        help="Build the structural map from this local HTML file instead of re-fetching the URL (keeps the map and extraction target identical)")
    args = parser.parse_args()

    if args.url and args.api_key and args.requirements:
        # Non-interactive CLI mode
        asyncio.run(main_cli(args.url, args.api_key, args.requirements, args.model, args.max_retries, args.max_input_tokens, args.html_file))
    else:
        # Interactive mode (original behavior)
        asyncio.run(main())
