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
)

random_num = random.randint(10000, 99999)

# --- Configuration ---
TARGET_TAGS = ['div', 'section', 'article', 'main', 'header', 'footer', 'nav',
               'ul', 'ol', 'li', 'a', 'h1', 'h2', 'h3', 'h4', 'p', 'span',
               'table', 'tr', 'td', 'th', 'figure', 'figcaption', 'time', 'img']
MAX_DEPTH = 10
MIN_MAP_DEPTH = 3      # floor when shrinking a too-large map to fit the token budget
MAX_RETRIES = 3

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
        paragraphs = body_div.find_all('p')
        data['body_text'] = '\\n'.join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))
    else:
        data['body_text'] = 'N/A'

    return data'''

# --- Structural Map Generation ---
def create_structural_map(soup: BeautifulSoup, depth: int = 0, max_depth: int = None) -> List[Dict]:
    """Recursively generates a simplified, nested structural map of the HTML.

    *max_depth* defaults to MAX_DEPTH; callers pass a smaller value to shrink an
    over-large map so the prompt fits the input-token budget.
    """
    if max_depth is None:
        max_depth = MAX_DEPTH
    if depth >= max_depth:
        return []

    structure = []
    
    for child in soup.children:
        if child.name and child.name.lower() in TARGET_TAGS:
            attributes = {}
            if child.get('class'):
                attributes['class'] = " ".join(child.get('class')[:2]) 
            if child.get('id'):
                attributes['id'] = child.get('id')

            node = {
                'tag': child.name.lower(),
                'attributes': attributes,
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
def _fit_map_to_budget(agent, html_content, structural_map, page_url, requirements, budget):
    """Return (structural_map, structural_map_json, depth_used, tokens).

    Serializes the map with ensure_ascii=False so non-ASCII scripts (e.g. Arabic)
    stay single characters instead of 6-char \\uXXXX escapes that would ~6x the
    token count. If the full-depth prompt still exceeds *budget* tokens, rebuilds
    the map at progressively shallower depths (MAX_DEPTH-1 down to MIN_MAP_DEPTH)
    until it fits, so large pages don't blow the model's context window or the
    per-minute input-token quota.
    """
    smj = json.dumps(structural_map, ensure_ascii=False)
    toks = count_tokens(agent.model, agent._build_prompt(smj, requirements, page_url))
    print(f"  📏 Input prompt: ~{toks} tokens at full depth {MAX_DEPTH} (budget {budget}).", flush=True)
    if toks <= budget:
        return structural_map, smj, MAX_DEPTH, toks

    print(f"  ⚠  Over budget by ~{toks - budget} tokens — shrinking map depth to fit...", flush=True)
    soup = BeautifulSoup(html_content, 'lxml')
    body = soup.body if soup.body else soup
    last = None
    for d in range(MAX_DEPTH - 1, MIN_MAP_DEPTH - 1, -1):
        m = create_structural_map(body, max_depth=d)
        mj = json.dumps(m, ensure_ascii=False)
        t = count_tokens(agent.model, agent._build_prompt(mj, requirements, page_url))
        fits = t <= budget
        print(f"     depth {d}: ~{t} tokens  {'✓ fits' if fits else '✗ still over'}", flush=True)
        if fits:
            return m, mj, d, t
        last = (m, mj, d, t)

    # Nothing fit even at the floor depth — send the smallest map anyway; the
    # model may still accept it, otherwise it fails loudly instead of truncating.
    m, mj, d, t = last
    print(f"  ⚠  Still ~{t} tokens at floor depth {d} (budget {budget}); sending anyway.", flush=True)
    return m, mj, d, t


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
async def main_cli(url: str, api_key: str, requirements: str, model: str = 'gemma-3-27b-it', max_retries: int = 0,
                   max_input_tokens: int = DEFAULT_INPUT_TOKEN_BUDGET):
    """Non-interactive entry point. Returns paths via JSON line on stdout.

    Prints a JSON object on success:
      {"status": "ok", "code_file": "...", "output_file": "...", "data": {...}}
    Or on failure:
      {"status": "error", "error": "..."}
    """
    agent = GemmaAgent(api_key, model)

    reset_token_usage()
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
    args = parser.parse_args()

    if args.url and args.api_key and args.requirements:
        # Non-interactive CLI mode
        asyncio.run(main_cli(args.url, args.api_key, args.requirements, args.model, args.max_retries, args.max_input_tokens))
    else:
        # Interactive mode (original behavior)
        asyncio.run(main())
