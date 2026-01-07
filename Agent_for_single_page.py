import asyncio
import json
import re
import random
from typing import Dict, List, Optional, Tuple
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
import google.generativeai as genai

# --- Configuration ---
TARGET_TAGS = ['div', 'section', 'article', 'main', 'header', 'footer', 'ul', 'ol', 'li', 'a', 'h1', 'h2', 'h3', 'p', 'span']
MAX_DEPTH = 5
MAX_RETRIES = 1

# Allowed imports for generated code
ALLOWED_IMPORTS = {
    'BeautifulSoup': BeautifulSoup,
    're': re,
    'json': json,
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
                attributes['class'] = " ".join(child.get('class')[:2]) 
            if child.get('id'):
                attributes['id'] = child.get('id')

            node = {
                'tag': child.name.lower(),
                'attributes': attributes,
                'children': create_structural_map(child, depth + 1)
            }
            
            if not node['children'] and child.text and len(child.text.strip()) > 5:
                text_content = child.text.strip()
                node['text_snippet'] = text_content[:50].replace('\n', ' ') + ('...' if len(text_content) > 50 else '')

            structure.append(node)
            
    return structure


async def fetch_page_structure(url: str) -> Tuple[Optional[str], Optional[List[Dict]]]:
    """Fetch HTML and generate structural map."""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        
        try:
            await page.goto(url, wait_until="networkidle", timeout=30000)
            html_content = await page.content()
        except Exception as e:
            await browser.close()
            return None, None

        await browser.close()
        
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


class GeminiAgent:
    def __init__(self, api_key: str, model_name: str = 'gemini-1.5-flash-latest'):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.conversation_history = []
        self.model_name = model_name
    
    def generate_extraction_code(self, structural_map: str, user_requirements: str, error_context: Optional[str] = None) -> str:
        """Generate Python extraction code using Gemini."""
        
        if not error_context:
            # First attempt
            prompt = f"""You are a web scraping expert. Given an HTML structural map and user requirements, generate Python code to extract the requested data.

HTML STRUCTURAL MAP:
{structural_map}

USER REQUIREMENTS:
{user_requirements}

Generate a Python function called `extract_data(html_content)` that:
1. Takes raw HTML content as input
2. Uses BeautifulSoup to parse and extract the requested data
3. Returns a dictionary with the extracted data
4. Handles errors gracefully with try-except blocks
5. Return 'N/A' ONLY if:
    - No matching tag exists in the structural map
    - AND all reasonable fallbacks fail


IMPORTANT RULES:
- Only use these imports: BeautifulSoup (from bs4), re, json
- Do NOT import any other libraries (no requests, no os, no subprocess, etc.)
- The function signature MUST be: def extract_data(html_content):
- Use CSS selectors and BeautifulSoup methods (select, select_one, find, find_all)
- Clean extracted text using strip() and regex where needed
- Return a dictionary with clear key names

CRITICAL RULES (MUST FOLLOW):
- You may ONLY use tag names, class names, and ids that appear EXACTLY in the HTML STRUCTURAL MAP
- Do NOT invent class names or ids (e.g. page-header, main-content, article-body)
- If no class or id exists, use tag-only selectors (e.g. h1, article p)
- If multiple candidates exist, extract ALL and concatenate text
- You are NOT allowed to claim data is missing unless the tag does not appear in the map

FORBIDDEN:
- Guessing common website structures
- Using selectors based on assumptions (news sites, blogs, CMS patterns)
- Comments like "Based on the provided structure, X is not present"

Before writing the function:
- Identify which node(s) in the structural map correspond to each requested field
- Choose selectors directly derived from those nodes
- Then write the extraction code

If an exact selector fails:
- Fallback to broader selectors
- Never return N/A without attempting at least one fallback

Output ONLY the Python function definition.
Do NOT include import statements.
Start directly with: def extract_data(html_content):
"""

        else:
            # Retry with error context
            prompt = f"""The previous code failed with this error:

ERROR:
{error_context}

ORIGINAL REQUIREMENTS:
{user_requirements}

HTML STRUCTURAL MAP:
{structural_map}

Please generate CORRECTED Python code that fixes the error. Remember:
- Only use: BeautifulSoup, re, json
- Function signature: def extract_data(html_content):
- Return a dictionary
- Handle errors with try-except

Output ONLY the corrected Python code, no explanations."""

        try:
            response = self.model.generate_content(prompt)
            code = response.text
            
            # Clean the code (remove markdown code blocks if present)
            code = re.sub(r'^```python\s*\n', '', code, flags=re.MULTILINE)
            code = re.sub(r'\n```\s*$', '', code, flags=re.MULTILINE)
            code = code.strip()
            
            # Save generated code to file
            random_num = random.randint(10000, 99999)
            code_filename = f"generated_code_{random_num}.py"
            with open(code_filename, 'w', encoding='utf-8') as f:
                f.write(code)
            print(f"💾 Code saved to: {code_filename}")
            
            # Store in conversation history
            self.conversation_history.append({
                'prompt': prompt,
                'response': code,
                'error': error_context,
                'saved_file': code_filename
            })
            
            return code
            
        except Exception as e:
            error_code = f"# Error generating code: {str(e)}"
            # Save error to file too
            random_num = random.randint(10000, 99999)
            code_filename = f"generated_code_{random_num}_ERROR.py"
            with open(code_filename, 'w', encoding='utf-8') as f:
                f.write(error_code)
            print(f"💾 Error saved to: {code_filename}")
            return error_code


# --- Safe Code Execution ---
def validate_code_safety(code: str) -> Tuple[bool, str]:
    """Basic safety validation for generated code."""
    dangerous_patterns = [
        r'import\s+os',
        r'import\s+sys',
        r'import\s+subprocess',
        r'import\s+requests',
        r'import\s+urllib',
        r'import\s+socket',
        r'import\s+pickle',
        r'__import__',
        r'eval\s*\(',
        r'exec\s*\(',
        r'compile\s*\(',
        r'open\s*\(',
        r'file\s*\(',
    ]
    
    for pattern in dangerous_patterns:
        if re.search(pattern, code, re.IGNORECASE):
            return False, f"Dangerous pattern detected: {pattern}"
    
    return True, "Code appears safe"


def execute_extraction_code(code: str, html_content: str) -> Tuple[bool, any]:
    """Execute the generated extraction code in a restricted environment."""
    
    # Validate safety first
    is_safe, safety_msg = validate_code_safety(code)
    if not is_safe:
        return False, f"SAFETY ERROR: {safety_msg}"
    
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
        },
        # Pre-import allowed modules
        'BeautifulSoup': BeautifulSoup,
        're': re,
        'json': json,
        'bs4': type('Module', (), {'BeautifulSoup': BeautifulSoup})(),  # Mock bs4 module
    }
    
    # Add a custom __import__ that only allows safe imports
    def safe_import(name, *args, **kwargs):
        allowed = {
            'bs4': type('Module', (), {'BeautifulSoup': BeautifulSoup})(),
            're': re,
            'json': json,
        }
        if name in allowed:
            return allowed[name]
        raise ImportError(f"Import of '{name}' is not allowed")
    
    restricted_globals['__builtins__']['__import__'] = safe_import
    
    try:
        # Execute the code to define the function
        exec(code, restricted_globals)
        
        # Check if extract_data function exists
        if 'extract_data' not in restricted_globals:
            return False, "ERROR: Generated code does not contain 'extract_data' function"
        
        # Call the extraction function
        result = restricted_globals['extract_data'](html_content)
        
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
    print("🤖 AI WEB SCRAPING AGENT")
    print("="*60)
    
    # Get API key
    api_key = input("\n🔑 Enter your Gemini API key: ").strip()
    if not api_key:
        print("❌ API key is required!")
        return
    
    # List available models
    print("\n⏳ Checking available models...")
    available_models = await list_available_models(api_key)
    
    if available_models:
        print(f"\n✓ Found {len(available_models)} available models:")
        for i, model in enumerate(available_models, 1):
            print(f"   {i}. {model}")
        
        print("\n📝 Select a model:")
        print("   [Enter number] Choose from list above")
        print("   [Press Enter] Use default (gemini-1.5-flash-latest)")
        
        choice = input("   → ").strip()
        
        if choice.isdigit() and 1 <= int(choice) <= len(available_models):
            selected_model = available_models[int(choice) - 1]
            # Remove 'models/' prefix if present for GenerativeModel
            if selected_model.startswith('models/'):
                selected_model = selected_model[7:]
            print(f"✓ Selected: {selected_model}")
        else:
            selected_model = 'gemini-1.5-flash-latest'
            print(f"✓ Using default: {selected_model}")
    else:
        selected_model = 'gemini-1.5-flash-latest'
        print(f"⚠️  Could not list models, using default: {selected_model}")
    
    # Get URL
    url = input("\n🌐 Enter the URL to scrape: ").strip()
    if not url:
        print("❌ URL is required!")
        return
    
    # Get extraction requirements
    print("\n📝 What data do you want to extract?")
    print("   (e.g., 'title, date, article text, author name')")
    requirements = input("   → ").strip()
    if not requirements:
        print("❌ Requirements are required!")
        return
    
    # Initialize agent
    agent = GeminiAgent(api_key, selected_model)
    
    # Step 1: Fetch page structure
    print(f"\n⏳ Fetching page structure from {url}...")
    html_content, structural_map = await fetch_page_structure(url)
    
    if not html_content or not structural_map:
        print("❌ Failed to fetch page structure!")
        return

    # 👇 ADD HERE
    rand_id = random.randint(10000, 99999)

    html_filename = f"html_content_{rand_id}.html"
    with open(html_filename, "w", encoding="utf-8") as f:
        f.write(html_content)

    structural_map_filename = f"structural_map_{rand_id}.json"
    with open(structural_map_filename, "w", encoding="utf-8") as f:
        json.dump(structural_map, f, indent=2, ensure_ascii=False)

    print(f"💾 Saved HTML to: {html_filename}")
    print(f"💾 Saved structural map to: {structural_map_filename}")

    print("✓ Page structure fetched successfully!")
    
    
    if not html_content or not structural_map:
        print("Failed to fetch page structure!")
        return
    
    print("✓ Page structure fetched successfully!")
    
    structural_map_json = json.dumps(structural_map, indent=2)
    
    # Step 2: Generate extraction code
    print("\n⏳ Generating extraction code with Gemini...")
    
    retry_count = 0
    error_context = None
    
    while retry_count <= MAX_RETRIES:
        # Generate code
        extraction_code = agent.generate_extraction_code(
            structural_map_json, 
            requirements, 
            error_context
        )
        
        print("✓ Code generated!")
        
        # Show generated code to user
        print("\n" + "-"*60)
        print("GENERATED CODE:")
        print("-"*60)
        print(extraction_code)
        print("-"*60)
        
        # Step 3: Execute code
        print("\n⏳ Executing extraction code...")
        success, result = execute_extraction_code(extraction_code, html_content)
        
        if success:
            # Analyze output
            stats = analyze_output(result)
            
            # Display sample
            display_sample_output(result, stats)
            
            # Ask user for decision
            print("\n❓ Are you satisfied with the results?")
            print("   [s] Save to JSON")
            print("   [r] Retry (generate new code)")
            print("   [q] Quit")
            
            decision = input("   → ").strip().lower()
            
            if decision == 's':
                # Save to JSON
                output_filename = f"extracted_data_{url.split('//')[1].split('/')[0].replace('.', '_')}.json"
                with open(output_filename, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                print(f"\n✅ Data saved to: {output_filename}")
                break
                
            elif decision == 'r':
                if retry_count >= MAX_RETRIES:
                    print(f"\n❌ Maximum retries ({MAX_RETRIES}) reached. Cannot retry further.")
                    break
                print("\n🔄 Retrying with new code generation...")
                error_context = "User requested retry. The previous extraction did not meet requirements. Please improve the extraction logic."
                retry_count += 1
                
            else:
                print("\n👋 Exiting without saving.")
                break
                
        else:
            # Execution failed
            print(f"\n❌ EXECUTION FAILED!")
            print(f"Error: {result}")
            
            if retry_count >= MAX_RETRIES:
                print(f"\nMaximum retries ({MAX_RETRIES}) reached. Cannot continue.")
                print("💡 Suggestions:")
                print("   - Check if the website structure is too complex")
                print("   - Try a different URL or simpler extraction requirements")
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


if __name__ == "__main__":
    asyncio.run(main())
