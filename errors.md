 python orch_interactive_pagination.py
- first page https://www.btselem.org/ota/100/all
- second page https://www.btselem.org/ota/100/all?page=1

Select a model:
   [Enter number] Choose from list above
   [Press Enter]  Use default (gemma-3-27b-it)
   → 2
✓ Selected: gemma-4-31b-it

📂 Run directory: orch_runs/run_20260627_200110

 How do you want to get article links?
   [1] Extract from a listing page (calls Links_Agent_gemma.py)
   [2] Load from an existing JSON file
   → 1

🌐 Enter the listing page URL: https://www.btselem.org/ota/100/all

📝 What data to extract from each article?
   (e.g., 'title, date, author, article body text')
   → 'title, date, author, article body text'

📄 Does this listing page have multiple pages (numbered pagination)?
   [1] Yes — I'll provide page 1 and page 2 URLs
   [2] No  — single page only
   → 1

   Page 1 URL [Enter to use the listing URL above]:
   (https://www.btselem.org/ota/100/all)
   → 

   Page 2 URL: https://www.btselem.org/ota/100/all?page=1

   ✓ Derived URL pattern (string diff): https://www.btselem.org/ota/100/all?page={page}
     (page 2 = 1)

   Is this pattern correct?
   [Enter] Yes, use it
   [type]  Enter corrected pattern (use {page} as placeholder)
   → 

   📄 How many pages to scrape (including page 1)? → 3
   ✓ Will scrape 3 pages (2 after page 1)
     First: https://www.btselem.org/ota/100/all?page=1
     Last:  https://www.btselem.org/ota/100/all?page=2

⏳ Extracting article links from page 1: https://www.btselem.org/ota/100/all...
  🔧 Calling: python Links_Agent_gemma_cloudflare.py --url https://www.btselem.org/ota/100/all...
✓ Extracted 12 article links from page 1

⏳ Fetching structural maps for page 1 (12 articles)...
  [1/12] [page 1] Fetching: ...
    ✓ OK (106424 bytes)
  [2/12] [page 1] Fetching: ...
    ✓ OK (139846 bytes)
  [3/12] [page 1] Fetching: ...
    ✓ OK (107077 bytes)
  [4/12] [page 1] Fetching: ...
    ✓ OK (226189 bytes)
  [5/12] [page 1] Fetching: ...
    ✓ OK (171489 bytes)
  [6/12] [page 1] Fetching: ...
    ✓ OK (94546 bytes)
  [7/12] [page 1] Fetching: ...
    ✓ OK (95420 bytes)
  [8/12] [page 1] Fetching: ...
    ✓ OK (89236 bytes)
  [9/12] [page 1] Fetching: ...
    ✓ OK (102604 bytes)
  [10/12] [page 1] Fetching: ...
    ✓ OK (141483 bytes)
  [11/12] [page 1] Fetching: ...
    ✓ OK (121115 bytes)
  [12/12] [page 1] Fetching: ...
    ✓ OK (105135 bytes)

⏳ Clustering page 1 articles and extracting data...

  🆕 New cluster f08528a15536 (5 article(s))
     Representative: 
  🔧 Calling: python Agent_for_single_page_gemma.py --url https://www.btselem.org/video/20260604_extreme_rodent_infestation_in_gaza_puttin...

──────────────────────────────────────── SUBPROCESS STDOUT ────────────────────────────────────────
⏳ Fetching page structure from https://www.btselem.org/video/20260604_extreme_rodent_infestation_in_gaza_putting_millions_at_risk...
❌ LLM call failed: 500 Internal error encountered.
  ❌ Attempt 1 failed: LLM ERROR: Error generating code: 500 Internal error encountered.
ORCH_RESULT:{"status": "error", "error": "All 1 attempt(s) failed. Last error: LLM ERROR: Error generating code: 500 Internal error encountered."}

──────────────────────────────────────── SUBPROCESS STDERR ────────────────────────────────────────
C:\Users\amlesmail\OneDrive - Microsoft\Documents\GitHub\AI-Agent-for-Web-Scraping\Agent_for_single_page_gemma.py:9: FutureWarning: 

All support for the `google.generativeai` package has ended. It will no longer be receiving 
updates or bug fixes. Please switch to the `google.genai` package as soon as possible.
See README for more details:

https://github.com/google-gemini/deprecated-generative-ai-python/blob/main/README.md

  import google.generativeai as genai

──────────────────────────────────────────────────────────────────────────────────────────────────
     ❌ Agent failed:
All 1 attempt(s) failed. Last error: LLM ERROR: Error generating code: 500 Internal error encountered.

  🆕 New cluster 3d794276b299 (1 article(s))
     Representative: 
  🔧 Calling: python Agent_for_single_page_gemma.py --url https://www.btselem.org/gaza_strip/20260223_39_including_children_died_in_gaza_d...
     ✓ Agent succeeded! Code: Agent_for_single_page_gemma_code/gen_code_44580.py

  🆕 New cluster 2e228a45707f (1 article(s))
     Representative: 
  🔧 Calling: python Agent_for_single_page_gemma.py --url https://projects.btselem.org/nibal/#en...

-------------------------------------------------------------------------------------------------------

Not an error but cloudflare blocking the extraction:
python dig.py https://stories.undp.org/categories/africa
C:\Users\amlesmail\OneDrive - Microsoft\Documents\GitHub\AI-Agent-for-Web-Scraping\Links_Agent_gemma_cloudflare.py:23: FutureWarning: 

All support for the `google.generativeai` package has ended. It will no longer be receiving 
updates or bug fixes. Please switch to the `google.genai` package as soon as possible.
See README for more details:

https://github.com/google-gemini/deprecated-generative-ai-python/blob/main/README.md

  import google.generativeai as genai
======================================================================
🔬 DIAGNOSTIC FETCH — https://stories.undp.org/categories/africa
======================================================================
  ⏳ Waiting for challenge [headless]... (5s)
  ⏳ Waiting for challenge [headless]... (10s)
  ⏳ Waiting for challenge [headless]... (15s)
  ⏳ Waiting for challenge [headless]... (20s)
  ⏳ Waiting for challenge [headless]... (25s)
  ⏳ Waiting for challenge [headless]... (30s)
  ⏳ Waiting for challenge [headless]... (35s)
  ⏳ Waiting for challenge [headless]... (40s)
  ⏳ Waiting for challenge [headless]... (45s)
  ⏳ Waiting for challenge [headless]... (50s)
  ⏳ Waiting for challenge [headless]... (55s)
  ⏳ Waiting for challenge [headless]... (60s)
  ⚠  Headless browser blocked by Cloudflare — trying headed browser...
  ⏳ Waiting for challenge [headed]... (5s)
  ⏳ Waiting for challenge [headed]... (10s)
  ⏳ Waiting for challenge [headed]... (15s)
  ⏳ Waiting for challenge [headed]... (20s)
  ⏳ Waiting for challenge [headed]... (25s)
  ⏳ Waiting for challenge [headed]... (30s)
  ⏳ Waiting for challenge [headed]... (35s)
  ⏳ Waiting for challenge [headed]... (40s)
  ⏳ Waiting for challenge [headed]... (45s)
  ⏳ Waiting for challenge [headed]... (50s)
  ⏳ Waiting for challenge [headed]... (55s)
  ⏳ Waiting for challenge [headed]... (60s)
  ⏳ Waiting for challenge [headed]... (65s)
  ⏳ Waiting for challenge [headed]... (70s)
  ⏳ Waiting for challenge [headed]... (75s)
  ⏳ Waiting for challenge [headed]... (80s)
  ⏳ Waiting for challenge [headed]... (85s)
  ⏳ Waiting for challenge [headed]... (90s)
  ⚠  Browser attempts unusable — trying plain HTTP (requests)...
  [requests] Attempt 1/3...
  [requests] Still got a challenge page

======================================================================
RESULT
======================================================================
✓ HTML received: 126417 bytes
  Structural map nodes (top level): 6

── First 800 chars of HTML ──
<!DOCTYPE html><html lang="en"><!-- /An Exposure site - https://exposure.co --><head>


      <!-- Google tag (gtag.js) -->
      <script type="text/javascript" async="" src="https://www.googletagmanager.com/gtag/js?id=G-5PMYWL03SK&amp;cx=c&amp;_slc=1"></script><script type="text/javascript" async="" src="https://www.googletagmanager.com/gtag/js?id=G-GM95J47GSV&amp;cx=c&amp;gtm=4e66o1"></script><script type="text/javascript" async="" src="https://www.googletagmanager.com/gtag/js?id=G-VLKXDT4YD0&amp;cx=c&amp;gtm=4e66o1"></script><script type="text/javascript" async="" src="https://www.googletagmanager.com/gtag/js?id=G-3W7LPK0WP1&amp;cx=c&amp;gtm=4e66o1"></script><script type="text/javascript" async="" src="https://www.googletagmanager.com/gtag/js?id=G-9WKDWWQSV2&amp;cx=c&amp;gtm=4e66o1"></s

⚠  WARNING: HTML still matches a Cloudflare/bot challenge marker.