# Scraping Run Results

Auto-generated stats, one section per orchestrator run.


## Run 2026-06-27 21:22:50

- **Run directory:** `orch_runs/run_20260627_211446`
- **Model:** gemma-4-31b-it
- **Pagination type:** single page
- **Input URLs:**
  - https://stories.undp.org/categories/africa
- **Requirements:** 'title, date, author, article body text'
- **Pages requested:** 1
- **Pages processed:** 1
- **Articles extracted:** 12
- **Articles failed:** 0
- **Clusters (unique structures):** 3
- **LLM calls:** 4
- **Total time:** 8m 17s
- **Errors:** none


## Run 2026-07-04 03:03:37

- **Run directory:** `orch_runs/run_20260704_025628`
- **Model:** gemma-4-31b-it
- **Pagination type:** numbered pagination
- **Input URLs:**
  - https://www.youm7.com/Section/%D8%A3%D8%AE%D8%A8%D8%A7%D8%B1-%D8%B9%D8%A7%D8%AC%D9%84%D8%A9/65/1
  - https://www.youm7.com/Section/%D8%A3%D8%AE%D8%A8%D8%A7%D8%B1-%D8%B9%D8%A7%D8%AC%D9%84%D8%A9/65/2
- **Requirements:** 'title, date, author, article body text'
- **Pages requested:** 3
- **Pages processed:** 3
- **Articles extracted:** 119
- **Articles failed:** 0
- **Clusters (unique structures):** 1
- **LLM calls:** 2
  - Article agent: 1
  - Links agent: 1
- **LLM calls per article:** 0.02
- **Code reuse rate:** 118/119 articles reused cluster code (99%)
- **Total time:** 7m 24s
- **Fetch method:** 119 via requests (100%), 0 via browser (0%)
- **Missing/N/A values (of 119 extracted):**
  - `title`: 0/119 N/A (0%)
  - `date`: 0/119 N/A (0%)
  - `author`: 10/119 N/A (8%)
  - `article body text`: 0/119 N/A (0%)
- **Errors:** none


## Run 2026-07-07 18:58:52

- **Run directory:** `orch_runs/run_20260707_184942`
- **Model:** gemma-4-31b-it
- **Pagination type:** infinite scroll
- **Input URLs:**
  - https://arabic.cnn.com/tag/gaza_strip
- **Requirements:** 'title, date, author, article body text'
- **Pages requested:** 1
- **Pages processed:** 1
- **Articles extracted:** 260
- **Articles failed:** 0
- **Clusters (unique structures):** 2
- **LLM calls:** 3
  - Article agent: 2
  - Links agent: 1
- **LLM calls per article:** 0.01
- **Code reuse rate:** 258/260 articles reused cluster code (99%)
- **Tokens:** 79,987 total (75,122 prompt + 1,032 output)
  - Article agent: 48,633 tokens
  - Links agent: 31,354 tokens
- **Estimated cost:** $0.0059 (at $0.075/$0.3 per 1M input/output tokens)
- **Estimated cost per 1,000 articles:** $0.0229
- **Total time:** 9m 29s
- **Fetch method:** 260 via requests (100%), 0 via browser (0%)
- **Missing/N/A values (of 260 extracted):**
  - `title`: 0/260 N/A (0%)
  - `date`: 0/260 N/A (0%)
  - `author`: 58/260 N/A (22%)
  - `article body text`: 0/260 N/A (0%)
- **Errors:** none


## Run 2026-07-07 19:20:09

- **Run directory:** `orch_runs/run_20260707_190029`
- **Model:** gemma-4-31b-it
- **Pagination type:** infinite scroll
- **Input URLs:**
  - https://www.aljadeedmagazine.com/%D9%85%D9%82%D8%A7%D9%84%D8%A7%D8%AA
- **Requirements:** 'title, date, author, article body text'
- **Pages requested:** 1
- **Pages processed:** 1
- **Articles extracted:** 1060
- **Articles failed:** 5
- **Clusters (unique structures):** 2
- **LLM calls:** 3
  - Article agent: 2
  - Links agent: 1
- **LLM calls per article:** 0.00
- **Code reuse rate:** 1058/1060 articles reused cluster code (100%)
- **Tokens:** 25,935 total (15,204 prompt + 852 output)
  - Article agent: 14,895 tokens
  - Links agent: 11,040 tokens
- **Estimated cost:** $0.0014 (at $0.075/$0.3 per 1M input/output tokens)
- **Estimated cost per 1,000 articles:** $0.0013
- **Total time:** 19m 56s
- **Fetch method:** 1059 via requests (100%), 1 via browser (0%)
- **Missing/N/A values (of 1060 extracted):**
  - `title`: 1/1060 N/A (0%)
  - `date`: 1060/1060 N/A (100%)
  - `author`: 1060/1060 N/A (100%)
  - `article body text`: 1060/1060 N/A (100%)
- **Errors (5):**
  - Empty response
  - Empty response
  - Empty response
  - Empty response
  - Empty response


## Run 2026-07-07 21:39:31

- **Run directory:** `orch_runs/run_20260707_203854`
- **Model:** gemma-4-31b-it
- **Pagination type:** infinite scroll
- **Input URLs:**
  - https://aawsat.com/%D8%A7%D9%84%D8%B1%D8%A3%D9%8A
- **Requirements:** 'title, date, author, article body text'
- **Pages requested:** 1
- **Pages processed:** 1
- **Articles extracted:** 0
- **Articles failed:** 1296
- **Clusters (unique structures):** 0
- **LLM calls:** 4
  - Article agent: 3
  - Links agent: 1
- **Tokens:** 19,080 total (15,824 prompt + 325 output)
  - Links agent: 19,080 tokens
- **Estimated cost:** $0.0013 (at $0.075/$0.3 per 1M input/output tokens)
- **Total time:** 61m 34s
- **Fetch method:** 1289 via requests (99%), 7 via browser (1%)
- **Errors (1296):**
  - Agent failed on new cluster representative: All 1 attempt(s) failed. Last error: LLM ERROR: Error generating code: 500 Internal error encountered.
  - Agent failed on new cluster representative: All 1 attempt(s) failed. Last error: LLM ERROR: Error generating code: 500 Internal error encountered.
  - Agent failed on new cluster representative: All 1 attempt(s) failed. Last error: LLM ERROR: Error generating code: 500 Internal error encountered.
  - Agent failed on new cluster representative: All 1 attempt(s) failed. Last error: LLM ERROR: Error generating code: 500 Internal error encountered.
  - Agent failed on new cluster representative: All 1 attempt(s) failed. Last error: LLM ERROR: Error generating code: 500 Internal error encountered.
  - Agent failed on new cluster representative: All 1 attempt(s) failed. Last error: LLM ERROR: Error generating code: 500 Internal error encountered.
  - Agent failed on new cluster representative: All 1 attempt(s) failed. Last error: LLM ERROR: Error generating code: 500 Internal error encountered.
  - Agent failed on new cluster representative: All 1 attempt(s) failed. Last error: LLM ERROR: Error generating code: 500 Internal error encountered.
  - Agent failed on new cluster representative: All 1 attempt(s) failed. Last error: LLM ERROR: Error generating code: 500 Internal error encountered.
  - Agent failed on new cluster representative: All 1 attempt(s) failed. Last error: LLM ERROR: Error generating code: 500 Internal error encountered.
  - ...and 1286 more


## Run 2026-07-07 22:11:24

- **Run directory:** `orch_runs/run_20260707_220730`
- **Model:** gemma-4-31b-it
- **Pagination type:** numbered pagination
- **Input URLs:**
  - https://pchrgaza.org/ar/category/genocide-on-gaza-ar/testimonies-from-the-war-ar/
  - https://pchrgaza.org/ar/category/genocide-on-gaza-ar/testimonies-from-the-war-ar/page/2/
- **Requirements:** 'title, date, author, article body text'
- **Pages requested:** 3
- **Pages processed:** 3
- **Articles extracted:** 41
- **Articles failed:** 0
- **Clusters (unique structures):** 1
- **LLM calls:** 2
  - Article agent: 1
  - Links agent: 1
- **LLM calls per article:** 0.05
- **Code reuse rate:** 40/41 articles reused cluster code (98%)
- **Tokens:** 75,564 total (73,377 prompt + 518 output)
  - Links agent: 39,057 tokens
  - Article agent: 36,507 tokens
- **Estimated cost:** $0.0057 (at $0.075/$0.3 per 1M input/output tokens)
- **Estimated cost per 1,000 articles:** $0.1380
- **Total time:** 4m 7s
- **Fetch method:** 41 via requests (100%), 0 via browser (0%)
- **Missing/N/A values (of 41 extracted):**
  - `title`: 0/41 N/A (0%)
  - `date`: 0/41 N/A (0%)
  - `author`: 41/41 N/A (100%)
  - `article body text`: 0/41 N/A (0%)
- **Errors:** none


## Run 2026-07-07 22:13:22

- **Run directory:** `orch_runs/run_20260707_221201`
- **Model:** gemma-4-31b-it
- **Pagination type:** infinite scroll
- **Input URLs:**
  - https://aawsat.com/%D8%A7%D9%84%D8%B1%D8%A3%D9%8A
- **Requirements:** 'title, date, author, article body text'
- **Pages requested:** 1
- **Pages processed:** 0
- **Articles extracted:** 0
- **Articles failed:** 0
- **Clusters (unique structures):** 0
- **LLM calls:** 1
  - Links agent: 1
- **Total time:** 1m 33s
- **Errors (1):**
  - Links extraction failed: All 1 attempt(s) failed. Last error: LLM ERROR: Error generating code: 500 Internal error encountered.


## Run 2026-07-07 22:48:50

- **Run directory:** `orch_runs/run_20260707_221647`
- **Model:** gemma-4-31b-it
- **Pagination type:** infinite scroll
- **Input URLs:**
  - https://aawsat.com/%D8%A7%D9%84%D8%B1%D8%A3%D9%8A
- **Requirements:** 'title, date, author, article body text'
- **Pages requested:** 1
- **Pages processed:** 1
- **Articles extracted:** 0
- **Articles failed:** 1355
- **Clusters (unique structures):** 0
- **LLM calls:** 6
  - Article agent: 5
  - Links agent: 1
- **Tokens:** 17,026 total (15,779 prompt + 318 output)
  - Links agent: 17,026 tokens
- **Estimated cost:** $0.0013 (at $0.075/$0.3 per 1M input/output tokens)
- **Total time:** 32m 12s
- **Fetch method:** 1349 via requests (100%), 6 via browser (0%)
- **Errors (1355):**
  - Agent failed on new cluster representative: All 1 attempt(s) failed. Last error: LLM ERROR: Error generating code: 500 Internal error encountered.
  - Agent failed on new cluster representative: All 1 attempt(s) failed. Last error: LLM ERROR: Error generating code: 500 Internal error encountered.
  - Agent failed on new cluster representative: All 1 attempt(s) failed. Last error: LLM ERROR: Error generating code: 500 Internal error encountered.
  - Agent failed on new cluster representative: All 1 attempt(s) failed. Last error: LLM ERROR: Error generating code: 500 Internal error encountered.
  - Agent failed on new cluster representative: All 1 attempt(s) failed. Last error: LLM ERROR: Error generating code: 500 Internal error encountered.
  - Agent failed on new cluster representative: All 1 attempt(s) failed. Last error: LLM ERROR: Error generating code: 500 Internal error encountered.
  - Agent failed on new cluster representative: All 1 attempt(s) failed. Last error: LLM ERROR: Error generating code: 500 Internal error encountered.
  - Agent failed on new cluster representative: All 1 attempt(s) failed. Last error: LLM ERROR: Error generating code: 500 Internal error encountered.
  - Agent failed on new cluster representative: All 1 attempt(s) failed. Last error: LLM ERROR: Error generating code: 500 Internal error encountered.
  - Agent failed on new cluster representative: All 1 attempt(s) failed. Last error: LLM ERROR: Error generating code: 500 Internal error encountered.
  - ...and 1345 more


## Run 2026-07-08 01:13:17

- **Run directory:** `orch_runs/run_20260708_004520`
- **Model:** gemma-4-31b-it
- **Pagination type:** infinite scroll
- **Input URLs:**
  - https://aawsat.com/%D8%A7%D9%84%D8%B1%D8%A3%D9%8A
- **Requirements:** 'title, date, author, article body text'
- **Pages requested:** 1
- **Pages processed:** 1
- **Articles extracted:** 693
- **Articles failed:** 2
- **Clusters (unique structures):** 4
- **LLM calls:** 6
  - Article agent: 5
  - Links agent: 1
- **LLM calls per article:** 0.01
- **Code reuse rate:** 688/693 articles reused cluster code (99%)
- **Tokens:** 90,820 total (76,884 prompt + 1,186 output)
  - Article agent: 71,116 tokens
  - Links agent: 19,704 tokens
- **Estimated cost:** $0.0061 (at $0.075/$0.3 per 1M input/output tokens)
- **Estimated cost per 1,000 articles:** $0.0088
- **Total time:** 28m 18s
- **Fetch method:** 690 via requests (99%), 4 via browser (1%)
- **Missing/N/A values (of 693 extracted):**
  - `title`: 3/693 N/A (0%)
  - `date`: 693/693 N/A (100%)
  - `author`: 693/693 N/A (100%)
  - `article body text`: 14/693 N/A (2%)
- **Errors (2):**
  - Empty response
  - Agent failed on new cluster representative: All 1 attempt(s) failed. Last error: LLM ERROR: Error generating code: 500 Internal error encountered.


## Run 2026-07-08 01:53:06

- **Run directory:** `orch_runs/run_20260708_011628`
- **Model:** gemma-4-31b-it
- **Pagination type:** load more button
- **Input URLs:**
  - https://www.arageek.com/tech
- **Requirements:** 'title, date, author, article body text'
- **Pages requested:** 1
- **Pages processed:** 1
- **Articles extracted:** 385
- **Articles failed:** 0
- **Clusters (unique structures):** 2
- **LLM calls:** 3
  - Article agent: 2
  - Links agent: 1
- **LLM calls per article:** 0.01
- **Code reuse rate:** 383/385 articles reused cluster code (99%)
- **Tokens:** 58,841 total (48,481 prompt + 1,065 output)
  - Links agent: 32,525 tokens
  - Article agent: 26,316 tokens
- **Estimated cost:** $0.0040 (at $0.075/$0.3 per 1M input/output tokens)
- **Estimated cost per 1,000 articles:** $0.0103
- **Total time:** 36m 52s
- **Fetch method:** 385 via requests (100%), 0 via browser (0%)
- **Missing/N/A values (of 385 extracted):**
  - `title`: 23/385 N/A (6%)
  - `date`: 385/385 N/A (100%)
  - `author`: 385/385 N/A (100%)
  - `article body text`: 0/385 N/A (0%)
- **Errors:** none


## Run 2026-07-08 02:03:23

- **Run directory:** `orch_runs/run_20260708_015602`
- **Model:** gemma-4-31b-it
- **Pagination type:** load more button
- **Input URLs:**
  - https://www.alarabiya.net/views
- **Requirements:** 'title, date, author, article body text'
- **Pages requested:** 1
- **Pages processed:** 1
- **Articles extracted:** 32
- **Articles failed:** 0
- **Clusters (unique structures):** 2
- **LLM calls:** 3
  - Article agent: 2
  - Links agent: 1
- **LLM calls per article:** 0.09
- **Code reuse rate:** 30/32 articles reused cluster code (94%)
- **Tokens:** 151,870 total (148,959 prompt + 796 output)
  - Article agent: 102,552 tokens
  - Links agent: 49,318 tokens
- **Estimated cost:** $0.0114 (at $0.075/$0.3 per 1M input/output tokens)
- **Estimated cost per 1,000 articles:** $0.3566
- **Total time:** 7m 31s
- **Fetch method:** 12 via requests (38%), 20 via browser (62%)
- **Missing/N/A values (of 32 extracted):**
  - `title`: 0/32 N/A (0%)
  - `date`: 0/32 N/A (0%)
  - `author`: 0/32 N/A (0%)
  - `article body text`: 0/32 N/A (0%)
- **Errors:** none


## Run 2026-07-08 12:57:41

- **Run directory:** `orch_runs/run_20260708_125109`
- **Model:** gemma-4-31b-it
- **Pagination type:** numbered pagination
- **Input URLs:**
  - https://mana.net/category/articles/
  - https://mana.net/category/articles/page/2/
- **Requirements:** 'title, date, author, article body text'
- **Pages requested:** 3
- **Pages processed:** 3
- **Articles extracted:** 6
- **Articles failed:** 0
- **Clusters (unique structures):** 2
- **LLM calls:** 3
  - Article agent: 2
  - Links agent: 1
- **LLM calls per article:** 0.50
- **Code reuse rate:** 4/6 articles reused cluster code (67%)
- **Tokens:** 71,355 total (62,962 prompt + 943 output)
  - Article agent: 44,798 tokens
  - Links agent: 26,557 tokens
- **Estimated cost:** $0.0050 (at $0.075/$0.3 per 1M input/output tokens)
- **Estimated cost per 1,000 articles:** $0.8342
- **Total time:** 6m 48s
- **Fetch method:** 6 via requests (100%), 0 via browser (0%)
- **Missing/N/A values (of 6 extracted):**
  - `title`: 0/6 N/A (0%)
  - `date`: 0/6 N/A (0%)
  - `author`: 6/6 N/A (100%)
  - `article body text`: 0/6 N/A (0%)
- **Errors:** none

