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

---------------new version------------


## Run 2026-07-08 13:45:43

- **Run directory:** `orch_runs/run_20260708_133946`
- **Model:** gemma-4-31b-it
- **Pagination type:** numbered pagination
- **Input URLs:**
  - https://mana.net/category/articles/
  - https://mana.net/category/articles/page/2/
- **Requirements:** 'title, date, author, article body text'
- **Pages requested:** 3
- **Pages processed:** 3
- **Articles extracted:** 1
- **Articles failed:** 0
- **Clusters (unique structures):** 1
- **LLM calls:** 2
  - Article agent: 1
  - Links agent: 1
- **LLM calls per article:** 2.00
- **Code reuse rate:** 0/1 articles reused cluster code (0%)
- **Tokens:** 75,667 total (66,781 prompt + 1,229 output)
  - Links agent: 54,291 tokens
  - Article agent: 21,376 tokens
- **Estimated cost:** $0.0054 (at $0.075/$0.3 per 1M input/output tokens)
- **Estimated cost per 1,000 articles:** $5.3773
- **Total time:** 6m 10s
- **Fetch method:** 1 via requests (100%), 0 via browser (0%)
- **Missing/N/A values (of 1 extracted):**
  - `title`: 1/1 N/A (100%)
  - `date`: 0/1 N/A (0%)
  - `author`: 1/1 N/A (100%)
  - `article body text`: 0/1 N/A (0%)
- **Errors:** none


## Run 2026-07-11 01:28:32

- **Run directory:** `orch_runs/run_20260711_012244`
- **Model:** gemma-4-31b-it
- **Pagination type:** numbered pagination
- **Input URLs:**
  - https://mana.net/category/articles/
  - https://mana.net/category/articles/page/2/
- **Requirements:** 'title, date, author, article body text'
- **Pages requested:** 3
- **Pages processed:** 3
- **Articles extracted:** 30
- **Articles failed:** 0
- **Clusters (unique structures):** 2
- **LLM calls:** 3
  - Article agent: 2
  - Links agent: 1
- **LLM calls per article:** 0.10
- **Code reuse rate:** 28/30 articles reused cluster code (93%)
- **Tokens:** 73,977 total (66,779 prompt + 1,010 output)
  - Article agent: 47,292 tokens
  - Links agent: 26,685 tokens
- **Estimated cost:** $0.0053 (at $0.075/$0.3 per 1M input/output tokens)
- **Estimated cost per 1,000 articles:** $0.1770
- **Total time:** 5m 59s
- **Fetch method:** 30 via requests (100%), 0 via browser (0%)
- **Missing/N/A values (of 30 extracted):**
  - `title`: 0/30 N/A (0%)
  - `date`: 0/30 N/A (0%)
  - `author`: 0/30 N/A (0%)
  - `article body text`: 0/30 N/A (0%)
- **Errors:** none


## Run 2026-07-15 23:15:31

- **Run directory:** `orch_runs/run_20260715_231408`
- **Model:** gemma-4-31b-it
- **Pagination type:** infinite scroll
- **Input URLs:**
  - https://www.independentarabia.com/%D8%AB%D9%82%D8%A7%D9%81%D8%A9/%D8%B3%D9%8A%D9%86%D9%85%D8%A7
- **Requirements:** 'title, date, author, article body text'
- **Pages requested:** 1
- **Pages processed:** 0
- **Articles extracted:** 0
- **Articles failed:** 0
- **Clusters (unique structures):** 0
- **LLM calls:** 1
  - Links agent: 1
- **Total time:** 1m 44s
- **Errors (1):**
  - Links extraction failed: All 3 attempt(s) failed. Last error: LLM ERROR: Error generating code: 429 You exceeded your current quota, please check your plan and billing details. For more information on


## Run 2026-07-15 23:21:49

- **Run directory:** `orch_runs/run_20260715_231912`
- **Model:** gemma-4-31b-it
- **Pagination type:** infinite scroll
- **Input URLs:**
  - https://www.independentarabia.com/%D8%AB%D9%82%D8%A7%D9%81%D8%A9/%D8%B3%D9%8A%D9%86%D9%85%D8%A7
- **Requirements:** 'title, date, author, article body text'
- **Pages requested:** 1
- **Pages processed:** 0
- **Articles extracted:** 0
- **Articles failed:** 0
- **Clusters (unique structures):** 0
- **LLM calls:** 1
  - Links agent: 1
- **Total time:** 2m 52s
- **Errors (1):**
  - Links extraction failed: All 3 attempt(s) failed. Last error: LLM ERROR: Error generating code: 429 You exceeded your current quota, please check your plan and billing details. For more information on


## Run 2026-07-16 02:16:03

- **Run directory:** `orch_runs/run_20260716_020819`
- **Model:** gemma-4-31b-it
- **Pagination type:** infinite scroll
- **Input URLs:**
  - https://www.independentarabia.com/%D8%AB%D9%82%D8%A7%D9%81%D8%A9/%D8%B3%D9%8A%D9%86%D9%85%D8%A7
- **Requirements:** 'title, date, author, article body text'
- **Pages requested:** 1
- **Pages processed:** 1
- **Articles extracted:** 136
- **Articles failed:** 0
- **Clusters (unique structures):** 1
- **LLM calls:** 2
  - Article agent: 1
  - Links agent: 1
- **LLM calls per article:** 0.01
- **Code reuse rate:** 135/136 articles reused cluster code (99%)
- **Tokens:** 25,143 total (19,394 prompt + 755 output)
  - Links agent: 15,565 tokens
  - Article agent: 9,578 tokens
- **Estimated cost:** $0.0017 (at $0.075/$0.3 per 1M input/output tokens)
- **Estimated cost per 1,000 articles:** $0.0124
- **Total time:** 7m 51s
- **Fetch method:** 136 via requests (100%), 0 via browser (0%)
- **Missing/N/A values (of 136 extracted):**
  - `title`: 0/136 N/A (0%)
  - `date`: 0/136 N/A (0%)
  - `author`: 0/136 N/A (0%)
  - `article body text`: 0/136 N/A (0%)
- **Errors:** none


## Run 2026-07-16 02:44:46

- **Run directory:** `orch_runs/run_20260716_021841`
- **Model:** gemma-4-31b-it
- **Pagination type:** numbered pagination
- **Input URLs:**
  - https://www.btselem.org/ota/100/all
  - https://www.btselem.org/ota/100/all?page=1
- **Requirements:** 'title, date, author, article body text'
- **Pages requested:** 4
- **Pages processed:** 4
- **Articles extracted:** 47
- **Articles failed:** 1
- **Clusters (unique structures):** 5
- **LLM calls:** 7
  - Article agent: 6
  - Links agent: 1
- **LLM calls per article:** 0.15
- **Code reuse rate:** 41/47 articles reused cluster code (87%)
- **Tokens:** 52,619 total (42,805 prompt + 1,700 output)
  - Article agent: 38,194 tokens
  - Links agent: 14,425 tokens
- **Estimated cost:** $0.0037 (at $0.075/$0.3 per 1M input/output tokens)
- **Estimated cost per 1,000 articles:** $0.0792
- **Structural-map depth:** 6 agent call(s), 0 shrunk below full depth 10 to fit the input-token budget
  - Links agent: depth 10 (~12,352 input tokens)
  - Article agent: depth 10 (~4,932 input tokens)
  - Article agent: depth 10 (~4,399 input tokens)
  - Article agent: depth 10 (~7,222 input tokens)
  - Article agent: depth 10 (~6,654 input tokens)
  - Article agent: depth 10 (~7,240 input tokens)
- **Total time:** 26m 11s
- **Fetch method:** 1 via requests (2%), 47 via browser (98%)
- **Missing/N/A values (of 47 extracted):**
  - `title`: 0/47 N/A (0%)
  - `date`: 2/47 N/A (4%)
  - `author`: 47/47 N/A (100%)
  - `article body text`: 30/47 N/A (64%)
  - `article_body_text`: 32/47 N/A (68%)
- **Errors (1):**
  - Agent failed on new cluster representative: All 1 attempt(s) failed. Last error: LLM ERROR: Error generating code: 429 You exceeded your current quota, please check your plan and billing details. For 


## Run 2026-07-18 00:39:39

- **Run directory:** `orch_runs/run_20260718_003144`
- **Model:** gemma-4-31b-it
- **Pagination type:** infinite scroll
- **Input URLs:**
  - https://eg.afedne.com/global-blogs/1/%D8%A3%D8%AE%D8%A8%D8%A7%D8%B1_%D9%85%D8%B5%D8%B1
- **Requirements:** 'title, date, author, article body text'
- **Pages requested:** 1
- **Pages processed:** 1
- **Articles extracted:** 107
- **Articles failed:** 0
- **Clusters (unique structures):** 2
- **LLM calls:** 3
  - Article agent: 2
  - Links agent: 1
- **LLM calls per article:** 0.03
- **Code reuse rate:** 105/107 articles reused cluster code (98%)
- **Tokens:** 51,819 total (46,600 prompt + 948 output)
  - Links agent: 38,679 tokens
  - Article agent: 13,140 tokens
- **Estimated cost:** $0.0038 (at $0.075/$0.3 per 1M input/output tokens)
- **Estimated cost per 1,000 articles:** $0.0353
- **Structural-map depth:** 3 agent call(s), 1 shrunk below full depth 10 to fit the input-token budget
  - Links agent: depth 3 (~37,726 input tokens)  ⚠ reduced
  - Article agent: depth 10 (~4,059 input tokens)
  - Article agent: depth 10 (~4,812 input tokens)
- **Total time:** 8m 9s
- **Fetch method:** 89 via requests (83%), 18 via browser (17%)
- **Missing/N/A values (of 107 extracted):**
  - `title`: 17/107 N/A (16%)
  - `date`: 107/107 N/A (100%)
  - `author`: 107/107 N/A (100%)
  - `article body text`: 17/107 N/A (16%)
- **Errors:** none


## Run 2026-07-18 21:18:39

- **Run directory:** `orch_runs/run_20260718_210902`
- **Model:** gemma-4-31b-it
- **Pagination type:** single page / infinite scroll (auto)
- **Input URLs:**
  - https://lakome2.com/category/art/
- **Requirements:** 'title, date, author, article body text'
- **Pages requested:** 1
- **Pages processed:** 0
- **Articles extracted:** 0
- **Articles failed:** 0
- **Clusters (unique structures):** 0
- **LLM calls:** 2
  - Links agent: 1
  - Pagination detection: 1
- **Tokens:** 1,183 total (930 prompt + 0 output)
  - Pagination detection: 1,183 tokens
- **Estimated cost:** $0.0001 (at $0.075/$0.3 per 1M input/output tokens)
- **Total time:** 9m 57s
- **Errors (1):**
  - Links extraction failed: Subprocess timed out (300s)


## Run 2026-07-18 22:38:15

- **Run directory:** `orch_runs/run_20260718_223156`
- **Model:** gemma-4-31b-it
- **Pagination type:** load more button
- **Input URLs:**
  - https://lakome2.com/category/art/
- **Requirements:** 'title, date, author, article body text'
- **Pages requested:** 1
- **Pages processed:** 0
- **Articles extracted:** 0
- **Articles failed:** 0
- **Clusters (unique structures):** 0
- **LLM calls:** 1
  - Links agent: 1
- **Total time:** 6m 29s
- **Errors (1):**
  - Links extraction failed: Subprocess timed out (300s)


## Run 2026-07-18 22:51:12

- **Run directory:** `orch_runs/run_20260718_224432`
- **Model:** gemma-4-31b-it
- **Pagination type:** numbered pagination
- **Input URLs:**
  - https://www.youm7.com/Section/%D8%A3%D8%AE%D8%A8%D8%A7%D8%B1-%D8%B9%D8%A7%D8%AC%D9%84%D8%A9/65/1
  - https://www.youm7.com/Section/%D8%A3%D8%AE%D8%A8%D8%A7%D8%B1-%D8%B9%D8%A7%D8%AC%D9%84%D8%A9/65/2
- **Requirements:** 'title, date, author, article body text'
- **Pages requested:** 4
- **Pages processed:** 0
- **Articles extracted:** 0
- **Articles failed:** 0
- **Clusters (unique structures):** 0
- **LLM calls:** 1
  - Links agent: 1
- **Total time:** 11m 10s
- **Errors (1):**
  - Links extraction failed: No ORCH_RESULT in output. stderr: C:\Users\amlesmail\OneDrive - Microsoft\Documents\GitHub\AI-Agent-for-Web-Scraping\Links_Agent_gemma_cloudflare.py:23: FutureWarning:   All s


## Run 2026-07-18 23:05:09

- **Run directory:** `orch_runs/run_20260718_225155`
- **Model:** gemma-4-31b-it
- **Pagination type:** numbered pagination
- **Input URLs:**
  - https://www.youm7.com/Section/%D8%A3%D8%AE%D8%A8%D8%A7%D8%B1-%D8%B9%D8%A7%D8%AC%D9%84%D8%A9/65/1
  - https://www.youm7.com/Section/%D8%A3%D8%AE%D8%A8%D8%A7%D8%B1-%D8%B9%D8%A7%D8%AC%D9%84%D8%A9/65/2
- **Requirements:** 'title, date, author, article body text'
- **Pages requested:** 8
- **Pages processed:** 8
- **Articles extracted:** 317
- **Articles failed:** 0
- **Clusters (unique structures):** 1
- **LLM calls:** 2
  - Article agent: 1
  - Links agent: 1
- **LLM calls per article:** 0.01
- **Code reuse rate:** 316/317 articles reused cluster code (100%)
- **Tokens:** 29,321 total (23,984 prompt + 890 output)
  - Article agent: 15,074 tokens
  - Links agent: 14,247 tokens
- **Estimated cost:** $0.0021 (at $0.075/$0.3 per 1M input/output tokens)
- **Estimated cost per 1,000 articles:** $0.0065
- **Structural-map depth:** 2 agent call(s), 2 shrunk below full depth 10 to fit the input-token budget
  - Links agent: depth 8 (~10,075 input tokens)  ⚠ reduced
  - Article agent: depth 9 (~13,907 input tokens)  ⚠ reduced
- **Total time:** 13m 47s
- **Fetch method:** 317 via requests (100%), 0 via browser (0%)
- **Missing/N/A values (of 317 extracted):**
  - `title`: 0/317 N/A (0%)
  - `date`: 0/317 N/A (0%)
  - `author`: 5/317 N/A (2%)
  - `article body text`: 316/317 N/A (100%)
- **Errors:** none


## Run 2026-07-20 13:29:30

- **Run directory:** `orch_runs/run_20260720_132407`
- **Model:** gemma-4-31b-it
- **Pagination type:** numbered pagination
- **Input URLs:**
  - https://pchrgaza.org/ar/category/genocide-on-gaza-ar/testimonies-from-the-war-ar/
  - https://pchrgaza.org/ar/category/genocide-on-gaza-ar/testimonies-from-the-war-ar/page/2/
- **Requirements:** 'title, date, author, article body text'
- **Pages requested:** 19
- **Pages processed:** 0
- **Articles extracted:** 0
- **Articles failed:** 0
- **Clusters (unique structures):** 0
- **LLM calls:** 1
  - Links agent: 1
- **Total time:** 5m 45s
- **Errors (1):**
  - Links extraction failed: All 3 attempt(s) failed. Last error: SANDBOX ERROR: Execution timed out after 8 seconds


## Run 2026-07-20 13:56:36

- **Run directory:** `orch_runs/run_20260720_134136`
- **Model:** gemma-4-31b-it
- **Pagination type:** numbered pagination
- **Input URLs:**
  - https://pchrgaza.org/ar/category/genocide-on-gaza-ar/testimonies-from-the-war-ar/
  - https://pchrgaza.org/ar/category/genocide-on-gaza-ar/testimonies-from-the-war-ar/page/2/
- **Requirements:** 'title, date, author, article body text'
- **Pages requested:** 19
- **Pages processed:** 19
- **Articles extracted:** 233
- **Articles failed:** 0
- **Clusters (unique structures):** 1
- **LLM calls:** 2
  - Article agent: 1
  - Links agent: 1
- **LLM calls per article:** 0.01
- **Code reuse rate:** 232/233 articles reused cluster code (100%)
- **Tokens:** 27,965 total (23,891 prompt + 714 output)
  - Links agent: 17,056 tokens
  - Article agent: 10,909 tokens
- **Estimated cost:** $0.0020 (at $0.075/$0.3 per 1M input/output tokens)
- **Estimated cost per 1,000 articles:** $0.0086
- **Structural-map depth:** 2 agent call(s), 1 shrunk below full depth 10 to fit the input-token budget
  - Links agent: depth 8 (~14,717 input tokens)  ⚠ reduced
  - Article agent: depth 10 (~9,172 input tokens)
- **Total time:** 15m 8s
- **Fetch method:** 233 via requests (100%), 0 via browser (0%)
- **Missing/N/A values (of 233 extracted):**
  - `title`: 0/233 N/A (0%)
  - `date`: 0/233 N/A (0%)
  - `author`: 233/233 N/A (100%)
  - `article body text`: 215/233 N/A (92%)
- **Errors:** none


## Run 2026-07-20 14:28:15

- **Run directory:** `orch_runs/run_20260720_141158`
- **Model:** gemma-4-31b-it
- **Pagination type:** numbered pagination
- **Input URLs:**
  - https://www.almasryalyoum.com/news/index?typeid=1&sectionid=10
  - https://www.almasryalyoum.com/news/index?typeid=1&sectionid=10&page=2
- **Requirements:** 'title, date, author, article body text'
- **Pages requested:** 20
- **Pages processed:** 20
- **Articles extracted:** 200
- **Articles failed:** 0
- **Clusters (unique structures):** 2
- **LLM calls:** 3
  - Article agent: 2
  - Links agent: 1
- **LLM calls per article:** 0.01
- **Code reuse rate:** 198/200 articles reused cluster code (99%)
- **Tokens:** 37,058 total (32,201 prompt + 1,106 output)
  - Article agent: 23,929 tokens
  - Links agent: 13,129 tokens
- **Estimated cost:** $0.0027 (at $0.075/$0.3 per 1M input/output tokens)
- **Estimated cost per 1,000 articles:** $0.0137
- **Structural-map depth:** 3 agent call(s), 0 shrunk below full depth 10 to fit the input-token budget
  - Links agent: depth 10 (~11,029 input tokens)
  - Article agent: depth 10 (~13,718 input tokens)
  - Article agent: depth 10 (~7,451 input tokens)
- **Total time:** 17m 14s
- **Fetch method:** 200 via requests (100%), 0 via browser (0%)
- **Missing/N/A values (of 200 extracted):**
  - `title`: 187/200 N/A (94%)
  - `date`: 11/200 N/A (6%)
  - `author`: 11/200 N/A (6%)
  - `article body text`: 12/200 N/A (6%)
  - `article_body_text`: 199/200 N/A (100%)
- **Errors:** none


## Run 2026-07-20 19:19:13

- **Run directory:** `orch_runs/run_20260720_191456`
- **Model:** gemma-4-31b-it
- **Pagination type:** load more button
- **Input URLs:**
  - https://alsifr.org/kam-kaif
- **Requirements:** 'title, date, author, article body text'
- **Pages requested:** 1
- **Pages processed:** 1
- **Articles extracted:** 12
- **Articles failed:** 0
- **Clusters (unique structures):** 1
- **LLM calls:** 2
  - Article agent: 1
  - Links agent: 1
- **LLM calls per article:** 0.17
- **Code reuse rate:** 11/12 articles reused cluster code (92%)
- **Tokens:** 20,029 total (16,059 prompt + 757 output)
  - Links agent: 11,684 tokens
  - Article agent: 8,345 tokens
- **Estimated cost:** $0.0014 (at $0.075/$0.3 per 1M input/output tokens)
- **Estimated cost per 1,000 articles:** $0.1193
- **Structural-map depth:** 2 agent call(s), 0 shrunk below full depth 10 to fit the input-token budget
  - Links agent: depth 10 (~8,848 input tokens)
  - Article agent: depth 10 (~7,209 input tokens)
- **Total time:** 4m 29s
- **Fetch method:** 12 via requests (100%), 0 via browser (0%)
- **Missing/N/A values (of 12 extracted):**
  - `title`: 0/12 N/A (0%)
  - `date`: 0/12 N/A (0%)
  - `author`: 0/12 N/A (0%)
  - `article body text`: 0/12 N/A (0%)
- **Errors:** none


## Run 2026-07-22 02:07:32

- **Run directory:** `orch_runs/run_20260722_015742`
- **Model:** gemma-4-31b-it
- **Pagination type:** numbered pagination
- **Input URLs:**
  - https://www.youm7.com/Section/%D8%A3%D8%AE%D8%A8%D8%A7%D8%B1-%D8%B9%D8%A7%D8%AC%D9%84%D8%A9/65/1
  - https://www.youm7.com/Section/%D8%A3%D8%AE%D8%A8%D8%A7%D8%B1-%D8%B9%D8%A7%D8%AC%D9%84%D8%A9/65/2
- **Requirements:** 'title, date, author, article body text'
- **Pages requested:** 3
- **Pages processed:** 3
- **Articles extracted:** 114
- **Articles failed:** 5
- **Clusters (unique structures):** 1
- **LLM calls:** 2
  - Article agent: 1
  - Links agent: 1
- **LLM calls per article:** 0.02
- **Code reuse rate:** 113/114 articles reused cluster code (99%)
- **Tokens:** 31,475 total (25,075 prompt + 792 output)
  - Article agent: 15,787 tokens
  - Links agent: 15,688 tokens
- **Estimated cost:** $0.0021 (at $0.075/$0.3 per 1M input/output tokens)
- **Estimated cost per 1,000 articles:** $0.0186
- **Structural-map depth:** 2 agent call(s), 2 shrunk below full depth 10 to fit the input-token budget
  - Links agent: depth 8 (~10,877 input tokens)  ⚠ reduced
  - Article agent: depth 9 (~14,196 input tokens)  ⚠ reduced
- **Total time:** 9m 59s
- **Fetch method:** 119 via requests (100%), 0 via browser (0%)
- **Missing/N/A values (of 114 extracted):**
  - `title`: 0/114 N/A (0%)
  - `date`: 0/114 N/A (0%)
  - `author`: 2/114 N/A (2%)
  - `article body text`: 0/114 N/A (0%)
- **Errors (5):**
  - EXECUTION ERROR: 'charmap' codec can't encode character '\u200e' in position 317: character maps to <undefined>  Traceback (most recent call last):   File "<string>", line 52, in <module>     main()  
  - EXECUTION ERROR: 'charmap' codec can't encode character '\u200c' in position 1289: character maps to <undefined>  Traceback (most recent call last):   File "<string>", line 52, in <module>     main() 
  - EXECUTION ERROR: 'charmap' codec can't encode characters in position 2373-2374: character maps to <undefined>  Traceback (most recent call last):   File "<string>", line 52, in <module>     main()    
  - EXECUTION ERROR: 'charmap' codec can't encode character '\u200e' in position 304: character maps to <undefined>  Traceback (most recent call last):   File "<string>", line 52, in <module>     main()  
  - EXECUTION ERROR: 'charmap' codec can't encode character '\u200c' in position 5868: character maps to <undefined>  Traceback (most recent call last):   File "<string>", line 52, in <module>     main() 


## Run 2026-07-22 12:22:50

- **Run directory:** `orch_runs/run_20260722_121323`
- **Model:** gemma-4-31b-it
- **Pagination type:** numbered pagination
- **Input URLs:**
  - https://www.youm7.com/Section/%D8%A3%D8%AE%D8%A8%D8%A7%D8%B1-%D8%B9%D8%A7%D8%AC%D9%84%D8%A9/65/1
  - https://www.youm7.com/Section/%D8%A3%D8%AE%D8%A8%D8%A7%D8%B1-%D8%B9%D8%A7%D8%AC%D9%84%D8%A9/65/2
- **Requirements:** 'title, date, author, article body text'
- **Pages requested:** 3
- **Pages processed:** 3
- **Articles extracted:** 120
- **Articles failed:** 0
- **Clusters (unique structures):** 1
- **LLM calls:** 2
  - Article agent: 1
  - Links agent: 1
- **LLM calls per article:** 0.02
- **Code reuse rate:** 119/120 articles reused cluster code (99%)
- **Tokens:** 29,527 total (25,675 prompt + 720 output)
  - Article agent: 15,694 tokens
  - Links agent: 13,833 tokens
- **Estimated cost:** $0.0021 (at $0.075/$0.3 per 1M input/output tokens)
- **Estimated cost per 1,000 articles:** $0.0178
- **Structural-map depth:** 2 agent call(s), 2 shrunk below full depth 10 to fit the input-token budget
  - Links agent: depth 8 (~10,879 input tokens)  ⚠ reduced
  - Article agent: depth 9 (~14,794 input tokens)  ⚠ reduced
- **Total time:** 9m 44s
- **Fetch method:** 120 via requests (100%), 0 via browser (0%)
- **Missing/N/A values (of 120 extracted):**
  - `title`: 0/120 N/A (0%)
  - `date`: 0/120 N/A (0%)
  - `author`: 2/120 N/A (2%)
  - `article body text`: 0/120 N/A (0%)
- **Errors:** none


## Run 2026-07-22 13:35:20

- **Run directory:** `orch_runs/run_20260722_124036`
- **Model:** gemma-4-31b-it
- **Pagination type:** single page
- **Input URLs:**
  - https://www.youm7.com/Section/%D8%A3%D8%AE%D8%A8%D8%A7%D8%B1-%D8%B9%D8%A7%D8%AC%D9%84%D8%A9/65/1
- **Requirements:** title, date, author, article body text
- **Pages requested:** 1
- **Pages processed:** 0
- **Articles extracted:** 0
- **Articles failed:** 0
- **Clusters (unique structures):** 0
- **LLM calls:** 1
  - Links agent: 1
- **Total time:** 54m 44s
- **Errors (1):**
  - Links extraction failed: All 3 attempt(s) failed. Last error: LLM ERROR: Error generating code: 429 You exceeded your current quota, please check your plan and billing details. For more information on


## Run 2026-07-22 13:55:48

- **Run directory:** `orch_runs/run_20260722_135048`
- **Model:** gemma-4-31b-it
- **Pagination type:** numbered pagination
- **Input URLs:**
  - https://www.youm7.com/Section/%D8%A3%D8%AE%D8%A8%D8%A7%D8%B1-%D8%B9%D8%A7%D8%AC%D9%84%D8%A9/65/1
  - https://www.youm7.com/Section/%D8%A3%D8%AE%D8%A8%D8%A7%D8%B1-%D8%B9%D8%A7%D8%AC%D9%84%D8%A9/65/2
- **Requirements:** title, date, author, article body text
- **Pages requested:** 3
- **Pages processed:** 0
- **Articles extracted:** 0
- **Articles failed:** 0
- **Clusters (unique structures):** 0
- **LLM calls:** 1
  - Links agent: 1
- **Total time:** 5m 0s
- **Errors (1):**
  - Links extraction failed: Subprocess timed out (300s)


## Run 2026-07-22 14:16:50

- **Run directory:** `orch_runs/run_20260722_140951`
- **Model:** gemma-4-31b-it
- **Pagination type:** numbered pagination
- **Input URLs:**
  - https://www.youm7.com/Section/%D8%A3%D8%AE%D8%A8%D8%A7%D8%B1-%D8%B9%D8%A7%D8%AC%D9%84%D8%A9/65/1
  - https://www.youm7.com/Section/%D8%A3%D8%AE%D8%A8%D8%A7%D8%B1-%D8%B9%D8%A7%D8%AC%D9%84%D8%A9/65/2
- **Requirements:** title, date, author, article body text
- **Pages requested:** 3
- **Pages processed:** 3
- **Articles extracted:** 118
- **Articles failed:** 0
- **Clusters (unique structures):** 1
- **LLM calls:** 2
  - Article agent: 1
  - Links agent: 1
- **LLM calls per article:** 0.02
- **Code reuse rate:** 117/118 articles reused cluster code (99%)
- **Tokens:** 30,414 total (25,486 prompt + 786 output)
  - Article agent: 15,824 tokens
  - Links agent: 14,590 tokens
- **Estimated cost:** $0.0021 (at $0.075/$0.3 per 1M input/output tokens)
- **Estimated cost per 1,000 articles:** $0.0182
- **Structural-map depth:** 2 agent call(s), 2 shrunk below full depth 10 to fit the input-token budget
  - Links agent: depth 8 (~10,911 input tokens)  ⚠ reduced
  - Article agent: depth 9 (~14,573 input tokens)  ⚠ reduced
- **Total time:** 6m 59s
- **Fetch method:** 118 via requests (100%), 0 via browser (0%)
- **Missing/N/A values (of 118 extracted):**
  - `title`: 0/118 N/A (0%)
  - `date`: 0/118 N/A (0%)
  - `author`: 1/118 N/A (1%)
  - `article body text`: 0/118 N/A (0%)
- **Errors:** none


## Run 2026-07-22 15:35:30

- **Run directory:** `orch_runs/run_20260722_145504`
- **Model:** gemma-4-31b-it
- **Pagination type:** numbered pagination
- **Input URLs:**
  - https://www.youm7.com/Section/%D8%A3%D8%AE%D8%A8%D8%A7%D8%B1-%D8%B9%D8%A7%D8%AC%D9%84%D8%A9/65/1
  - https://www.youm7.com/Section/%D8%A3%D8%AE%D8%A8%D8%A7%D8%B1-%D8%B9%D8%A7%D8%AC%D9%84%D8%A9/65/2
- **Requirements:** title, date, author, article body text
- **Pages requested:** 35
- **Pages processed:** 35
- **Articles extracted:** 1398
- **Articles failed:** 1
- **Clusters (unique structures):** 1
- **LLM calls:** 2
  - Article agent: 1
  - Links agent: 1
- **LLM calls per article:** 0.00
- **Code reuse rate:** 1397/1398 articles reused cluster code (100%)
- **Tokens:** 30,605 total (25,082 prompt + 782 output)
  - Article agent: 15,323 tokens
  - Links agent: 15,282 tokens
- **Estimated cost:** $0.0021 (at $0.075/$0.3 per 1M input/output tokens)
- **Estimated cost per 1,000 articles:** $0.0015
- **Structural-map depth:** 2 agent call(s), 2 shrunk below full depth 10 to fit the input-token budget
  - Links agent: depth 8 (~10,908 input tokens)  ⚠ reduced
  - Article agent: depth 9 (~14,172 input tokens)  ⚠ reduced
- **Total time:** 40m 27s
- **Fetch method:** 1399 via requests (100%), 0 via browser (0%)
- **Missing/N/A values (of 1398 extracted):**
  - `title`: 0/1398 N/A (0%)
  - `date`: 0/1398 N/A (0%)
  - `author`: 25/1398 N/A (2%)
  - `article body text`: 0/1398 N/A (0%)
- **Errors (1):**
  - SANDBOX ERROR: Invalid JSON from worker: Expecting value: line 1 column 1 (char 0)


## Run 2026-07-22 19:52:05

- **Run directory:** `orch_runs/run_20260722_194427`
- **Model:** gemma-4-31b-it
- **Pagination type:** numbered pagination
- **Input URLs:**
  - https://www.youm7.com/Section/%D8%A3%D8%AE%D8%A8%D8%A7%D8%B1-%D8%B9%D8%A7%D8%AC%D9%84%D8%A9/65/1
  - https://www.youm7.com/Section/%D8%A3%D8%AE%D8%A8%D8%A7%D8%B1-%D8%B9%D8%A7%D8%AC%D9%84%D8%A9/65/2
- **Requirements:** title, date, author, article body text
- **Pages requested:** 3
- **Pages processed:** 3
- **Articles extracted:** 126
- **Articles failed:** 0
- **Clusters (unique structures):** 1
- **LLM calls:** 2
  - Article agent: 1
  - Links agent: 1
- **LLM calls per article:** 0.02
- **Code reuse rate:** 125/126 articles reused cluster code (99%)
- **Tokens:** 31,387 total (25,150 prompt + 784 output)
  - Links agent: 15,981 tokens
  - Article agent: 15,406 tokens
- **Estimated cost:** $0.0021 (at $0.075/$0.3 per 1M input/output tokens)
- **Estimated cost per 1,000 articles:** $0.0168
- **Structural-map depth:** 2 agent call(s), 2 shrunk below full depth 10 to fit the input-token budget
  - Links agent: depth 8 (~10,923 input tokens)  ⚠ reduced
  - Article agent: depth 9 (~14,225 input tokens)  ⚠ reduced
- **Total time:** 7m 38s
- **Fetch method:** 126 via requests (100%), 0 via browser (0%)
- **Missing/N/A values (of 126 extracted):**
  - `title`: 0/126 N/A (0%)
  - `date`: 0/126 N/A (0%)
  - `author`: 3/126 N/A (2%)
  - `article body text`: 0/126 N/A (0%)
- **Errors:** none


## Run 2026-07-23 23:59:39

- **Run directory:** `orch_runs/run_20260723_235507`
- **Model:** gemma-4-31b-it
- **Pagination type:** infinite scroll
- **Input URLs:**
  - https://aawsat.com/%D8%A7%D9%84%D8%B1%D8%A3%D9%8A
- **Requirements:** title, date, author, article body text
- **Pages requested:** 1
- **Pages processed:** 0
- **Articles extracted:** 0
- **Articles failed:** 0
- **Clusters (unique structures):** 0
- **Links dropped (non-article filter):** 0
- **LLM calls:** 1
  - Links agent: 1
- **Total time:** 4m 32s
- **Errors (1):**
  - Links extraction failed: All 3 attempt(s) failed. Last error: AST SAFETY ERROR: Disallowed syntax: ImportFrom


## Run 2026-07-24 00:11:52

- **Run directory:** `orch_runs/run_20260724_001043`
- **Model:** gemma-4-31b-it
- **Pagination type:** infinite scroll
- **Input URLs:**
  - - https://aawsat.com/%D8%A7%D9%84%D8%B1%D8%A3%D9%8A
- **Requirements:** 'title, date, author, article body text
- **Pages requested:** 1
- **Pages processed:** 0
- **Articles extracted:** 0
- **Articles failed:** 0
- **Clusters (unique structures):** 0
- **Links dropped (non-article filter):** 0
- **LLM calls:** 1
  - Links agent: 1
- **Total time:** 1m 39s
- **Errors (1):**
  - Links extraction failed: Failed to fetch page structure for - https://aawsat.com/%D8%A7%D9%84%D8%B1%D8%A3%D9%8A  ── Headless browser attempt ── [headless] Error: Page.goto: Protocol error (Page.naviga


## Run 2026-07-24 00:19:45

- **Run directory:** `orch_runs/run_20260724_001427`
- **Model:** gemma-4-31b-it
- **Pagination type:** infinite scroll
- **Input URLs:**
  - https://aawsat.com/%D8%A7%D9%84%D8%B1%D8%A3%D9%8A
- **Requirements:** 'title, date, author, article body text
- **Pages requested:** 1
- **Pages processed:** 0
- **Articles extracted:** 0
- **Articles failed:** 0
- **Clusters (unique structures):** 0
- **Links dropped (non-article filter):** 0
- **LLM calls:** 1
  - Links agent: 1
- **Total time:** 5m 36s
- **Errors (1):**
  - Links extraction failed: Subprocess timed out (300s)


## Run 2026-07-24 00:33:44

- **Run directory:** `orch_runs/run_20260724_002914`
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
- **Links dropped (non-article filter):** 0
- **LLM calls:** 1
  - Links agent: 1
- **Tokens:** 28,353 total (21,168 prompt + 1,387 output)
  - Links agent: 28,353 tokens
- **Estimated cost:** $0.0020 (at $0.075/$0.3 per 1M input/output tokens)
- **Structural-map depth:** 1 agent call(s), 0 shrunk below full depth 10 to fit the input-token budget
  - Links agent: depth 10 (~7,570 input tokens)
- **Total time:** 4m 44s
- **Errors (1):**
  - No article links found on page 1


## Run 2026-07-24 00:48:34

- **Run directory:** `orch_runs/run_20260724_004324`
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
- **Links dropped (non-article filter):** 0
- **LLM calls:** 1
  - Links agent: 1
- **Tokens:** 30,579 total (21,267 prompt + 1,375 output)
  - Links agent: 30,579 tokens
- **Estimated cost:** $0.0020 (at $0.075/$0.3 per 1M input/output tokens)
- **Structural-map depth:** 1 agent call(s), 0 shrunk below full depth 10 to fit the input-token budget
  - Links agent: depth 10 (~7,603 input tokens)
- **Total time:** 5m 22s
- **Errors (1):**
  - No article links found on page 1


## Run 2026-07-24 01:31:35

- **Run directory:** `orch_runs/run_20260724_012900`
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
- **Links dropped (non-article filter):** 0
- **LLM calls:** 1
  - Links agent: 1
- **Total time:** 2m 40s
- **Errors (1):**
  - Links extraction failed: All 3 attempt(s) failed. Last error: LLM ERROR: Error generating code: 429 You exceeded your current quota, please check your plan and billing details. For more information on


## Run 2026-07-24 01:57:16

- **Run directory:** `orch_runs/run_20260724_015110`
- **Model:** gemma-4-31b-it
- **Pagination type:** infinite scroll
- **Input URLs:**
  - https://aawsat.com/%D8%A7%D9%84%D8%B1%D8%A3%D9%8A
- **Requirements:** 'title, date, author, article body text'
- **Pages requested:** 1
- **Pages processed:** 1
- **Articles extracted:** 83
- **Articles failed:** 0
- **Clusters (unique structures):** 1
- **Links dropped (non-article filter):** 0
- **LLM calls:** 2
  - Article agent: 1
  - Links agent: 1
- **LLM calls per article:** 0.02
- **Code reuse rate:** 82/83 articles reused cluster code (99%)
- **Tokens:** 27,986 total (19,736 prompt + 893 output)
  - Links agent: 16,750 tokens
  - Article agent: 11,236 tokens
- **Estimated cost:** $0.0017 (at $0.075/$0.3 per 1M input/output tokens)
- **Estimated cost per 1,000 articles:** $0.0211
- **Structural-map depth:** 2 agent call(s), 0 shrunk below full depth 10 to fit the input-token budget
  - Links agent: depth 11 (~13,798 input tokens)
  - Article agent: depth 10 (~5,936 input tokens)
- **Total time:** 6m 14s
- **Fetch method:** 83 via requests (100%), 0 via browser (0%)
- **Missing/N/A values (of 83 extracted):**
  - `title`: 0/83 N/A (0%)
  - `date`: 83/83 N/A (100%)
  - `author`: 83/83 N/A (100%)
  - `article body text`: 0/83 N/A (0%)
- **Errors:** none


## Run 2026-07-24 04:59:24

- **Run directory:** `orch_runs/run_20260724_045636`
- **Model:** gemma-4-31b-it
- **Pagination type:** infinite scroll
- **Input URLs:**
  - https://www.independentarabia.com/%D8%AB%D9%82%D8%A7%D9%81%D8%A9/%D8%B3%D9%8A%D9%86%D9%85%D8%A7
- **Requirements:** title, date, author, article body text
- **Pages requested:** 1
- **Pages processed:** 1
- **Articles extracted:** 0
- **Articles failed:** 4
- **Clusters (unique structures):** 0
- **Links dropped (non-article filter):** 0
- **LLM calls:** 2
  - Article agent: 1
  - Links agent: 1
- **Tokens:** 15,300 total (13,295 prompt + 482 output)
  - Links agent: 15,300 tokens
- **Estimated cost:** $0.0011 (at $0.075/$0.3 per 1M input/output tokens)
- **Structural-map depth:** 1 agent call(s), 1 shrunk below full depth 10 to fit the input-token budget
  - Links agent: depth 6 (~13,294 input tokens)  ⚠ reduced
- **Total time:** 2m 48s
- **Fetch method:** 4 via requests (100%), 0 via browser (0%)
- **Errors (4):**
  - Agent failed on new cluster representative: All 1 attempt(s) failed. Last error: LLM ERROR: Error generating code: 429 You exceeded your current quota, please check your plan and billing details. For 
  - Agent failed on new cluster representative: All 1 attempt(s) failed. Last error: LLM ERROR: Error generating code: 429 You exceeded your current quota, please check your plan and billing details. For 
  - Agent failed on new cluster representative: All 1 attempt(s) failed. Last error: LLM ERROR: Error generating code: 429 You exceeded your current quota, please check your plan and billing details. For 
  - Agent failed on new cluster representative: All 1 attempt(s) failed. Last error: LLM ERROR: Error generating code: 429 You exceeded your current quota, please check your plan and billing details. For 


## Run 2026-07-24 05:03:36

- **Run directory:** `orch_runs/run_20260724_045823`
- **Model:** gemma-4-31b-it
- **Pagination type:** load more button
- **Input URLs:**
  - https://www.arageek.com/tech
- **Requirements:** title, date, author, article body text
- **Pages requested:** 1
- **Pages processed:** 1
- **Articles extracted:** 68
- **Articles failed:** 0
- **Clusters (unique structures):** 1
- **Links dropped (non-article filter):** 0
- **LLM calls:** 2
  - Article agent: 1
  - Links agent: 1
- **LLM calls per article:** 0.03
- **Code reuse rate:** 67/68 articles reused cluster code (99%)
- **Tokens:** 33,175 total (28,456 prompt + 886 output)
  - Links agent: 16,817 tokens
  - Article agent: 16,358 tokens
- **Estimated cost:** $0.0024 (at $0.075/$0.3 per 1M input/output tokens)
- **Estimated cost per 1,000 articles:** $0.0353
- **Structural-map depth:** 2 agent call(s), 1 shrunk below full depth 10 to fit the input-token budget
  - Links agent: depth 6 (~14,243 input tokens)  ⚠ reduced
  - Article agent: depth 16 (~14,211 input tokens)
- **Total time:** 5m 13s
- **Fetch method:** 68 via requests (100%), 0 via browser (0%)
- **Missing/N/A values (of 68 extracted):**
  - `title`: 0/68 N/A (0%)
  - `date`: 0/68 N/A (0%)
  - `author`: 0/68 N/A (0%)
  - `article body text`: 0/68 N/A (0%)
- **Errors:** none


## Run 2026-07-24 05:18:38

- **Run directory:** `orch_runs/run_20260724_051233`
- **Model:** gemma-4-31b-it
- **Pagination type:** infinite scroll
- **Input URLs:**
  - https://www.independentarabia.com/%D8%AB%D9%82%D8%A7%D9%81%D8%A9/%D8%B3%D9%8A%D9%86%D9%85%D8%A7
- **Requirements:** title, date, author, article body text
- **Pages requested:** 1
- **Pages processed:** 1
- **Articles extracted:** 37
- **Articles failed:** 0
- **Clusters (unique structures):** 1
- **Links dropped (non-article filter):** 0
- **LLM calls:** 2
  - Article agent: 1
  - Links agent: 1
- **LLM calls per article:** 0.05
- **Code reuse rate:** 36/37 articles reused cluster code (97%)
- **Tokens:** 32,060 total (27,773 prompt + 794 output)
  - Article agent: 16,434 tokens
  - Links agent: 15,626 tokens
- **Estimated cost:** $0.0023 (at $0.075/$0.3 per 1M input/output tokens)
- **Estimated cost per 1,000 articles:** $0.0627
- **Structural-map depth:** 2 agent call(s), 1 shrunk below full depth 10 to fit the input-token budget
  - Links agent: depth 6 (~13,284 input tokens)  ⚠ reduced
  - Article agent: depth 14 (~14,487 input tokens)
- **Total time:** 6m 4s
- **Fetch method:** 37 via requests (100%), 0 via browser (0%)
- **Missing/N/A values (of 37 extracted):**
  - `title`: 0/37 N/A (0%)
  - `date`: 0/37 N/A (0%)
  - `author`: 0/37 N/A (0%)
  - `article body text`: 0/37 N/A (0%)
- **Errors:** none


## Run 2026-07-28 13:49:38

- **Run directory:** `orch_runs/run_20260728_134505`
- **Model:** gemma-4-31b-it
- **Pagination type:** infinite scroll
- **Input URLs:**
  - https://www.independentarabia.com/%D8%AB%D9%82%D8%A7%D9%81%D8%A9/%D8%B3%D9%8A%D9%86%D9%85%D8%A7
- **Requirements:** title, date, author, article body text
- **Pages requested:** 1
- **Pages processed:** 1
- **Articles extracted:** 4
- **Articles failed:** 0
- **Clusters (unique structures):** 2
- **Links dropped (non-article filter):** 0
- **LLM calls:** 3
  - Article agent: 2
  - Links agent: 1
- **LLM calls per article:** 0.75
- **Code reuse rate:** 2/4 articles reused cluster code (50%)
- **Tokens:** 47,462 total (42,498 prompt + 1,126 output)
  - Article agent: 31,388 tokens
  - Links agent: 16,074 tokens
- **Estimated cost:** $0.0035 (at $0.075/$0.3 per 1M input/output tokens)
- **Estimated cost per 1,000 articles:** $0.8813
- **Structural-map depth:** 3 agent call(s), 1 shrunk below full depth 10 to fit the input-token budget
  - Links agent: depth 6 (~13,457 input tokens)  ⚠ reduced
  - Article agent: depth 14 (~14,565 input tokens)
  - Article agent: depth 14 (~14,473 input tokens)
- **Total time:** 4m 33s
- **Fetch method:** 4 via requests (100%), 0 via browser (0%)
- **Missing/N/A values (of 4 extracted):**
  - `title`: 0/4 N/A (0%)
  - `date`: 0/4 N/A (0%)
  - `author`: 0/4 N/A (0%)
  - `article body text`: 0/4 N/A (0%)
- **Errors:** none


## Run 2026-07-28 14:16:56

- **Run directory:** `orch_runs/run_20260728_141124`
- **Model:** gemma-4-31b-it
- **Pagination type:** infinite scroll
- **Scroll rounds requested:** 4
- **Input URLs:**
  - https://www.independentarabia.com/%D8%AB%D9%82%D8%A7%D9%81%D8%A9/%D8%B3%D9%8A%D9%86%D9%85%D8%A7
- **Requirements:** title, date, author, article body text
- **Pages requested:** 1
- **Pages processed:** 1
- **Articles extracted:** 34
- **Articles failed:** 0
- **Clusters (unique structures):** 2
- **Links dropped (non-article filter):** 0
- **LLM calls:** 3
  - Article agent: 2
  - Links agent: 1
- **LLM calls per article:** 0.09
- **Code reuse rate:** 32/34 articles reused cluster code (94%)
- **Tokens:** 48,257 total (43,325 prompt + 1,128 output)
  - Article agent: 31,611 tokens
  - Links agent: 16,646 tokens
- **Estimated cost:** $0.0036 (at $0.075/$0.3 per 1M input/output tokens)
- **Estimated cost per 1,000 articles:** $0.1055
- **Structural-map depth:** 3 agent call(s), 1 shrunk below full depth 10 to fit the input-token budget
  - Links agent: depth 6 (~14,192 input tokens)  ⚠ reduced
  - Article agent: depth 14 (~14,565 input tokens)
  - Article agent: depth 14 (~14,565 input tokens)
- **Total time:** 5m 32s
- **Fetch method:** 34 via requests (100%), 0 via browser (0%)
- **Missing/N/A values (of 34 extracted):**
  - `title`: 0/34 N/A (0%)
  - `date`: 0/34 N/A (0%)
  - `author`: 0/34 N/A (0%)
  - `article_body_text`: 2/34 N/A (6%)
  - `body_text`: 32/34 N/A (94%)
- **Errors:** none


## Run 2026-07-28 16:14:49

- **Run directory:** `orch_runs/run_20260728_161411`
- **Model:** gemma-3-27b-it
- **Pagination type:** infinite scroll
- **Scroll rounds requested:** 4
- **Input URLs:**
  - https://www.independentarabia.com/%D8%AB%D9%82%D8%A7%D9%81%D8%A9/%D8%B3%D9%8A%D9%86%D9%85%D8%A7
- **Requirements:** title, date, author, article body text
- **Pages requested:** 1
- **Pages processed:** 0
- **Articles extracted:** 0
- **Articles failed:** 0
- **Clusters (unique structures):** 0
- **Links dropped (non-article filter):** 0
- **LLM calls:** 1
  - Links agent: 1
- **Total time:** 0m 39s
- **Errors (1):**
  - Links extraction failed: All 3 attempt(s) failed. Last error: LLM ERROR: Error generating code: 404 models/gemma-3-27b-it is not found for API version v1beta, or is not supported for generateContent. 


## Run 2026-07-28 16:23:56

- **Run directory:** `orch_runs/run_20260728_161728`
- **Model:** gemma-4-31b-it
- **Pagination type:** infinite scroll
- **Scroll rounds requested:** 4
- **Input URLs:**
  - https://www.independentarabia.com/%D8%AB%D9%82%D8%A7%D9%81%D8%A9/%D8%B3%D9%8A%D9%86%D9%85%D8%A7
- **Requirements:** title, date, author, article body text
- **Pages requested:** 1
- **Pages processed:** 1
- **Articles extracted:** 34
- **Articles failed:** 0
- **Clusters (unique structures):** 2
- **Links dropped (non-article filter):** 0
- **Link recovery guard:** triggered on 1 page(s)
  - page 1: 4 generated accepted -> 34 after recovery (+30)
- **LLM calls:** 3
  - Article agent: 2
  - Links agent: 1
- **LLM calls per article:** 0.09
- **Code reuse rate:** 32/34 articles reused cluster code (94%)
- **Tokens:** 50,823 total (43,494 prompt + 1,099 output)
  - Article agent: 32,794 tokens
  - Links agent: 18,029 tokens
- **Estimated cost:** $0.0036 (at $0.075/$0.3 per 1M input/output tokens)
- **Estimated cost per 1,000 articles:** $0.1056
- **Structural-map depth:** 3 agent call(s), 1 shrunk below full depth 10 to fit the input-token budget
  - Links agent: depth 6 (~14,183 input tokens)  ⚠ reduced
  - Article agent: depth 14 (~14,740 input tokens)
  - Article agent: depth 14 (~14,568 input tokens)
- **Total time:** 6m 28s
- **Fetch method:** 34 via requests (100%), 0 via browser (0%)
- **Missing/N/A values (of 34 extracted):**
  - `title`: 0/34 N/A (0%)
  - `date`: 0/34 N/A (0%)
  - `author`: 0/34 N/A (0%)
  - `article_body_text`: 2/34 N/A (6%)
  - `article body text`: 32/34 N/A (94%)
- **Errors:** none


## Run 2026-07-28 19:44:11

- **Run directory:** `orch_runs/run_20260728_193947`
- **Model:** gemma-4-31b-it
- **Pagination type:** infinite scroll
- **Scroll rounds requested:** 4
- **Input URLs:**
  - https://www.independentarabia.com/%D8%AB%D9%82%D8%A7%D9%81%D8%A9/%D8%B3%D9%8A%D9%86%D9%85%D8%A7
- **Requirements:** title, date, author, article body text
- **Pages requested:** 1
- **Pages processed:** 1
- **Articles extracted:** 34
- **Articles failed:** 0
- **Clusters (unique structures):** 2
- **Links dropped (non-article filter):** 0
- **Link recovery guard:** triggered on 1 page(s)
  - page 1: 4 generated accepted -> 34 after recovery (+30)
- **LLM calls:** 3
  - Article agent: 2
  - Links agent: 1
- **LLM calls per article:** 0.09
- **Code reuse rate:** 32/34 articles reused cluster code (94%)
- **Tokens:** 48,426 total (43,494 prompt + 1,088 output)
  - Article agent: 31,440 tokens
  - Links agent: 16,986 tokens
- **Estimated cost:** $0.0036 (at $0.075/$0.3 per 1M input/output tokens)
- **Estimated cost per 1,000 articles:** $0.1055
- **Structural-map depth:** 3 agent call(s), 1 shrunk below full depth 10 to fit the input-token budget
  - Links agent: depth 6 (~14,183 input tokens)  ⚠ reduced
  - Article agent: depth 14 (~14,740 input tokens)
  - Article agent: depth 14 (~14,568 input tokens)
- **Total time:** 4m 24s
- **Fetch method:** 34 via requests (100%), 0 via browser (0%)
- **Missing/N/A values (of 34 extracted):**
  - `title`: 0/34 N/A (0%)
  - `date`: 0/34 N/A (0%)
  - `author`: 0/34 N/A (0%)
  - `article body text`: 0/34 N/A (0%)
- **Errors:** none


## Run 2026-07-28 19:52:42

- **Run directory:** `orch_runs/run_20260728_194601`
- **Model:** gemma-4-31b-it
- **Pagination type:** infinite scroll
- **Scroll rounds requested:** 1
- **Input URLs:**
  - https://www.independentarabia.com/%D8%AB%D9%82%D8%A7%D9%81%D8%A9/%D8%B3%D9%8A%D9%86%D9%85%D8%A7
- **Requirements:** title, date, author, article body text
- **Pages requested:** 1
- **Pages processed:** 1
- **Articles extracted:** 25
- **Articles failed:** 0
- **Clusters (unique structures):** 2
- **Links dropped (non-article filter):** 0
- **Link recovery guard:** triggered on 1 page(s)
  - page 1: 4 generated accepted -> 25 after recovery (+21)
- **LLM calls:** 3
  - Article agent: 2
  - Links agent: 1
- **LLM calls per article:** 0.12
- **Code reuse rate:** 23/25 articles reused cluster code (92%)
- **Tokens:** 49,865 total (43,069 prompt + 1,051 output)
  - Article agent: 31,862 tokens
  - Links agent: 18,003 tokens
- **Estimated cost:** $0.0035 (at $0.075/$0.3 per 1M input/output tokens)
- **Estimated cost per 1,000 articles:** $0.1418
- **Structural-map depth:** 3 agent call(s), 1 shrunk below full depth 10 to fit the input-token budget
  - Links agent: depth 6 (~13,758 input tokens)  ⚠ reduced
  - Article agent: depth 14 (~14,740 input tokens)
  - Article agent: depth 14 (~14,568 input tokens)
- **Total time:** 6m 41s
- **Fetch method:** 25 via requests (100%), 0 via browser (0%)
- **Missing/N/A values (of 25 extracted):**
  - `title`: 0/25 N/A (0%)
  - `date`: 0/25 N/A (0%)
  - `author`: 0/25 N/A (0%)
  - `article body text`: 0/25 N/A (0%)
- **Errors:** none


## Run 2026-07-28 20:11:37

- **Run directory:** `orch_runs/run_20260728_195312`
- **Model:** gemma-4-31b-it
- **Pagination type:** infinite scroll
- **Scroll rounds requested:** 100
- **Input URLs:**
  - https://aawsat.com/%D8%A7%D9%84%D8%B1%D8%A3%D9%8A
- **Requirements:** title, date, author, article body text
- **Pages requested:** 1
- **Pages processed:** 1
- **Articles extracted:** 500
- **Articles failed:** 0
- **Clusters (unique structures):** 1
- **Links dropped (non-article filter):** 0
- **LLM calls:** 2
  - Article agent: 1
  - Links agent: 1
- **LLM calls per article:** 0.00
- **Code reuse rate:** 499/500 articles reused cluster code (100%)
- **Tokens:** 48,638 total (39,511 prompt + 1,131 output)
  - Links agent: 32,158 tokens
  - Article agent: 16,480 tokens
- **Estimated cost:** $0.0033 (at $0.075/$0.3 per 1M input/output tokens)
- **Estimated cost per 1,000 articles:** $0.0066
- **Structural-map depth:** 2 agent call(s), 0 shrunk below full depth 10 to fit the input-token budget
  - Links agent: depth 11 (~13,567 input tokens)
  - Article agent: depth 13 (~13,262 input tokens)
- **Total time:** 18m 26s
- **Fetch method:** 500 via requests (100%), 0 via browser (0%)
- **Missing/N/A values (of 500 extracted):**
  - `title`: 500/500 N/A (100%)
  - `date`: 500/500 N/A (100%)
  - `author`: 500/500 N/A (100%)
  - `article_body_text`: 499/500 N/A (100%)
- **Errors:** none


## Run 2026-07-29 00:14:27

- **Run directory:** `orch_runs/run_20260729_000940`
- **Model:** gemma-4-31b-it
- **Pagination type:** infinite scroll
- **Scroll rounds requested:** 3
- **Input URLs:**
  - https://aawsat.com/%D8%A7%D9%84%D8%B1%D8%A3%D9%8A
- **Requirements:** title, date, author, article body text
- **Pages requested:** 1
- **Pages processed:** 1
- **Articles extracted:** 0
- **Articles failed:** 61
- **Clusters (unique structures):** 0
- **Links dropped (non-article filter):** 0
- **Article extractor validation:** 1 warning(s)
  - cluster 81726d321529 representative attempt 1: skip - `title` is N/A; `date` is N/A; `author` is N/A
- **LLM calls:** 2
  - Article agent: 1
  - Links agent: 1
- **Tokens:** 34,489 total (26,824 prompt + 1,018 output)
  - Links agent: 17,504 tokens
  - Article agent: 16,985 tokens
- **Estimated cost:** $0.0023 (at $0.075/$0.3 per 1M input/output tokens)
- **Structural-map depth:** 2 agent call(s), 0 shrunk below full depth 10 to fit the input-token budget
  - Links agent: depth 11 (~13,569 input tokens)
  - Article agent: depth 13 (~13,253 input tokens)
- **Total time:** 4m 46s
- **Fetch method:** 61 via requests (100%), 0 via browser (0%)
- **Errors (61):**
  - Agent failed on new cluster representative: Article extractor rejected by validation
  - Agent failed on new cluster representative: Article extractor rejected by validation
  - Agent failed on new cluster representative: Article extractor rejected by validation
  - Agent failed on new cluster representative: Article extractor rejected by validation
  - Agent failed on new cluster representative: Article extractor rejected by validation
  - Agent failed on new cluster representative: Article extractor rejected by validation
  - Agent failed on new cluster representative: Article extractor rejected by validation
  - Agent failed on new cluster representative: Article extractor rejected by validation
  - Agent failed on new cluster representative: Article extractor rejected by validation
  - Agent failed on new cluster representative: Article extractor rejected by validation
  - ...and 51 more


## Run 2026-07-29 00:25:09

- **Run directory:** `orch_runs/run_20260729_001903`
- **Model:** gemma-4-31b-it
- **Pagination type:** infinite scroll
- **Scroll rounds requested:** 1
- **Input URLs:**
  - https://aawsat.com/%D8%A7%D9%84%D8%B1%D8%A3%D9%8A
- **Requirements:** title, date, author, article body text
- **Pages requested:** 1
- **Pages processed:** 0
- **Articles extracted:** 0
- **Articles failed:** 0
- **Clusters (unique structures):** 0
- **Links dropped (non-article filter):** 0
- **LLM calls:** 1
  - Links agent: 1
- **Tokens:** 49,039 total (36,911 prompt + 1,473 output)
  - Links agent: 49,039 tokens
- **Estimated cost:** $0.0032 (at $0.075/$0.3 per 1M input/output tokens)
- **Structural-map depth:** 1 agent call(s), 1 shrunk below full depth 10 to fit the input-token budget
  - Links agent: depth 9 (~12,830 input tokens)  ⚠ reduced
- **Total time:** 6m 7s
- **Errors (1):**
  - No article links found on page 1


## Run 2026-07-29 00:58:03

- **Run directory:** `orch_runs/run_20260729_004338`
- **Model:** gemma-4-31b-it
- **Pagination type:** infinite scroll
- **Scroll rounds requested:** 1
- **Input URLs:**
  - https://aawsat.com/%D8%A7%D9%84%D8%B1%D8%A3%D9%8A
- **Requirements:** title, date, author, article body text
- **Pages requested:** 1
- **Pages processed:** 1
- **Articles extracted:** 12
- **Articles failed:** 37
- **Clusters (unique structures):** 1
- **Links dropped (non-article filter):** 0
- **Link recovery guard:** triggered on 1 page(s)
  - page 1: 0 generated accepted -> 49 after recovery (+49)
- **Article extractor validation:** 3 warning(s)
  - cluster 81726d321529 representative attempt 1: retry - `title` is N/A; `author` is N/A; `article_body_text` is N/A
  - cluster 81726d321529 representative attempt 2: skip - `title` is N/A; `author` is N/A; `article body text` is N/A
  - cluster 341cd4b49fe8 representative attempt 1: accept - `author` is N/A
- **LLM calls:** 4
  - Article agent: 3
  - Links agent: 1
- **LLM calls per article:** 0.33
- **Code reuse rate:** 9/12 articles reused cluster code (75%)
- **Tokens:** 95,038 total (76,147 prompt + 2,383 output)
  - Article agent: 48,299 tokens
  - Links agent: 46,739 tokens
- **Estimated cost:** $0.0064 (at $0.075/$0.3 per 1M input/output tokens)
- **Estimated cost per 1,000 articles:** $0.5355
- **Structural-map depth:** 4 agent call(s), 1 shrunk below full depth 10 to fit the input-token budget
  - Links agent: depth 9 (~12,835 input tokens)  ⚠ reduced
  - Article agent: depth 13 (~13,260 input tokens)
  - Article agent: depth 13 (~13,496 input tokens)
  - Article agent: depth 13 (~12,559 input tokens)
- **Total time:** 14m 25s
- **Fetch method:** 49 via requests (100%), 0 via browser (0%)
- **Missing/N/A values (of 12 extracted):**
  - `title`: 0/12 N/A (0%)
  - `date`: 0/12 N/A (0%)
  - `author`: 12/12 N/A (100%)
  - `article body text`: 0/12 N/A (0%)
- **Errors (37):**
  - Agent failed on new cluster representative: Article extractor rejected by validation
  - Agent failed on new cluster representative: Article extractor rejected by validation
  - Agent failed on new cluster representative: Article extractor rejected by validation
  - Agent failed on new cluster representative: Article extractor rejected by validation
  - Agent failed on new cluster representative: Article extractor rejected by validation
  - Agent failed on new cluster representative: Article extractor rejected by validation
  - Agent failed on new cluster representative: Article extractor rejected by validation
  - Agent failed on new cluster representative: Article extractor rejected by validation
  - Agent failed on new cluster representative: Article extractor rejected by validation
  - Agent failed on new cluster representative: Article extractor rejected by validation
  - ...and 27 more


## Run 2026-07-29 01:15:16

- **Run directory:** `orch_runs/run_20260729_010906`
- **Model:** gemma-4-31b-it
- **Pagination type:** infinite scroll
- **Scroll rounds requested:** 1
- **Input URLs:**
  - https://aawsat.com/%D8%A7%D9%84%D8%B1%D8%A3%D9%8A
- **Requirements:** title, date, author, article body text
- **Pages requested:** 1
- **Pages processed:** 0
- **Articles extracted:** 0
- **Articles failed:** 0
- **Clusters (unique structures):** 0
- **Links dropped (non-article filter):** 0
- **LLM calls:** 1
  - Links agent: 1
- **Total time:** 6m 9s
- **Errors (1):**
  - Links extraction failed: All 3 attempt(s) failed. Last error: AST SAFETY ERROR: Disallowed syntax: ImportFrom


## Run 2026-07-29 01:27:34

- **Run directory:** `orch_runs/run_20260729_012203`
- **Model:** gemma-4-31b-it
- **Pagination type:** infinite scroll
- **Scroll rounds requested:** 2
- **Input URLs:**
  - https://aawsat.com/%D8%A7%D9%84%D8%B1%D8%A3%D9%8A
- **Requirements:** title, date, author, article body text
- **Pages requested:** 1
- **Pages processed:** 0
- **Articles extracted:** 0
- **Articles failed:** 0
- **Clusters (unique structures):** 0
- **Links dropped (non-article filter):** 0
- **LLM calls:** 1
  - Links agent: 1
- **Total time:** 5m 31s
- **Errors (1):**
  - Links extraction failed: All 3 attempt(s) failed. Last error: AST SAFETY ERROR: Disallowed syntax: Lambda


## Run 2026-07-29 02:29:19

- **Run directory:** `orch_runs/run_20260729_021324`
- **Model:** gemma-4-31b-it
- **Pagination type:** infinite scroll
- **Scroll rounds requested:** 2
- **Input URLs:**
  - https://aawsat.com/%D8%A7%D9%84%D8%B1%D8%A3%D9%8A
- **Requirements:** title, date, author, article body text
- **Pages requested:** 1
- **Pages processed:** 1
- **Articles extracted:** 63
- **Articles failed:** 0
- **Clusters (unique structures):** 3
- **Links dropped (non-article filter):** 0
- **Article extractor validation:** 3 warning(s)
  - cluster 81726d321529 representative attempt 1: retry - `title` is N/A; listing title exists but extractor returned N/A: دورة الطبيعة
  - cluster 7007be81aa53 representative attempt 1: retry - `article body text` is N/A
  - cluster 341cd4b49fe8 representative attempt 1: retry - `body_text` is N/A
- **LLM calls:** 7
  - Article agent: 6
  - Links agent: 1
- **LLM calls per article:** 0.11
- **Code reuse rate:** 57/63 articles reused cluster code (90%)
- **Tokens:** 119,047 total (92,747 prompt + 2,475 output)
  - Article agent: 103,027 tokens
  - Links agent: 16,020 tokens
- **Estimated cost:** $0.0077 (at $0.075/$0.3 per 1M input/output tokens)
- **Estimated cost per 1,000 articles:** $0.1222
- **Structural-map depth:** 7 agent call(s), 1 shrunk below full depth 10 to fit the input-token budget
  - Links agent: depth 9 (~12,824 input tokens)  ⚠ reduced
  - Article agent: depth 13 (~12,849 input tokens)
  - Article agent: depth 13 (~13,437 input tokens)
  - Article agent: depth 11 (~12,296 input tokens)
  - Article agent: depth 11 (~12,552 input tokens)
  - Article agent: depth 14 (~14,278 input tokens)
  - Article agent: depth 14 (~14,504 input tokens)
- **Total time:** 15m 55s
- **Fetch method:** 62 via requests (98%), 1 via browser (2%)
- **Missing/N/A values (of 63 extracted):**
  - `title`: 0/63 N/A (0%)
  - `date`: 0/63 N/A (0%)
  - `author`: 0/63 N/A (0%)
  - `article body text`: 0/63 N/A (0%)
- **Errors:** none


## Run 2026-07-29 19:03:23

- **Run directory:** `orch_runs/run_20260729_184616`
- **Model:** gemma-4-31b-it
- **Pagination type:** infinite scroll
- **Scroll rounds requested:** 2
- **Input URLs:**
  - https://aawsat.com/%D8%A7%D9%84%D8%B1%D8%A3%D9%8A
- **Requirements:** title, date, author, article body text
- **Pages requested:** 1
- **Pages processed:** 1
- **Articles extracted:** 63
- **Articles failed:** 0
- **Clusters (unique structures):** 4
- **Links dropped (non-article filter):** 0
- **Article extractor validation:** 3 warning(s)
  - cluster 81726d321529 representative attempt 1: retry - `title` is N/A; listing title exists but extractor returned N/A: دورة الطبيعة
  - cluster 7007be81aa53 representative attempt 1: retry - `article body text` is N/A
  - cluster a04dad562962 representative attempt 1: retry - `body_text` is N/A
- **LLM calls:** 8
  - Article agent: 7
  - Links agent: 1
- **LLM calls per article:** 0.13
- **Code reuse rate:** 56/63 articles reused cluster code (89%)
- **Tokens:** 135,441 total (107,810 prompt + 2,948 output)
  - Article agent: 119,166 tokens
  - Links agent: 16,275 tokens
- **Estimated cost:** $0.0090 (at $0.075/$0.3 per 1M input/output tokens)
- **Estimated cost per 1,000 articles:** $0.1424
- **Structural-map depth:** 8 agent call(s), 1 shrunk below full depth 10 to fit the input-token budget
  - Links agent: depth 9 (~12,844 input tokens)  ⚠ reduced
  - Article agent: depth 13 (~12,858 input tokens)
  - Article agent: depth 13 (~13,448 input tokens)
  - Article agent: depth 11 (~12,292 input tokens)
  - Article agent: depth 11 (~12,537 input tokens)
  - Article agent: depth 12 (~14,726 input tokens)
  - Article agent: depth 12 (~14,987 input tokens)
  - Article agent: depth 14 (~14,110 input tokens)
- **Total time:** 17m 7s
- **Fetch method:** 62 via requests (98%), 1 via browser (2%)
- **Missing/N/A values (of 63 extracted):**
  - `title`: 0/63 N/A (0%)
  - `date`: 0/63 N/A (0%)
  - `author`: 1/63 N/A (2%)
  - `article body text`: 1/63 N/A (2%)
- **Errors:** none


## Run 2026-07-30 23:50:17

- **Run directory:** `orch_runs/run_20260730_234932`
- **Model:** gemma-3-27b-it
- **Pagination type:** numbered pagination
- **Input URLs:**
  - https://www.youm7.com/Section/%D8%A3%D8%AE%D8%A8%D8%A7%D8%B1-%D8%B9%D8%A7%D8%AC%D9%84%D8%A9/65/1
  - https://www.youm7.com/Section/%D8%A3%D8%AE%D8%A8%D8%A7%D8%B1-%D8%B9%D8%A7%D8%AC%D9%84%D8%A9/65/2
- **Requirements:** title, date, author, article body text
- **Pages requested:** 3
- **Pages processed:** 0
- **Articles extracted:** 0
- **Articles failed:** 0
- **Clusters (unique structures):** 0
- **Links dropped (non-article filter):** 0
- **Link coverage audit:** 1 page(s), 0 raw links, 0 accepted article links, 0 new unique links
  - ⚠ 1 page(s) flagged for review:
    - page 1: zero_accepted_links, zero_raw_links (0 accepted, 0 new)
  - Full audit saved to `orch_runs/run_20260730_234932/link_coverage.json`
- **LLM calls:** 1
  - Links agent: 1
- **Total time:** 0m 45s
- **Errors (1):**
  - Links extraction failed: All 3 attempt(s) failed. Last error: LLM ERROR: Error generating code: 404 models/gemma-3-27b-it is not found for API version v1beta, or is not supported for generateContent. 


## Run 2026-07-30 23:51:28

- **Run directory:** `orch_runs/run_20260730_235017`
- **Model:** gemma-3-27b-it
- **Pagination type:** numbered pagination
- **Input URLs:**
  - https://pchrgaza.org/ar/category/genocide-on-gaza-ar/testimonies-from-the-war-ar/
  - https://pchrgaza.org/ar/category/genocide-on-gaza-ar/testimonies-from-the-war-ar/page/2/
- **Requirements:** title, date, author, article body text
- **Pages requested:** 3
- **Pages processed:** 0
- **Articles extracted:** 0
- **Articles failed:** 0
- **Clusters (unique structures):** 0
- **Links dropped (non-article filter):** 0
- **Link coverage audit:** 1 page(s), 0 raw links, 0 accepted article links, 0 new unique links
  - ⚠ 1 page(s) flagged for review:
    - page 1: zero_accepted_links, zero_raw_links (0 accepted, 0 new)
  - Full audit saved to `orch_runs/run_20260730_235017/link_coverage.json`
- **LLM calls:** 1
  - Links agent: 1
- **Total time:** 1m 10s
- **Errors (1):**
  - Links extraction failed: All 3 attempt(s) failed. Last error: LLM ERROR: Error generating code: 404 models/gemma-3-27b-it is not found for API version v1beta, or is not supported for generateContent. 


## Run 2026-07-31 00:11:57

- **Run directory:** `orch_runs/run_20260731_000306`
- **Model:** gemma-4-31b-it
- **Pagination type:** numbered pagination
- **Input URLs:**
  - https://www.youm7.com/Section/%D8%A3%D8%AE%D8%A8%D8%A7%D8%B1-%D8%B9%D8%A7%D8%AC%D9%84%D8%A9/65/1
  - https://www.youm7.com/Section/%D8%A3%D8%AE%D8%A8%D8%A7%D8%B1-%D8%B9%D8%A7%D8%AC%D9%84%D8%A9/65/2
- **Requirements:** title, date, author, article body text
- **Pages requested:** 3
- **Pages processed:** 3
- **Articles extracted:** 146
- **Articles failed:** 0
- **Clusters (unique structures):** 4
- **Links dropped (non-article filter):** 0
- **Link coverage audit:** 3 page(s), 120 raw links, 174 accepted article links, 146 new unique links
  - Full audit saved to `orch_runs/run_20260731_000306/link_coverage.json`
- **Link recovery guard:** triggered on 1 page(s)
  - page 3: 40 generated accepted -> 94 after recovery (+54)
- **Link preflight:** passed - 10/10 article-like, 0/10 singleton structures
- **Article extractor validation:** 4 warning(s)
  - cluster 89a058fc046a representative attempt 1: accept - `author` is N/A
  - cluster ae0dd5dc208d representative attempt 1: accept - `author` is N/A
  - cluster f88af79e900c representative attempt 1: accept - `author` is N/A
  - cluster a54dca892f4c representative attempt 1: accept - `author` is N/A
- **LLM calls:** 5
  - Article agent: 4
  - Links agent: 1
- **LLM calls per article:** 0.03
- **Code reuse rate:** 142/146 articles reused cluster code (97%)
- **Tokens:** 61,624 total (51,368 prompt + 1,724 output)
  - Article agent: 57,323 tokens
  - Links agent: 4,301 tokens
- **Estimated cost:** $0.0044 (at $0.075/$0.3 per 1M input/output tokens)
- **Estimated cost per 1,000 articles:** $0.0299
- **Structural-map depth:** 5 agent call(s), 2 shrunk below full depth 10 to fit the input-token budget
  - Links agent: depth 12 (~3,370 input tokens)
  - Article agent: depth 16 (~9,849 input tokens)
  - Article agent: depth 6 (~12,304 input tokens)  ⚠ reduced
  - Article agent: depth 8 (~11,308 input tokens)  ⚠ reduced
  - Article agent: depth 10 (~14,532 input tokens)
- **Total time:** 8m 51s
- **Fetch method:** 146 via requests (100%), 0 via browser (0%)
- **Missing/N/A values (of 146 extracted):**
  - `title`: 0/146 N/A (0%)
  - `date`: 5/146 N/A (3%)
  - `author`: 15/146 N/A (10%)
  - `article body text`: 0/146 N/A (0%)
- **Errors:** none


## Run 2026-07-31 00:19:12

- **Run directory:** `orch_runs/run_20260731_001158`
- **Model:** gemma-4-31b-it
- **Pagination type:** numbered pagination
- **Input URLs:**
  - https://pchrgaza.org/ar/category/genocide-on-gaza-ar/testimonies-from-the-war-ar/
  - https://pchrgaza.org/ar/category/genocide-on-gaza-ar/testimonies-from-the-war-ar/page/2/
- **Requirements:** title, date, article body text
- **Pages requested:** 3
- **Pages processed:** 0
- **Articles extracted:** 0
- **Articles failed:** 0
- **Clusters (unique structures):** 0
- **Links dropped (non-article filter):** 0
- **Link coverage audit:** 1 page(s), 0 raw links, 0 accepted article links, 0 new unique links
  - ⚠ 1 page(s) flagged for review:
    - page 1: zero_accepted_links, zero_raw_links (0 accepted, 0 new)
  - Full audit saved to `orch_runs/run_20260731_001158/link_coverage.json`
- **LLM calls:** 1
  - Links agent: 1
- **Tokens:** 23,476 total (10,018 prompt + 1,480 output)
  - Links agent: 23,476 tokens
- **Estimated cost:** $0.0012 (at $0.075/$0.3 per 1M input/output tokens)
- **Structural-map depth:** 1 agent call(s), 0 shrunk below full depth 10 to fit the input-token budget
  - Links agent: depth 12 (~3,869 input tokens)
- **Total time:** 7m 14s
- **Errors (1):**
  - No article links found on page 1


## Run 2026-07-31 00:23:56

- **Run directory:** `orch_runs/run_20260731_001912`
- **Model:** gemma-4-31b-it
- **Pagination type:** numbered pagination
- **Input URLs:**
  - https://www.almasryalyoum.com/news/index?typeid=1&sectionid=10
  - https://www.almasryalyoum.com/news/index?typeid=1&sectionid=10&page=2
- **Requirements:** title, date, author, article body text
- **Pages requested:** 3
- **Pages processed:** 3
- **Articles extracted:** 30
- **Articles failed:** 0
- **Clusters (unique structures):** 2
- **Links dropped (non-article filter):** 0
- **Link coverage audit:** 3 page(s), 30 raw links, 30 accepted article links, 30 new unique links
  - Full audit saved to `orch_runs/run_20260731_001912/link_coverage.json`
- **Link preflight:** passed - 10/10 article-like, 0/10 singleton structures
- **Article extractor validation:** 2 warning(s)
  - cluster 41ffd7e75508 representative attempt 1: accept - `date` is N/A; `author` is N/A; `article body text` is N/A
  - cluster 41ffd7e75508 sample attempt 1: accept - `date` is N/A; `author` is N/A; `article body text` is N/A
- **LLM calls:** 3
  - Article agent: 2
  - Links agent: 1
- **LLM calls per article:** 0.10
- **Code reuse rate:** 28/30 articles reused cluster code (93%)
- **Tokens:** 26,128 total (19,921 prompt + 1,001 output)
  - Article agent: 20,683 tokens
  - Links agent: 5,445 tokens
- **Estimated cost:** $0.0018 (at $0.075/$0.3 per 1M input/output tokens)
- **Estimated cost per 1,000 articles:** $0.0598
- **Structural-map depth:** 3 agent call(s), 0 shrunk below full depth 10 to fit the input-token budget
  - Links agent: depth 12 (~3,695 input tokens)
  - Article agent: depth 16 (~9,894 input tokens)
  - Article agent: depth 16 (~6,329 input tokens)
- **Total time:** 4m 44s
- **Fetch method:** 30 via requests (100%), 0 via browser (0%)
- **Missing/N/A values (of 30 extracted):**
  - `title`: 0/30 N/A (0%)
  - `date`: 9/30 N/A (30%)
  - `author`: 9/30 N/A (30%)
  - `article body text`: 9/30 N/A (30%)
- **Errors:** none


## Run 2026-07-31 00:28:04

- **Run directory:** `orch_runs/run_20260731_002357`
- **Model:** gemma-4-31b-it
- **Pagination type:** numbered pagination
- **Input URLs:**
  - https://euromedmonitor.org/ar/category/26/%D8%A7%D9%84%D9%86%D8%B2%D8%A7%D8%B9%D8%A7%D8%AA-%D8%A7%D9%84%D9%85%D8%B3%D9%84%D8%AD%D8%A9
  - https://euromedmonitor.org/ar/category/26/%D8%A7%D9%84%D9%86%D8%B2%D8%A7%D8%B9%D8%A7%D8%AA-%D8%A7%D9%84%D9%85%D8%B3%D9%84%D8%AD%D8%A9?page=2
- **Requirements:** title, date, article body text
- **Pages requested:** 3
- **Pages processed:** 3
- **Articles extracted:** 27
- **Articles failed:** 0
- **Clusters (unique structures):** 1
- **Links dropped (non-article filter):** 0
- **Link coverage audit:** 3 page(s), 27 raw links, 27 accepted article links, 27 new unique links
  - Full audit saved to `orch_runs/run_20260731_002357/link_coverage.json`
- **Link preflight:** passed - 0/9 article-like, 0/9 singleton structures
- **LLM calls:** 2
  - Article agent: 1
  - Links agent: 1
- **LLM calls per article:** 0.07
- **Code reuse rate:** 26/27 articles reused cluster code (96%)
- **Tokens:** 24,797 total (17,971 prompt + 1,083 output)
  - Article agent: 14,400 tokens
  - Links agent: 10,397 tokens
- **Estimated cost:** $0.0017 (at $0.075/$0.3 per 1M input/output tokens)
- **Estimated cost per 1,000 articles:** $0.0620
- **Structural-map depth:** 2 agent call(s), 0 shrunk below full depth 10 to fit the input-token budget
  - Links agent: depth 12 (~2,549 input tokens)
  - Article agent: depth 15 (~13,597 input tokens)
- **Total time:** 4m 7s
- **Fetch method:** 27 via requests (100%), 0 via browser (0%)
- **Missing/N/A values (of 27 extracted):**
  - `title`: 0/27 N/A (0%)
  - `date`: 0/27 N/A (0%)
  - `article body text`: 0/27 N/A (0%)
- **Errors:** none


## Run 2026-07-31 00:34:08

- **Run directory:** `orch_runs/run_20260731_002805`
- **Model:** gemma-4-31b-it
- **Pagination type:** infinite scroll
- **Scroll rounds requested:** 2
- **Input URLs:**
  - https://aawsat.com/%D8%A7%D9%84%D8%B1%D8%A3%D9%8A
- **Requirements:** title, date, author, article body text
- **Pages requested:** 1
- **Pages processed:** 1
- **Articles extracted:** 48
- **Articles failed:** 0
- **Clusters (unique structures):** 2
- **Links dropped (non-article filter):** 0
- **Link coverage audit:** 1 page(s), 48 raw links, 48 accepted article links, 48 new unique links
  - Full audit saved to `orch_runs/run_20260731_002805/link_coverage.json`
- **Link preflight:** passed - 10/10 article-like, 0/10 singleton structures
- **Article extractor validation:** 3 warning(s)
  - cluster 81726d321529 representative attempt 1: accept - `title` is N/A; listing title exists but extractor returned N/A: الخوف من الجواب
  - cluster 81726d321529 sample attempt 1: accept - `title` is N/A; listing title exists but extractor returned N/A: خيوط الحرب الخفية في السودان
  - cluster 99b6ba5fabde representative attempt 1: accept - `title` is N/A; listing title exists but extractor returned N/A: بلقيس وأروى وسيادة المرأة
- **LLM calls:** 3
  - Article agent: 2
  - Links agent: 1
- **LLM calls per article:** 0.06
- **Code reuse rate:** 46/48 articles reused cluster code (96%)
- **Tokens:** 48,167 total (38,286 prompt + 1,064 output)
  - Article agent: 34,276 tokens
  - Links agent: 13,891 tokens
- **Estimated cost:** $0.0032 (at $0.075/$0.3 per 1M input/output tokens)
- **Estimated cost per 1,000 articles:** $0.0665
- **Structural-map depth:** 3 agent call(s), 0 shrunk below full depth 10 to fit the input-token budget
  - Links agent: depth 11 (~11,083 input tokens)
  - Article agent: depth 13 (~12,872 input tokens)
  - Article agent: depth 13 (~14,328 input tokens)
- **Total time:** 6m 3s
- **Fetch method:** 47 via requests (98%), 1 via browser (2%)
- **Missing/N/A values (of 48 extracted):**
  - `title`: 48/48 N/A (100%)
  - `date`: 0/48 N/A (0%)
  - `author`: 0/48 N/A (0%)
  - `article body text`: 0/48 N/A (0%)
- **Errors:** none


## Run 2026-07-31 00:40:19

- **Run directory:** `orch_runs/run_20260731_003409`
- **Model:** gemma-4-31b-it
- **Pagination type:** infinite scroll
- **Scroll rounds requested:** 2
- **Input URLs:**
  - https://www.independentarabia.com/%D8%AB%D9%82%D8%A7%D9%81%D8%A9/%D8%B3%D9%8A%D9%86%D9%85%D8%A7
- **Requirements:** title, date, author, article body text
- **Pages requested:** 1
- **Pages processed:** 1
- **Articles extracted:** 62
- **Articles failed:** 0
- **Clusters (unique structures):** 2
- **Links dropped (non-article filter):** 0
- **Link coverage audit:** 1 page(s), 4 raw links, 62 accepted article links, 62 new unique links
  - Full audit saved to `orch_runs/run_20260731_003409/link_coverage.json`
- **Link recovery guard:** triggered on 1 page(s)
  - page 1: 4 generated accepted -> 62 after recovery (+58)
- **Link preflight:** passed - 10/10 article-like, 0/10 singleton structures
- **LLM calls:** 3
  - Article agent: 2
  - Links agent: 1
- **LLM calls per article:** 0.05
- **Code reuse rate:** 60/62 articles reused cluster code (97%)
- **Tokens:** 51,867 total (41,413 prompt + 1,330 output)
  - Article agent: 35,870 tokens
  - Links agent: 15,997 tokens
- **Estimated cost:** $0.0035 (at $0.075/$0.3 per 1M input/output tokens)
- **Estimated cost per 1,000 articles:** $0.0565
- **Structural-map depth:** 3 agent call(s), 1 shrunk below full depth 10 to fit the input-token budget
  - Links agent: depth 6 (~13,872 input tokens)  ⚠ reduced
  - Article agent: depth 16 (~13,990 input tokens)
  - Article agent: depth 16 (~13,548 input tokens)
- **Total time:** 6m 11s
- **Fetch method:** 62 via requests (100%), 0 via browser (0%)
- **Missing/N/A values (of 62 extracted):**
  - `title`: 0/62 N/A (0%)
  - `date`: 0/62 N/A (0%)
  - `author`: 0/62 N/A (0%)
  - `article body text`: 0/62 N/A (0%)
- **Errors:** none


## Run 2026-07-31 00:47:11

- **Run directory:** `orch_runs/run_20260731_004020`
- **Model:** gemma-4-31b-it
- **Pagination type:** load more button
- **Load-more clicks requested:** 2
- **Load-more selector:** auto-detect
- **Input URLs:**
  - https://www.arageek.com/tech
- **Requirements:** title, date, author, article body text
- **Pages requested:** 1
- **Pages processed:** 1
- **Articles extracted:** 80
- **Articles failed:** 0
- **Clusters (unique structures):** 3
- **Links dropped (non-article filter):** 0
- **Link coverage audit:** 1 page(s), 80 raw links, 80 accepted article links, 80 new unique links
  - Full audit saved to `orch_runs/run_20260731_004020/link_coverage.json`
- **Link preflight:** passed - 10/10 article-like, 1/10 singleton structures
- **Article extractor validation:** 2 warning(s)
  - cluster 4ed95a6f062f representative attempt 1: accept - `date` is N/A; `author` is N/A
  - cluster a6723d8a48b7 representative attempt 1: accept - `author` is N/A
- **LLM calls:** 4
  - Article agent: 3
  - Links agent: 1
- **LLM calls per article:** 0.05
- **Code reuse rate:** 77/80 articles reused cluster code (96%)
- **Tokens:** 51,778 total (40,031 prompt + 1,467 output)
  - Article agent: 33,091 tokens
  - Links agent: 18,687 tokens
- **Estimated cost:** $0.0034 (at $0.075/$0.3 per 1M input/output tokens)
- **Estimated cost per 1,000 articles:** $0.0430
- **Structural-map depth:** 4 agent call(s), 1 shrunk below full depth 10 to fit the input-token budget
  - Links agent: depth 6 (~14,291 input tokens)  ⚠ reduced
  - Article agent: depth 16 (~10,433 input tokens)
  - Article agent: depth 16 (~2,786 input tokens)
  - Article agent: depth 16 (~12,517 input tokens)
- **Total time:** 6m 51s
- **Fetch method:** 80 via requests (100%), 0 via browser (0%)
- **Missing/N/A values (of 80 extracted):**
  - `title`: 0/80 N/A (0%)
  - `date`: 8/80 N/A (10%)
  - `author`: 24/80 N/A (30%)
  - `article body text`: 0/80 N/A (0%)
- **Errors:** none


## Run 2026-07-31 00:49:27

- **Run directory:** `orch_runs/run_20260731_004711`
- **Model:** gemma-4-31b-it
- **Pagination type:** load more button
- **Load-more clicks requested:** 2
- **Load-more selector:** auto-detect
- **Input URLs:**
  - https://www.majalla.com/sections/%D8%B3%D9%8A%D8%A7%D8%B3%D8%A9
- **Requirements:** title, date, author, article body text
- **Pages requested:** 1
- **Pages processed:** 1
- **Articles extracted:** 33
- **Articles failed:** 0
- **Clusters (unique structures):** 1
- **Links dropped (non-article filter):** 0
- **Link coverage audit:** 1 page(s), 33 raw links, 33 accepted article links, 33 new unique links
  - Full audit saved to `orch_runs/run_20260731_004711/link_coverage.json`
- **Link preflight:** passed - 0/10 article-like, 0/10 singleton structures
- **LLM calls:** 2
  - Article agent: 1
  - Links agent: 1
- **LLM calls per article:** 0.06
- **Code reuse rate:** 32/33 articles reused cluster code (97%)
- **Tokens:** 25,985 total (22,675 prompt + 742 output)
  - Links agent: 15,430 tokens
  - Article agent: 10,555 tokens
- **Estimated cost:** $0.0019 (at $0.075/$0.3 per 1M input/output tokens)
- **Estimated cost per 1,000 articles:** $0.0583
- **Structural-map depth:** 2 agent call(s), 0 shrunk below full depth 10 to fit the input-token budget
  - Links agent: depth 11 (~13,675 input tokens)
  - Article agent: depth 16 (~8,998 input tokens)
- **Total time:** 2m 16s
- **Fetch method:** 33 via requests (100%), 0 via browser (0%)
- **Missing/N/A values (of 33 extracted):**
  - `title`: 0/33 N/A (0%)
  - `date`: 0/33 N/A (0%)
  - `author`: 0/33 N/A (0%)
  - `article body text`: 0/33 N/A (0%)
- **Errors:** none


## Run 2026-07-31 01:10:58

- **Run directory:** `orch_runs/run_20260731_004927`
- **Model:** gemma-4-31b-it
- **Pagination type:** numbered pagination
- **Input URLs:**
  - https://www.theguardian.com/world/gaza
  - https://www.theguardian.com/world/gaza?page=2
- **Requirements:** title, date, author, article body text
- **Pages requested:** 3
- **Pages processed:** 3
- **Articles extracted:** 60
- **Articles failed:** 0
- **Clusters (unique structures):** 17
- **Links dropped (non-article filter):** 0
- **Link coverage audit:** 3 page(s), 24 raw links, 60 accepted article links, 60 new unique links
  - ⚠ 1 page(s) flagged for review:
    - page 2: zero_raw_links (20 accepted, 20 new)
  - Full audit saved to `orch_runs/run_20260731_004927/link_coverage.json`
- **Link recovery guard:** triggered on 2 page(s)
  - page 2: 0 generated accepted -> 20 after recovery (+20)
  - page 3: 4 generated accepted -> 20 after recovery (+16)
- **Link preflight:** passed - 10/10 article-like, 4/10 singleton structures
- **Article extractor validation:** 16 warning(s)
  - cluster b042292a5c27 representative attempt 1: accept - `date` is N/A; `author` is N/A
  - cluster 5a6a1650ac31 representative attempt 1: accept - `date` is N/A
  - cluster ee1c6d710597 representative attempt 1: accept - `date` is N/A; `author` is N/A; `article body text` is N/A
  - cluster ee1c6d710597 sample attempt 1: accept - `date` is N/A; `author` is N/A; `article body text` is N/A
  - cluster 884e227259d3 representative attempt 1: accept - `date` is N/A; `author` is N/A
  - cluster d79b3e86475d representative attempt 1: accept - `date` is N/A; `author` is N/A
  - cluster f3146a3da920 representative attempt 1: accept - `date` is N/A; `author` is N/A
  - cluster b64603ef2895 representative attempt 1: accept - `date` is N/A
  - cluster 5b76c43c1536 representative attempt 1: accept - `date` is N/A
  - cluster ba9ec57003b0 representative attempt 1: accept - `date` is N/A; `author` is N/A
  - ...and 6 more
- **LLM calls:** 18
  - Article agent: 17
  - Links agent: 1
- **LLM calls per article:** 0.30
- **Code reuse rate:** 43/60 articles reused cluster code (72%)
- **Tokens:** 188,000 total (149,601 prompt + 5,772 output)
  - Article agent: 171,624 tokens
  - Links agent: 16,376 tokens
- **Estimated cost:** $0.0130 (at $0.075/$0.3 per 1M input/output tokens)
- **Estimated cost per 1,000 articles:** $0.2159
- **Structural-map depth:** 18 agent call(s), 1 shrunk below full depth 10 to fit the input-token budget
  - Links agent: depth 9 (~14,243 input tokens)  ⚠ reduced
  - Article agent: depth 16 (~9,034 input tokens)
  - Article agent: depth 16 (~8,833 input tokens)
  - Article agent: depth 16 (~5,105 input tokens)
  - Article agent: depth 16 (~7,449 input tokens)
  - Article agent: depth 16 (~6,207 input tokens)
  - Article agent: depth 16 (~7,919 input tokens)
  - Article agent: depth 16 (~7,241 input tokens)
  - Article agent: depth 16 (~6,993 input tokens)
  - Article agent: depth 16 (~9,397 input tokens)
  - Article agent: depth 16 (~8,265 input tokens)
  - Article agent: depth 14 (~14,001 input tokens)
  - Article agent: depth 16 (~9,324 input tokens)
  - Article agent: depth 16 (~6,314 input tokens)
  - Article agent: depth 16 (~7,780 input tokens)
  - Article agent: depth 16 (~8,280 input tokens)
  - Article agent: depth 16 (~7,753 input tokens)
  - Article agent: depth 16 (~5,445 input tokens)
- **Total time:** 21m 30s
- **Fetch method:** 60 via requests (100%), 0 via browser (0%)
- **Missing/N/A values (of 60 extracted):**
  - `title`: 0/60 N/A (0%)
  - `date`: 58/60 N/A (97%)
  - `author`: 46/60 N/A (77%)
  - `article body text`: 4/60 N/A (7%)
- **Errors:** none


## Run 2026-07-31 01:28:41

- **Run directory:** `orch_runs/run_20260731_011058`
- **Model:** gemma-4-31b-it
- **Pagination type:** numbered pagination
- **Input URLs:**
  - https://www.btselem.org/ota/100/all
  - https://www.btselem.org/ota/100/all?page=1
- **Requirements:** title, date, article body text
- **Pages requested:** 3
- **Pages processed:** 3
- **Articles extracted:** 36
- **Articles failed:** 0
- **Clusters (unique structures):** 6
- **Links dropped (non-article filter):** 0
- **Link coverage audit:** 3 page(s), 36 raw links, 36 accepted article links, 36 new unique links
  - Full audit saved to `orch_runs/run_20260731_011058/link_coverage.json`
- **Link preflight:** passed - 0/10 article-like, 4/10 singleton structures
- **Article extractor validation:** 1 warning(s)
  - cluster ea43c63c7e65 representative attempt 1: accept - `title` is N/A; `date` is N/A; `article body text` is N/A
- **LLM calls:** 7
  - Article agent: 6
  - Links agent: 1
- **LLM calls per article:** 0.19
- **Code reuse rate:** 30/36 articles reused cluster code (83%)
- **Tokens:** 77,141 total (66,499 prompt + 2,045 output)
  - Article agent: 69,697 tokens
  - Links agent: 7,444 tokens
- **Estimated cost:** $0.0056 (at $0.075/$0.3 per 1M input/output tokens)
- **Estimated cost per 1,000 articles:** $0.1556
- **Structural-map depth:** 7 agent call(s), 0 shrunk below full depth 10 to fit the input-token budget
  - Links agent: depth 12 (~4,917 input tokens)
  - Article agent: depth 16 (~12,108 input tokens)
  - Article agent: depth 15 (~14,767 input tokens)
  - Article agent: depth 16 (~1,115 input tokens)
  - Article agent: depth 16 (~10,538 input tokens)
  - Article agent: depth 16 (~11,658 input tokens)
  - Article agent: depth 16 (~11,389 input tokens)
- **Total time:** 17m 43s
- **Fetch method:** 1 via requests (3%), 35 via browser (97%)
- **Missing/N/A values (of 36 extracted):**
  - `title`: 1/36 N/A (3%)
  - `date`: 1/36 N/A (3%)
  - `article body text`: 3/36 N/A (8%)
- **Errors:** none


## Run 2026-07-31 01:33:05

- **Run directory:** `orch_runs/run_20260731_012841`
- **Model:** gemma-4-31b-it
- **Pagination type:** infinite scroll
- **Scroll rounds requested:** 2
- **Input URLs:**
  - https://arabic.cnn.com/tag/gaza_strip
- **Requirements:** title, date, article body text
- **Pages requested:** 1
- **Pages processed:** 1
- **Articles extracted:** 35
- **Articles failed:** 0
- **Clusters (unique structures):** 3
- **Links dropped (non-article filter):** 0
- **Link coverage audit:** 1 page(s), 0 raw links, 35 accepted article links, 35 new unique links
  - ⚠ 1 page(s) flagged for review:
    - page 1: zero_raw_links (35 accepted, 35 new)
  - Full audit saved to `orch_runs/run_20260731_012841/link_coverage.json`
- **Link recovery guard:** triggered on 1 page(s)
  - page 1: 0 generated accepted -> 35 after recovery (+35)
- **Link preflight:** passed - 10/10 article-like, 1/10 singleton structures
- **LLM calls:** 4
  - Article agent: 3
  - Links agent: 1
- **LLM calls per article:** 0.11
- **Code reuse rate:** 32/35 articles reused cluster code (91%)
- **Tokens:** 44,204 total (37,023 prompt + 1,965 output)
  - Links agent: 23,048 tokens
  - Article agent: 21,156 tokens
- **Estimated cost:** $0.0034 (at $0.075/$0.3 per 1M input/output tokens)
- **Estimated cost per 1,000 articles:** $0.0962
- **Structural-map depth:** 4 agent call(s), 0 shrunk below full depth 10 to fit the input-token budget
  - Links agent: depth 12 (~6,758 input tokens)
  - Article agent: depth 16 (~7,265 input tokens)
  - Article agent: depth 16 (~4,668 input tokens)
  - Article agent: depth 16 (~6,440 input tokens)
- **Total time:** 4m 24s
- **Fetch method:** 35 via requests (100%), 0 via browser (0%)
- **Missing/N/A values (of 35 extracted):**
  - `title`: 0/35 N/A (0%)
  - `date`: 0/35 N/A (0%)
  - `article body text`: 0/35 N/A (0%)
- **Errors:** none


## Run 2026-07-31 01:37:26

- **Run directory:** `orch_runs/run_20260731_013306`
- **Model:** gemma-4-31b-it
- **Pagination type:** infinite scroll
- **Scroll rounds requested:** 2
- **Input URLs:**
  - https://www.aljadeedmagazine.com/%D9%85%D9%82%D8%A7%D9%84%D8%A7%D8%AA
- **Requirements:** title, date, author, article body text
- **Pages requested:** 1
- **Pages processed:** 1
- **Articles extracted:** 64
- **Articles failed:** 0
- **Clusters (unique structures):** 1
- **Links dropped (non-article filter):** 0
- **Link coverage audit:** 1 page(s), 64 raw links, 64 accepted article links, 64 new unique links
  - Full audit saved to `orch_runs/run_20260731_013306/link_coverage.json`
- **Link preflight:** passed - 10/10 article-like, 0/10 singleton structures
- **Article extractor validation:** 2 warning(s)
  - cluster 77647a512b9c representative attempt 1: accept - `author` is N/A
  - cluster 77647a512b9c sample attempt 1: accept - `author` is N/A; `article body text` is N/A
- **LLM calls:** 2
  - Article agent: 1
  - Links agent: 1
- **LLM calls per article:** 0.03
- **Code reuse rate:** 63/64 articles reused cluster code (98%)
- **Tokens:** 26,451 total (19,599 prompt + 1,014 output)
  - Links agent: 14,642 tokens
  - Article agent: 11,809 tokens
- **Estimated cost:** $0.0018 (at $0.075/$0.3 per 1M input/output tokens)
- **Estimated cost per 1,000 articles:** $0.0277
- **Structural-map depth:** 2 agent call(s), 1 shrunk below full depth 10 to fit the input-token budget
  - Links agent: depth 4 (~12,129 input tokens)  ⚠ reduced
  - Article agent: depth 16 (~7,468 input tokens)
- **Total time:** 4m 20s
- **Fetch method:** 64 via requests (100%), 0 via browser (0%)
- **Missing/N/A values (of 64 extracted):**
  - `title`: 28/64 N/A (44%)
  - `date`: 28/64 N/A (44%)
  - `author`: 64/64 N/A (100%)
  - `article body text`: 43/64 N/A (67%)
- **Errors:** none


## Run 2026-07-31 01:39:10

- **Run directory:** `orch_runs/run_20260731_013726`
- **Model:** gemma-4-31b-it
- **Pagination type:** load more button
- **Load-more clicks requested:** 2
- **Load-more selector:** auto-detect
- **Input URLs:**
  - https://alsifr.org/kam-kaif
- **Requirements:** title, date, author, article body text
- **Pages requested:** 1
- **Pages processed:** 1
- **Articles extracted:** 36
- **Articles failed:** 0
- **Clusters (unique structures):** 1
- **Links dropped (non-article filter):** 0
- **Link coverage audit:** 1 page(s), 36 raw links, 36 accepted article links, 36 new unique links
  - Full audit saved to `orch_runs/run_20260731_013726/link_coverage.json`
- **Link preflight:** passed - 10/10 article-like, 0/10 singleton structures
- **LLM calls:** 2
  - Article agent: 1
  - Links agent: 1
- **LLM calls per article:** 0.06
- **Code reuse rate:** 35/36 articles reused cluster code (97%)
- **Tokens:** 18,609 total (16,307 prompt + 691 output)
  - Article agent: 10,530 tokens
  - Links agent: 8,079 tokens
- **Estimated cost:** $0.0014 (at $0.075/$0.3 per 1M input/output tokens)
- **Estimated cost per 1,000 articles:** $0.0397
- **Structural-map depth:** 2 agent call(s), 0 shrunk below full depth 10 to fit the input-token budget
  - Links agent: depth 12 (~6,552 input tokens)
  - Article agent: depth 16 (~9,753 input tokens)
- **Total time:** 1m 43s
- **Fetch method:** 36 via requests (100%), 0 via browser (0%)
- **Missing/N/A values (of 36 extracted):**
  - `title`: 0/36 N/A (0%)
  - `date`: 0/36 N/A (0%)
  - `author`: 0/36 N/A (0%)
  - `article body text`: 0/36 N/A (0%)
- **Errors:** none


## Run 2026-07-31 01:41:46

- **Run directory:** `orch_runs/run_20260731_013910`
- **Model:** gemma-4-31b-it
- **Pagination type:** numbered pagination
- **Input URLs:**
  - https://www.ida2at.com/category/art-literature/
  - https://www.ida2at.com/category/art-literature/page/2/
- **Requirements:** title, date, author, article body text
- **Pages requested:** 3
- **Pages processed:** 3
- **Articles extracted:** 24
- **Articles failed:** 0
- **Clusters (unique structures):** 1
- **Links dropped (non-article filter):** 0
- **Link coverage audit:** 3 page(s), 24 raw links, 24 accepted article links, 24 new unique links
  - Full audit saved to `orch_runs/run_20260731_013910/link_coverage.json`
- **Link preflight:** passed - 10/10 article-like, 0/10 singleton structures
- **LLM calls:** 2
  - Article agent: 1
  - Links agent: 1
- **LLM calls per article:** 0.08
- **Code reuse rate:** 23/24 articles reused cluster code (96%)
- **Tokens:** 19,682 total (16,247 prompt + 872 output)
  - Links agent: 12,650 tokens
  - Article agent: 7,032 tokens
- **Estimated cost:** $0.0015 (at $0.075/$0.3 per 1M input/output tokens)
- **Estimated cost per 1,000 articles:** $0.0617
- **Structural-map depth:** 2 agent call(s), 0 shrunk below full depth 10 to fit the input-token budget
  - Links agent: depth 12 (~10,236 input tokens)
  - Article agent: depth 16 (~6,009 input tokens)
- **Total time:** 2m 36s
- **Fetch method:** 24 via requests (100%), 0 via browser (0%)
- **Missing/N/A values (of 24 extracted):**
  - `title`: 0/24 N/A (0%)
  - `date`: 0/24 N/A (0%)
  - `author`: 0/24 N/A (0%)
  - `article body text`: 0/24 N/A (0%)
- **Errors:** none


## Run 2026-07-31 01:44:41

- **Run directory:** `orch_runs/run_20260731_014147`
- **Model:** gemma-4-31b-it
- **Pagination type:** load more button
- **Load-more clicks requested:** 2
- **Load-more selector:** auto-detect
- **Input URLs:**
  - https://lakome2.com/category/art/
- **Requirements:** title, date, article body text
- **Pages requested:** 1
- **Pages processed:** 1
- **Articles extracted:** 54
- **Articles failed:** 0
- **Clusters (unique structures):** 1
- **Links dropped (non-article filter):** 0
- **Link coverage audit:** 1 page(s), 54 raw links, 54 accepted article links, 54 new unique links
  - Full audit saved to `orch_runs/run_20260731_014147/link_coverage.json`
- **Link preflight:** passed - 10/10 article-like, 0/10 singleton structures
- **Article extractor validation:** 2 warning(s)
  - cluster 83f4cf73149a representative attempt 1: accept - `date` is N/A; `article body text` is N/A
  - cluster 83f4cf73149a sample attempt 1: accept - `date` is N/A; `article body text` is N/A
- **LLM calls:** 2
  - Article agent: 1
  - Links agent: 1
- **LLM calls per article:** 0.04
- **Code reuse rate:** 53/54 articles reused cluster code (98%)
- **Tokens:** 26,816 total (23,102 prompt + 755 output)
  - Links agent: 16,358 tokens
  - Article agent: 10,458 tokens
- **Estimated cost:** $0.0020 (at $0.075/$0.3 per 1M input/output tokens)
- **Estimated cost per 1,000 articles:** $0.0363
- **Structural-map depth:** 2 agent call(s), 1 shrunk below full depth 10 to fit the input-token budget
  - Links agent: depth 9 (~14,091 input tokens)  ⚠ reduced
  - Article agent: depth 16 (~9,009 input tokens)
- **Total time:** 2m 54s
- **Fetch method:** 54 via requests (100%), 0 via browser (0%)
- **Missing/N/A values (of 54 extracted):**
  - `title`: 0/54 N/A (0%)
  - `date`: 54/54 N/A (100%)
  - `article body text`: 50/54 N/A (93%)
- **Errors:** none


## Run 2026-07-31 01:47:58

- **Run directory:** `orch_runs/run_20260731_014441`
- **Model:** gemma-4-31b-it
- **Pagination type:** numbered pagination
- **Input URLs:**
  - https://mana.net/category/articles/
  - https://mana.net/category/articles/page/2/
- **Requirements:** title, date, article body text
- **Pages requested:** 3
- **Pages processed:** 3
- **Articles extracted:** 34
- **Articles failed:** 0
- **Clusters (unique structures):** 2
- **Links dropped (non-article filter):** 0
- **Link coverage audit:** 3 page(s), 42 raw links, 42 accepted article links, 34 new unique links
  - Full audit saved to `orch_runs/run_20260731_014441/link_coverage.json`
- **Link preflight:** passed - 10/10 article-like, 0/10 singleton structures
- **LLM calls:** 3
  - Article agent: 2
  - Links agent: 1
- **LLM calls per article:** 0.09
- **Code reuse rate:** 32/34 articles reused cluster code (94%)
- **Tokens:** 41,468 total (38,413 prompt + 959 output)
  - Article agent: 28,538 tokens
  - Links agent: 12,930 tokens
- **Estimated cost:** $0.0032 (at $0.075/$0.3 per 1M input/output tokens)
- **Estimated cost per 1,000 articles:** $0.0932
- **Structural-map depth:** 3 agent call(s), 0 shrunk below full depth 10 to fit the input-token budget
  - Links agent: depth 12 (~11,481 input tokens)
  - Article agent: depth 12 (~13,524 input tokens)
  - Article agent: depth 12 (~13,405 input tokens)
- **Total time:** 3m 17s
- **Fetch method:** 34 via requests (100%), 0 via browser (0%)
- **Missing/N/A values (of 34 extracted):**
  - `title`: 0/34 N/A (0%)
  - `date`: 0/34 N/A (0%)
  - `article body text`: 0/34 N/A (0%)
- **Errors:** none


## Run 2026-07-31 01:59:59

- **Run directory:** `orch_runs/run_20260731_014759`
- **Model:** gemma-4-31b-it
- **Pagination type:** numbered pagination
- **Input URLs:**
  - https://acpss.ahram.org.eg/OuterWriter/28/%D9%85%D9%82%D8%A7%D9%84%D8%A7%D8%AA/0.aspx
  - https://acpss.ahram.org.eg/OuterWriter/28/%D9%85%D9%82%D8%A7%D9%84%D8%A7%D8%AA/30.aspx
- **Requirements:** title, date, author, article body text
- **Pages requested:** 3
- **Pages processed:** 3
- **Articles extracted:** 70
- **Articles failed:** 0
- **Clusters (unique structures):** 1
- **Links dropped (non-article filter):** 0
- **Link coverage audit:** 3 page(s), 90 raw links, 90 accepted article links, 70 new unique links
  - Full audit saved to `orch_runs/run_20260731_014759/link_coverage.json`
- **Link preflight:** passed - 0/10 article-like, 0/10 singleton structures
- **Article extractor validation:** 2 warning(s)
  - cluster 662db8628ddb representative attempt 1: accept - `title` is N/A; `date` is N/A; `author` is N/A
  - cluster 662db8628ddb sample attempt 1: accept - `title` is N/A; `date` is N/A; `author` is N/A
- **LLM calls:** 2
  - Article agent: 1
  - Links agent: 1
- **LLM calls per article:** 0.03
- **Code reuse rate:** 69/70 articles reused cluster code (99%)
- **Tokens:** 8,787 total (4,774 prompt + 614 output)
  - Links agent: 4,559 tokens
  - Article agent: 4,228 tokens
- **Estimated cost:** $0.0005 (at $0.075/$0.3 per 1M input/output tokens)
- **Estimated cost per 1,000 articles:** $0.0077
- **Structural-map depth:** 2 agent call(s), 0 shrunk below full depth 10 to fit the input-token budget
  - Links agent: depth 12 (~2,868 input tokens)
  - Article agent: depth 16 (~1,904 input tokens)
- **Total time:** 12m 0s
- **Fetch method:** 0 via requests (0%), 70 via browser (100%)
- **Missing/N/A values (of 70 extracted):**
  - `title`: 70/70 N/A (100%)
  - `date`: 70/70 N/A (100%)
  - `author`: 70/70 N/A (100%)
  - `article body text`: 70/70 N/A (100%)
- **Errors:** none

