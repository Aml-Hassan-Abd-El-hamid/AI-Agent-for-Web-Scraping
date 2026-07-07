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

