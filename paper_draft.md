
**Related work:**
- ```
  **LLM-based extraction-code synthesis.** Among the closest prior systems is EVAPORATE [Arora et al., VLDB 2024], which uses an LLM to build structured tables from collections of documents. Its strongest variant, EVAPORATE-CODE+, does not call the LLM on every document. Instead, it writes reusable extraction _functions_ from a small sample and runs them over the whole collection, so cost grows with the number of target fields rather than the number of documents.

We build on the same idea—generate extraction code once, then reuse it—but differ in three ways:

- **How we decide what to reuse.** EVAPORATE groups documents by text similarity (TF-IDF + K-means) and reuses code per field. We group pages by their _HTML structure_ (an exact signature over the DOM skeleton) and write one program per template. Code is reused only when two pages share the same layout, and our LLM cost grows with the number of _distinct structures_.
    
- **How extraction programs are validated.** EVAPORATE has no labels, so it generates many candidate functions per field and combines them statistically with weak supervision. We instead _run_ each generated program, check its output, and ask the LLM to fix it if it fails. We verify against real page output instead of averaging over unverified functions.
    
- **Problem scope.** EVAPORATE works on static, already-downloaded corpora (HTML files, PDFs, emails) and does not handle navigation. Our system runs on the _live_ web: it discovers pagination, expands dynamically loaded content, gets past bot defenses, and extracts full article bodies (not just short field values) across multilingual, right-to-left sites at thousands of pages per site.

Consequently, while EVAPORATE addresses scalable information extraction from heterogeneous document collections, our work addresses autonomous, end-to-end extraction from large, live websites.

**EVAPORATE solves:**

Document collection
↓
Extract structured fields

**Our system solves:**

Website
↓
Discover pages
↓
Navigate
↓
Extract structured fields
```
  this section talk about this paper [[Language Models Enable Simple Systems for Generating Structured Views of Heterogeneous Data Lakes]]
- ```
  Executable web agents for extraction at scale. Among the closest recent systems is BardeenAgent, evaluated on the WebLists benchmark [Bohra et al., 2025]. Like our system, it converts an LLM agent's successful execution into a reusable program that can be replayed across similar webpages, building a generalizable CSS selector to capture items and cutting the cost per extracted row. We share the central idea—exploit the regular structure of HTML so the LLM is used once and the resulting code is reused at scale—but differ in four ways:

- What we extract: BardeenAgent targets lists of short structured records (enterprise-style datasets with fixed schemas). We target full news articles, including long-form body text, whose quality is evaluated using token-level F1 against manually annotated references. Their selector-fitting is built for repeated items on a single page; our programs extract one record per page across many pages.

- How code is reused: BardeenAgent generalizes a CSS selector for items within a page. We cluster whole article pages by a canonical structural signature derived from the page's HTML template and synthesize one full extraction program per template, reused across all pages of that template and across pagination.

- How we move between pages: We handle four pagination modes explicitly—numbered, infinite scroll, load-more, and single page—deriving numbered patterns by diffing two example URLs rather than relying on step-by-step agent navigation.

- Problem scope: WebLists focuses on English enterprise sites. Our system runs on multilingual, right-to-left (Arabic) news sites at thousands of pages per site, and incorporates multiple retrieval strategies to improve robustness against common bot-detection mechanisms encountered on the live web.

WebLists is also the closest existing benchmark to our setting, but it covers list-style enterprise extraction rather than long-form, multilingual article text; we therefore evaluate on a released news corpus for which no public benchmark currently exists.
  ```
   this section talk about this paper [[WebLists Extracting Structured Information From Complex Interactive Websites Using Executable LLM Agents]]
**Evaluation:**
- the golden dataset that we are building |:
		- **how much pages? 10 how many domains?** 20
		- "1. **Reproducibility of the "system" claims.** SWDE is reproducible; your live-web runs aren't (sites change, Cloudflare varies). You need to freeze a snapshot corpus (saved HTML) so the deployment numbers are re-runnable." Claude
		 I think that part is verry important, we need to save snapshots of our domains so they can be part of our golden dataset #ToDo1  
- Existing benchmark????
	- No good luck till now

**An ablation table:**
quantify each component's contribution: 
	(a) structural map vs raw HTML (token cost + accuracy)
		- raw HTML won't even fit inside the call of Gemma especially that we are using the free version with 1600 tokens per minute rate limit
			-Claude:```
			``` Report **token counts**: mean/median tokens of raw HTML vs the structural map across the corpus, and the **% of pages that exceed the context/rate budget** as raw HTML. That single number ("X% of pages don't even fit without the map") is a strong, honest finding.
			- For accuracy, compare only on the **subset of small pages where raw HTML _does_ fit**, plus a **truncated-HTML baseline** (first N tokens) for the rest — so you have an accuracy column, not just "it doesn't fit."```
			  ```
	(b) with/without clustering (LLM-call count) 
		- I think that's easy and trivial, it's exactly the number of pages extracted VS the number of LLM calls that where done for the Agent_for_single_page -but we will need to rer-run the tests for multiple domains to catch them at results.md #ToDo2 btw #ToDo1 will do so automatically :) 
	(c) with/without the self-repair retry loop (success rate). 
		- **how many domain exactly should we do to get the rate?** until now -20 domain- I saw that issue only once -need to re-review the results.md and the logs -search inside the runs and catch the errors using any LLMs--
