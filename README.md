# AI Web Scraping Agent

An AI Agent that automates web scraping by using an LLM to generate extraction code.

## What It Does
- Takes a URL that you need to scrape + description of the data you want to extract
- Automatically generates scraping code using an LLM
- Outputs the scraped data in JSON format

## How It Works

### The agent follows this flow:

1. **Fetch & Simplify**: The code fetches the entire HTML content of the page that needs to be scraped, then creates a **structural map** of that HTML content.
   
   > **What is a structural map?** A simplified, nested representation of the HTML structure showing only relevant tags (div, section, article, etc.) with their classes and IDs. This makes it easier for the LLM to understand the page layout without processing the entire raw HTML.

2. **Generate Code**: The structural map + the user's description of the data that needs to be scraped is fed to the LLM (using Gemini's free API), and the LLM generates Python code to scrape the page.

3. **Execute & Show Sample**: An executor runs the generated code in a basic sandbox and shows a sample of the output to the user. If the code produces an error, the code and context get sent to the LLM again so it can generate new code.

4. **User Validation**: The user decides if the sample is good enough. If not, the user's objection gets sent to the LLM along with the previous context so it can generate improved code.

## Work Plan

### Set up the manual scraping process:
- [x] Code for scraping
- [x] Pinpoint the data that needs to be fed to the code manually (ex: CSS selectors)

### Automate the process:
- [x] Replace human input with LLM (ex: use LLM to find the CSS selectors)
- [ ] Scale to scrape multiple pages (next step)

### Ship into a simple UI:
- [ ] Create Streamlit interface

## Current Status
The agent successfully scrapes **single pages**. The next part will be to use this agent to help scrape multiple pages.
