"""
This script is made to ease up the process of developing the system.
It tests on multiple domains -from 5 to 20-
The domains names are put in a list and the user get to input the start index and the end index, each index corresponds to a domain name in the list.
The script directly invokes orch_interactive_pagination.py
It collect the data for all the domain in a table and save it incrementally in a file named testing_multiple_domains.md, each time the script is run, it will append the new table to the file with a line indicating the time above the table.
The table should contain the following columns:
- Domain Name
- Pages requested
- Pages extracted
- Failed pages
- kind of pagination
- Articles extracted
- Articles failed
- Run folder
- useful run metrics for comparing code changes
- user input
- a column called NaN_precentage that indicates the percentage of NaN values in the data collected for that domain, the cell of this column might have multiple lines, each line indicating the percentage of NaN values for a specific column in the data collected for that domain, for example: 80% N/A in text body, same as the data stored in results.md

the api key is stored in the .env file under the name: GOOGLE_API_KEY 
"""
import asyncio
import builtins
import os
import re
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional


ROOT = Path(__file__).resolve().parent
RESULTS_PATH = ROOT / "results.md"
REPORT_PATH = ROOT / "testing_multiple_domains.md"
DEFAULT_REQUIREMENTS = "title, date, author, article body text"
DEFAULT_MODEL_CHOICE = "2"  # models/gemma-4-31b-it in the orchestrator menu
REPORT_COLUMNS = [
    "Domain Name",
    "Status",
    "Run folder",
    "Pages requested",
    "Pages extracted",
    "Failed pages",
    "Articles extracted",
    "Articles failed",
    "kind of pagination",
    "Required fields",
    "LLM calls",
    "Clusters",
    "Total time",
    "Fetch method",
    "Estimated cost",
    "Errors",
    "user input",
    "NaN_precentage",
]


@dataclass
class DomainConfig:
    domain: str
    pagination_type: str   # "numbered" / "load_more" / "infinite_scroll"
    first_link: str         # must be a valid link
    second_link: Optional[str]  # None for "infinite_scroll" or "load_more" pagination
    count: int             # number of pages/clicks/scrolls
    required_fields: str

domains = [
    DomainConfig(
        domain="youm7",
        pagination_type="numbered",
        first_link="https://www.youm7.com/Section/%D8%A3%D8%AE%D8%A8%D8%A7%D8%B1-%D8%B9%D8%A7%D8%AC%D9%84%D8%A9/65/1",
        second_link="https://www.youm7.com/Section/%D8%A3%D8%AE%D8%A8%D8%A7%D8%B1-%D8%B9%D8%A7%D8%AC%D9%84%D8%A9/65/2",
        count=3,
        required_fields=DEFAULT_REQUIREMENTS
    ),
    DomainConfig(
        domain="pchrgaza",
        pagination_type="numbered",
        first_link="https://pchrgaza.org/ar/category/genocide-on-gaza-ar/testimonies-from-the-war-ar/",
        second_link="https://pchrgaza.org/ar/category/genocide-on-gaza-ar/testimonies-from-the-war-ar/page/2/",
        count=3,
        required_fields="title, date, article body text"
    ),
    DomainConfig(
        domain="almasryalyoum",
        pagination_type="numbered",
        first_link="https://www.almasryalyoum.com/news/index?typeid=1&sectionid=10",
        second_link="https://www.almasryalyoum.com/news/index?typeid=1&sectionid=10&page=2",
        count=3,
        required_fields=DEFAULT_REQUIREMENTS
    ),
    
    DomainConfig(
        domain="euromedmonitor",
        pagination_type="numbered",
        first_link="https://euromedmonitor.org/ar/category/26/%D8%A7%D9%84%D9%86%D8%B2%D8%A7%D8%B9%D8%A7%D8%AA-%D8%A7%D9%84%D9%85%D8%B3%D9%84%D8%AD%D8%A9",
        second_link="https://euromedmonitor.org/ar/category/26/%D8%A7%D9%84%D9%86%D8%B2%D8%A7%D8%B9%D8%A7%D8%AA-%D8%A7%D9%84%D9%85%D8%B3%D9%84%D8%AD%D8%A9?page=2",
        count=3,
        required_fields="title, date, article body text"
    ),
    DomainConfig(
        domain="aawsat",
        pagination_type="infinite_scroll",
        first_link="https://aawsat.com/%D8%A7%D9%84%D8%B1%D8%A3%D9%8A",
        second_link=None,
        count=2,
        required_fields=DEFAULT_REQUIREMENTS
    ),
    DomainConfig(
            domain="independentarabia",
            pagination_type="infinite_scroll",
            first_link="https://www.independentarabia.com/%D8%AB%D9%82%D8%A7%D9%81%D8%A9/%D8%B3%D9%8A%D9%86%D9%85%D8%A7",
            second_link=None,
            count=2,
            required_fields=DEFAULT_REQUIREMENTS
        ),
    DomainConfig(
        domain="arageek",
        pagination_type="load_more",
        first_link="https://www.arageek.com/tech",
        second_link=None,
        count=2,
        required_fields=DEFAULT_REQUIREMENTS
    ),
    DomainConfig(
        domain="majalla",
        pagination_type="load_more",
        first_link="https://www.majalla.com/sections/%D8%B3%D9%8A%D8%A7%D8%B3%D8%A9",
        second_link=None,
        count=2,
        required_fields=DEFAULT_REQUIREMENTS
    ),
    DomainConfig(
            domain="theguardian",
            pagination_type="numbered",
            first_link="https://www.theguardian.com/world/gaza",
            second_link="https://www.theguardian.com/world/gaza?page=2",
            count=3,
            required_fields=DEFAULT_REQUIREMENTS
    ),
    DomainConfig(
        domain="btselem",
        pagination_type="numbered",
        first_link="https://www.btselem.org/ota/100/all",
        second_link="https://www.btselem.org/ota/100/all?page=1",
        count=3,
        required_fields="title, date, article body text"
    ),
    DomainConfig(
        domain="arabic.cnn",
        pagination_type="infinite_scroll",
        first_link="https://arabic.cnn.com/tag/gaza_strip",
        second_link=None,
        count=2,
        required_fields="title, date, article body text"
    ),
    DomainConfig(
            domain="aljadeedmagazine",
            pagination_type="infinite_scroll",
            first_link="https://www.aljadeedmagazine.com/%D9%85%D9%82%D8%A7%D9%84%D8%A7%D8%AA",
            second_link=None,
            count=2,
            required_fields=DEFAULT_REQUIREMENTS
    ),
    DomainConfig(
        domain="alsifr",
        pagination_type="load_more",
        first_link="https://alsifr.org/kam-kaif",
        second_link=None,
        count=2,
        required_fields=DEFAULT_REQUIREMENTS
    ),
    DomainConfig(
        domain="ida2at",
        pagination_type="numbered",
        first_link="https://www.ida2at.com/category/art-literature/",
        second_link="https://www.ida2at.com/category/art-literature/page/2/",
        count=3,
        required_fields=DEFAULT_REQUIREMENTS
    ),
    DomainConfig(
        domain="lakome2",
        pagination_type="load_more",
        first_link="https://lakome2.com/category/art/",
        second_link=None,
        count=2,
        required_fields="title, date, article body text"
    ),
    DomainConfig(
        domain="mana",
        pagination_type="numbered",
        first_link="https://mana.net/category/articles/",
        second_link="https://mana.net/category/articles/page/2/",
        count=3,
        required_fields="title, date, article body text"
    ),
    DomainConfig(
        domain="acpss.ahram",
        pagination_type="numbered",
        first_link="https://acpss.ahram.org.eg/OuterWriter/28/%D9%85%D9%82%D8%A7%D9%84%D8%A7%D8%AA/0.aspx",
        second_link="https://acpss.ahram.org.eg/OuterWriter/28/%D9%85%D9%82%D8%A7%D9%84%D8%A7%D8%AA/30.aspx",
        count=3,
        required_fields=DEFAULT_REQUIREMENTS
    ), 
    DomainConfig(
        domain="palestine-studies",
        pagination_type="infinite_scroll",
        first_link="https://www.palestine-studies.org/ar/blogs/explorer?f%5B0%5D=field_blog_series%3A19943",
        second_link=None,
        count=2,
        required_fields=DEFAULT_REQUIREMENTS
    )              
]


class OrchestratorInput:
    def __init__(self, config: DomainConfig, api_key: str):
        self.config = config
        self.api_key = api_key
        self.step = 0

    def __call__(self, prompt: str = "") -> str:
        answer = self._answer(prompt)
        self.step += 1
        return answer

    def _answer(self, prompt: str) -> str:
        if self.step == 0:
            return self.api_key
        if self.step == 1:
            return DEFAULT_MODEL_CHOICE
        if self.step == 2:
            return self.config.first_link
        if self.step == 3:
            return self.config.required_fields
        if self.step == 4:
            return pagination_choice(self.config)

        if self.config.pagination_type == "numbered":
            return self._numbered_answer(prompt)
        if self.config.pagination_type == "load_more":
            return self._load_more_answer()
        if self.config.pagination_type == "infinite_scroll":
            return str(self.config.count)

        return ""

    def _numbered_answer(self, prompt: str) -> str:
        if "How many pages" in prompt:
            return str(self.config.count)
        if self.step == 5:
            return ""  # page 1: use listing URL
        if self.step == 6:
            return self.config.second_link or ""
        return ""  # accept derived pattern, or skip manual pattern if derivation fails

    def _load_more_answer(self) -> str:
        if self.step == 5:
            return ""  # auto-detect the load-more button
        return str(self.config.count)


@contextmanager
def patched_input(config: DomainConfig, api_key: str):
    original_input = builtins.input
    builtins.input = OrchestratorInput(config, api_key)
    try:
        yield
    finally:
        builtins.input = original_input


def pagination_choice(config: DomainConfig) -> str:
    choices = {
        "numbered": "1",
        "infinite_scroll": "3",
        "load_more": "4",
    }
    return choices[config.pagination_type]


def load_api_key() -> str:
    api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if api_key:
        return api_key

    env_path = ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            key, separator, value = line.partition("=")
            if separator and key.strip() == "GOOGLE_API_KEY":
                return value.strip().strip('"').strip("'")

    raise RuntimeError("GOOGLE_API_KEY was not found in the environment or .env file.")


def choose_domains() -> list[DomainConfig]:
    print("Available domains:")
    for index, config in enumerate(domains):
        print(f"  {index}: {config.domain}")

    choice = input("\nEnter count or slice (example: 5 or 5:10, blank for all): ").strip()
    if not choice:
        return domains

    if ":" in choice:
        start_text, end_text = choice.split(":", 1)
        start = int(start_text) if start_text else 0
        end = int(end_text) if end_text else len(domains)
        return domains[start:end]

    count = int(choice)
    return domains[:count]


def start_report(selected_domains: list[DomainConfig]) -> None:
    names = ", ".join(config.domain for config in selected_domains)
    with REPORT_PATH.open("a", encoding="utf-8") as file:
        file.write(f"\n\n## Test run {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        file.write(f"Selected domains: {names}\n\n")
        file.write("| " + " | ".join(REPORT_COLUMNS) + " |\n")
        file.write("|" + "|".join("---" for _ in REPORT_COLUMNS) + "|\n")


def append_report_row(row: list[str]) -> None:
    escaped = [escape_markdown_cell(value) for value in row]
    with REPORT_PATH.open("a", encoding="utf-8") as file:
        file.write("| " + " | ".join(escaped) + " |\n")


def escape_markdown_cell(value: str) -> str:
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def read_results() -> str:
    if RESULTS_PATH.exists():
        return RESULTS_PATH.read_text(encoding="utf-8")
    return ""


def latest_new_section(before_results: str) -> str:
    after_results = read_results()
    if after_results.startswith(before_results):
        new_text = after_results[len(before_results):]
        sections = re.findall(r"\n## Run .*?(?=\n## Run |\Z)", new_text, flags=re.S)
        if sections:
            return sections[-1]

    sections = re.findall(r"\n## Run .*?(?=\n## Run |\Z)", after_results, flags=re.S)
    return sections[-1] if sections else ""


def section_value(section: str, label: str, default: str = "") -> str:
    match = re.search(rf"- \*\*{re.escape(label)}:\*\*\s*(.+)", section)
    return match.group(1).strip() if match else default


def run_folder(section: str) -> str:
    folder = section_value(section, "Run directory", "not reported").strip("`")
    if folder == "not reported":
        return folder
    return f"[{folder}]({folder})"


def section_int(section: str, label: str) -> Optional[int]:
    value = section_value(section, label)
    match = re.search(r"\d+", value.replace(",", ""))
    return int(match.group()) if match else None


def nan_summary(section: str) -> str:
    lines = []
    for field, percent in re.findall(r"  - `([^`]+)`: \d+/\d+ N/A \((\d+)%\)", section):
        lines.append(f"{percent}% N/A in {field}")
    return "\n".join(lines) if lines else "not reported"


def error_summary(section: str) -> str:
    error_line = re.search(r"- \*\*Errors(?: \((\d+)\))?:\*\*\s*(.+)", section)
    if error_line and error_line.group(2).strip().lower() == "none":
        return "none"

    count = error_line.group(1) if error_line else ""
    first_error = re.search(r"- \*\*Errors(?: \(\d+\))?:\*\*.*?\n  - (.+)", section, flags=re.S)
    if first_error:
        prefix = f"{count} errors" if count else "errors"
        return f"{prefix}: {first_error.group(1).strip()}"
    return "not reported"


def failed_pages(section: str) -> str:
    requested = section_int(section, "Pages requested")
    processed = section_int(section, "Pages processed")
    if requested is None or processed is None:
        return "unknown"
    return str(max(requested - processed, 0))


def run_status(section: str, error: Optional[BaseException]) -> str:
    if error:
        return "script failed"
    if not section:
        return "no results section"
    if failed_pages(section) not in {"0", "unknown"}:
        return "page failures"
    articles_failed = section_int(section, "Articles failed") or 0
    errors = error_summary(section)
    if articles_failed or errors != "none":
        return "warnings"
    return "ok"


def user_input_summary(config: DomainConfig) -> str:
    if config.pagination_type == "numbered":
        return f"page 1: {config.first_link}\npage 2: {config.second_link}"
    if config.pagination_type == "load_more":
        return f"{config.first_link}\nclicks: {config.count}"
    return f"{config.first_link}\nscrolls: {config.count}"


def build_row(config: DomainConfig, section: str, error: Optional[BaseException]) -> list[str]:
    if error:
        error_text = f"script failed: {error}"
    else:
        error_text = error_summary(section)

    return [
        config.domain,
        run_status(section, error),
        run_folder(section),
        section_value(section, "Pages requested", "unknown"),
        section_value(section, "Pages processed", "unknown"),
        failed_pages(section),
        section_value(section, "Articles extracted", "unknown"),
        section_value(section, "Articles failed", "unknown"),
        section_value(section, "Pagination type", config.pagination_type),
        config.required_fields,
        section_value(section, "LLM calls", "unknown"),
        section_value(section, "Clusters (unique structures)", "unknown"),
        section_value(section, "Total time", "unknown"),
        section_value(section, "Fetch method", "not reported"),
        section_value(section, "Estimated cost", "not reported"),
        error_text,
        user_input_summary(config),
        nan_summary(section) if not error else "not reported",
    ]


def run_domain(config: DomainConfig, api_key: str) -> list[str]:
    import orch_interactive_pagination as orchestrator

    print(f"\nRunning {config.domain}...")
    before_results = read_results()
    error = None

    try:
        with patched_input(config, api_key):
            asyncio.run(orchestrator.main())
    except SystemExit as exc:
        if exc.code not in (None, 0):
            error = exc
    except Exception as exc:
        error = exc

    section = latest_new_section(before_results)
    return build_row(config, section, error)


def main() -> None:
    api_key = load_api_key()
    selected_domains = choose_domains()
    if not selected_domains:
        print("No domains selected.")
        return

    start_report(selected_domains)
    for config in selected_domains:
        row = run_domain(config, api_key)
        append_report_row(row)
        print(f"Saved report row for {config.domain}.")

    print(f"\nDone. Report saved to {REPORT_PATH}")


if __name__ == "__main__":
    main()