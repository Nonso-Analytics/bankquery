"""
BankQuery scraper for Fidelity Bank.

Structure: the FAQ page uses Elementor "Advanced Accordion" widgets.
Each category lives in an outer tab container:
    <div id="<category-slug>-tab" class="eael-tab-content-item">
Inside each category, individual Q&A pairs are:
    <div class="eael-accordion-list">
      <div class="eael-accordion-header">
        <span class="eael-accordion-tab-title">QUESTION</span>
      </div>
      <div class="eael-accordion-content">ANSWER (html)</div>
    </div>

Run:
    uv run python scrapers/scrape_fidelity.py
"""

import re
import sys
import requests
from bs4 import BeautifulSoup

sys.path.insert(0, "scrapers")
from common import make_faq, save_faqs, HEADERS

URL = "https://www.fidelitybank.ng/help-support/faqs/"

CATEGORIES = [
    "Account Opening/Upgrade", "E Channels", "Cards", "Loans",
    "Savings Accounts", "Virtual Account", "Diaspora Banking",
    "SME Banking", "Complaints and Enquiries",
    # Note: "Complaints and Enquiries" has no accordion Q&A content on the
    # live page — it's just a single sentence pointing to a contact link.
    # 0 FAQs from this category is expected, not a scraping failure.
]


def slugify(text: str) -> str:
    """Match the site's own id-slug style, e.g. 'E Channels' -> 'e-channels'."""
    text = text.lower().replace("/", "")
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-")


def scrape_fidelity():
    session = requests.Session()
    browser_headers = {
        **HEADERS,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }
    resp = session.get(URL, headers=browser_headers, timeout=20)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "lxml")

    faqs = []

    for category in CATEGORIES:
        slug = slugify(category)
        container = soup.find(id=f"{slug}-tab")
        if container is None:
            print(f"  WARNING: could not find container for category '{category}' (slug: {slug}-tab)")
            continue

        accordion_items = container.select(".eael-accordion-list")
        for item in accordion_items:
            q_el = item.select_one(".eael-accordion-tab-title")
            a_el = item.select_one(".eael-accordion-content")
            if not q_el or not a_el:
                continue

            question = q_el.get_text(strip=True)
            # Answer may contain <p> and <ul><li> — join list items with commas
            # so requirements lists don't get mashed together with no separator.
            answer_parts = []
            for child in a_el.find_all(["p", "li"], recursive=True):
                text = child.get_text(strip=True)
                if text:
                    answer_parts.append(text)
            answer = "; ".join(answer_parts) if answer_parts else a_el.get_text(" ", strip=True)

            if len(question) < 5 or len(answer) < 3:
                continue

            faqs.append(make_faq("Fidelity Bank", category, question, answer, URL))

    return faqs


if __name__ == "__main__":
    faqs = scrape_fidelity()
    save_faqs(faqs, "fidelity")