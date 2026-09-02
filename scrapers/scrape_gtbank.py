"""
BankQuery scraper for GTBank.

Structure: the FAQ page uses tabs to separate categories
(#nav-account-services, #nav-e-channels-737, #nav-card-services,
#nav-non-resident-nigerian, #nav-forms), each containing its own
Bootstrap accordion of Q&A pairs:
    <div class="accordion-item">
      <div class="accordion-question">QUESTION</div>
      <div class="accordion-answer">ANSWER (html)</div>
    </div>

Run:
    uv run python scrapers/scrape_gtbank.py
"""

import re
import sys
import requests
from bs4 import BeautifulSoup

sys.path.insert(0, "scrapers")
from common import make_faq, save_faqs, HEADERS

URL = "https://www.gtbank.com/help-centre"

CATEGORY_TAB_IDS = {
    "nav-account-services": "Account Services",
    "nav-e-channels-737": "E-Channels & 737",
    "nav-card-services": "Card Services",
    "nav-non-resident-nigerian": "Non-resident Nigerian",
    "nav-forms": "Forms",
}


def scrape_gtbank():
    browser_headers = {
        **HEADERS,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }
    resp = requests.get(URL, headers=browser_headers, timeout=20)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "lxml")

    faqs = []

    for tab_id, category in CATEGORY_TAB_IDS.items():
        container = soup.find(id=tab_id)
        if container is None:
            print(f"  WARNING: could not find tab container '#{tab_id}' for category '{category}'")
            continue

        for item in container.select(".accordion-item"):
            q_el = item.select_one(".accordion-question")
            a_el = item.select_one(".accordion-answer")
            if not q_el or not a_el:
                continue

            # Question text sits alongside the plus/minus icon spans;
            # strip those, collapse whitespace, and remove any leading
            # "1. " numbering the site adds.
            question = q_el.get_text(" ", strip=True)
            question = re.sub(r"^\d+\.\s*", "", question)

            answer_parts = []
            for child in a_el.find_all(["p", "li"], recursive=True):
                text = child.get_text(" ", strip=True)
                if text:
                    answer_parts.append(text)
            answer = "; ".join(answer_parts) if answer_parts else a_el.get_text(" ", strip=True)

            if len(question) < 5 or len(answer) < 2:
                continue

            faqs.append(make_faq("GTBank", category, question, answer, URL))

    return faqs


if __name__ == "__main__":
    faqs = scrape_gtbank()
    save_faqs(faqs, "gtbank")