"""
BankQuery parser for Zenith Bank — parses a manually-saved HTML file
instead of live-scraping, since Zenith's Incapsula bot protection blocks
both plain requests and headless browsers (Playwright).

To regenerate data/zenith.json:
  1. Open https://www.zenithbank.com/customer-service/faqs/ in your own
     regular browser (passes Incapsula fine since it's human-driven).
  2. View Page Source (Ctrl+U) or copy the rendered DOM from DevTools.
  3. Save the full HTML to data/zenith_manual.html
  4. Run: uv run python scrapers/parse_zenith_manual.py

Structure: each category is a <section class="block-faq layoutB">
containing an <h2> title and a list of <div class="block-faq__item">
Q&A pairs (button = question, .block-faq__answer__content = answer).
"""

import sys
from bs4 import BeautifulSoup

sys.path.insert(0, "scrapers")
from common import make_faq, save_faqs

SOURCE_FILE = "data/zenith_manual.html"
SOURCE_URL = "https://www.zenithbank.com/customer-service/faqs/"


def parse_zenith(html_path: str):
    with open(html_path, encoding="utf-8") as f:
        html = f.read()

    soup = BeautifulSoup(html, "lxml")
    faqs = []
    sections = soup.select("section.block-faq")

    for section in sections:
        title_el = section.select_one(".block-faq__title h2")
        category = title_el.get_text(strip=True) if title_el else "General"

        for item in section.select(".block-faq__item"):
            q_el = item.select_one(".block-faq__question")
            a_el = item.select_one(".block-faq__answer__content")
            if not q_el or not a_el:
                continue

            question = q_el.get_text(" ", strip=True)

            answer_parts = []
            for child in a_el.find_all(["p", "li", "td"], recursive=True):
                text = child.get_text(" ", strip=True)
                if text:
                    answer_parts.append(text)
            answer = "; ".join(answer_parts) if answer_parts else a_el.get_text(" ", strip=True)

            if len(question) < 5 or len(answer) < 2:
                continue

            faqs.append(make_faq("Zenith Bank", category, question, answer, SOURCE_URL))

    return faqs

if __name__ == "__main__":
    faqs = parse_zenith(SOURCE_FILE)
    save_faqs(faqs, "zenith")