"""
BankQuery scraper for UBA.

Structure: same Elementor "Toggle" pattern as First Bank. Each category
is preceded by one or more heading widgets (the last one is the real
category/product name, e.g. "Diaspora Account", "UBA Internet Banking")
immediately followed by its toggle group in DOM order:
    <heading widget>Category Name</heading widget>
    <toggle widget>
      <div class="elementor-toggle-item">
        <a class="elementor-toggle-title">QUESTION</a>
        <div class="elementor-tab-content">ANSWER (html)</div>
      </div>
      ...
    </toggle widget>

Run:
    uv run python scrapers/scrape_uba.py
"""

import sys
import requests
from bs4 import BeautifulSoup

sys.path.insert(0, "scrapers")
from common import make_faq, save_faqs, HEADERS

URL = "https://roa.ubagroup.com/about-uba/help/faq/"


def scrape_uba():
    browser_headers = {
        **HEADERS,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }
    resp = requests.get(URL, headers=browser_headers, timeout=20)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "lxml")

    faqs = []
    ordered = soup.find_all(class_=["elementor-widget-heading", "elementor-widget-toggle"])

    current_category = "General"
    for el in ordered:
        classes = el.get("class", [])
        if "elementor-widget-heading" in classes:
            title_el = el.select_one(".elementor-heading-title")
            if title_el:
                text = title_el.get_text(strip=True)
                # Skip long intro/subtitle sentences — real category names
                # are short (a few words), intro text is a full sentence.
                if text and len(text.split()) <= 6:
                    current_category = text
        elif "elementor-widget-toggle" in classes:
            for item in el.select(".elementor-toggle-item"):
                q_el = item.select_one(".elementor-toggle-title")
                a_el = item.select_one(".elementor-tab-content")
                if not q_el or not a_el:
                    continue

                question = q_el.get_text(" ", strip=True)

                answer_parts = []
                for child in a_el.find_all(["p", "li"], recursive=True):
                    text = child.get_text(" ", strip=True)
                    if text:
                        answer_parts.append(text)
                answer = "; ".join(answer_parts) if answer_parts else a_el.get_text(" ", strip=True)

                if len(question) < 5 or len(answer) < 2:
                    continue

                faqs.append(make_faq("UBA", current_category, question, answer, URL))

    return faqs


if __name__ == "__main__":
    faqs = scrape_uba()
    save_faqs(faqs, "uba")