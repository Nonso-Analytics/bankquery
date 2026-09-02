"""
BankQuery scraper for First Bank.

Structure: the FAQ page is built with Elementor "Toggle" widgets, one
per product/category. Each category is preceded by a heading widget
(e.g. "Savings Account FAQs") immediately followed by its toggle
group in the page's DOM order:
    <heading widget>Category Name FAQs</heading widget>
    <toggle widget>
      <div class="elementor-toggle-item">
        <a class="elementor-toggle-title">QUESTION</a>
        <div class="elementor-tab-content">ANSWER (html)</div>
      </div>
      ...
    </toggle widget>

Run:
    uv run python scrapers/scrape_firstbank.py
"""

import re
import sys
import requests
from bs4 import BeautifulSoup

sys.path.insert(0, "scrapers")
from common import make_faq, save_faqs, HEADERS

URL = "https://www.firstbanknigeria.com/contact/faqs/"


def scrape_firstbank():
    browser_headers = {
        **HEADERS,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }
    resp = requests.get(URL, headers=browser_headers, timeout=20)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "lxml")

    faqs = []

    # Find every heading widget and every toggle widget, in DOM order.
    heading_widgets = soup.select(".elementor-widget-heading")
    toggle_widgets = soup.select(".elementor-widget-toggle")

    # Tag each element with its position in the document so we can
    # walk them in true document order together.
    def doc_position(el):
        return list(soup.descendants).index(el)

    # Faster approach: use find_all on the whole soup once, in order,
    # filtering to just headings and toggles.
    ordered = soup.find_all(class_=["elementor-widget-heading", "elementor-widget-toggle"])

    current_category = "General"
    for el in ordered:
        classes = el.get("class", [])
        if "elementor-widget-heading" in classes:
            title_el = el.select_one(".elementor-heading-title")
            if title_el:
                text = title_el.get_text(strip=True)
                # Skip the page's own top intro heading/subheading (too short
                # or doesn't end in "FAQs").
                if text.endswith("FAQs") and text != "FAQs":
                    current_category = text.replace(" FAQs", "").strip()
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

                faqs.append(make_faq("First Bank", current_category, question, answer, URL))

    return faqs


if __name__ == "__main__":
    faqs = scrape_firstbank()
    save_faqs(faqs, "firstbank")