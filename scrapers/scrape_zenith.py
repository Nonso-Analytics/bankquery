"""
BankQuery scraper for Zenith Bank.

Site is protected by Incapsula (blocks plain `requests`), so this uses
Playwright to drive a real headless browser. Structure is clean and
consistent: each category is a <section class="block-faq layoutB">
containing a <h2> title and a list of <div class="block-faq__item">
Q&A pairs.

Run:
    uv run python scrapers/scrape_zenith.py
"""

import sys
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

sys.path.insert(0, "scrapers")
from common import make_faq, save_faqs

URL = "https://www.zenithbank.com/customer-service/faqs/"


def fetch_rendered_html(url: str) -> str:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            )
        )
        page.goto(url, timeout=30000, wait_until="networkidle")
        page.wait_for_timeout(5000)
        html = page.content()
        browser.close()
    return html


def scrape_zenith():
    html = fetch_rendered_html(URL)
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

            # Question text is the button's direct text, excluding the
            # arrow icon's (empty) svg content.
            question = q_el.get_text(" ", strip=True)

            # Answer may contain <p>, <ul><li>, or <table> — join with
            # "; " so lists/requirements don't get mashed together.
            answer_parts = []
            for child in a_el.find_all(["p", "li", "td"], recursive=True):
                text = child.get_text(" ", strip=True)
                if text:
                    answer_parts.append(text)
            answer = "; ".join(answer_parts) if answer_parts else a_el.get_text(" ", strip=True)

            if len(question) < 5 or len(answer) < 2:
                continue

            faqs.append(make_faq("Zenith Bank", category, question, answer, URL))

    return faqs


if __name__ == "__main__":
    faqs = scrape_zenith()
    save_faqs(faqs, "zenith")