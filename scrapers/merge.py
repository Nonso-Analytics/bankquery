"""
Merge all per-bank JSON files into data/faqs.json.

Run after all five scrapers have produced their per-bank output:
    uv run python scrapers/merge.py
"""

from common import merge_all

BANK_SLUGS = ["fidelity", "zenith", "gtbank", "firstbank", "uba"]

if __name__ == "__main__":
    merge_all(BANK_SLUGS)