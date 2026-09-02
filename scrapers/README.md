## Status: Phase 1 - Data Collection Complete

| Bank | FAQs | Categories | Method |
|---|---|---|---|
| Fidelity Bank | 228 | 8 | Live scrape (`scrape_fidelity.py`) — needed browser-like headers |
| Zenith Bank | 295 | 11 | Manual browser copy + parse (`parse_zenith_manual.py`) — site protected by Incapsula, blocks both `requests` and headless Playwright |
| GTBank | 55 | 4 | Live scrape (`scrape_gtbank.py`) — main help-centre page only |
| First Bank | 329 | ~60 | Live scrape (`scrape_firstbank.py`) |
| UBA | 337 | 29 | Live scrape (`scrape_uba.py`) — used real FAQ URL `roa.ubagroup.com/about-uba/help/faq/`, not the initial guessed hub URL |
| **Total** | **1,244** | — | — |

Combined dataset: `data/faqs.json`