"""
Reusable helpers for ground-truth generation and evaluation.
Adapted from the LLM Zoomcamp module 4 pattern.
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm.auto import tqdm

# Pricing per 1M tokens. Update these if you switch models.
# gpt-5.4-mini pricing used as the course's reference point.
PRICING = {
    "gpt-5.4-mini": {"input": 0.15, "output": 0.60},
}


def llm_structured(client, instructions, user_prompt, response_model, model="gpt-5.4-mini"):
    messages = [
        {"role": "developer", "content": instructions},
        {"role": "user", "content": user_prompt},
    ]

    response = client.responses.parse(
        model=model,
        input=messages,
        text_format=response_model,
    )

    return response.output_parsed, response.usage


def llm_structured_retry(client, instructions, user_prompt, response_model,
                          model="gpt-5.4-mini", max_retries=3, backoff=2.0):
    last_error = None
    for attempt in range(max_retries):
        try:
            return llm_structured(client, instructions, user_prompt, response_model, model=model)
        except Exception as e:
            last_error = e
            wait = backoff * (attempt + 1)
            print(f"  retry {attempt + 1}/{max_retries} after error: {e} (waiting {wait:.1f}s)")
            time.sleep(wait)

    raise last_error


def calc_price(usage, model="gpt-5.4-mini"):
    rates = PRICING.get(model, PRICING["gpt-5.4-mini"])
    input_cost = usage.input_tokens * rates["input"] / 1_000_000
    output_cost = usage.output_tokens * rates["output"] / 1_000_000
    total_cost = input_cost + output_cost

    return {
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost,
    }


def calc_total_price(usages, model="gpt-5.4-mini"):
    total = 0.0
    for usage in usages:
        total += calc_price(usage, model=model)["total_cost"]
    return total


def map_progress(pool: ThreadPoolExecutor, items, func):
    """
    Submits `func` for every item in `items` using the given thread pool,
    shows a progress bar, and returns results in the SAME ORDER as `items`
    (important, since we zip results back up with ground_truth/documents).
    """
    futures = {pool.submit(func, item): idx for idx, item in enumerate(items)}
    results = [None] * len(items)

    for future in tqdm(as_completed(futures), total=len(futures)):
        idx = futures[future]
        results[idx] = future.result()

    return results