"""
Zero-shot LLM Translation Evaluation
Uses OpenAI GPT-4o-mini for direct translation without examples.
"""
import time
from typing import List, Optional

from openai import OpenAI
from utils import evaluate_files, save_results, get_api_key, parse_numbered

# ── Config ────────────────────────────────────────────────────────────────────
MODEL      = "gpt-4o-mini"
BATCH_SIZE = 10
MAX_RETRIES = 3
RETRY_DELAY = 2.0   # seconds; doubles on each retry

LANG_NAMES = {"en": "English", "sk": "Slovak"}

# ── Lazy client ───────────────────────────────────────────────────────────────
_client: Optional[OpenAI] = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=get_api_key("OPENAI_API_KEY"))
    return _client


# ── API call with retry ───────────────────────────────────────────────────────
def _call_with_retry(messages: List[dict], temperature: float = 0.3) -> str:
    """Single API call with exponential backoff."""
    client = _get_client()
    delay = RETRY_DELAY
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=temperature,
                max_tokens=2048,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if attempt == MAX_RETRIES:
                print(f"  API error after {MAX_RETRIES} retries: {e}")
                return ""
            print(f"  Retry {attempt}/{MAX_RETRIES} after error: {e}")
            time.sleep(delay)
            delay *= 2
    return ""


# ── Translation ───────────────────────────────────────────────────────────────
def translate_zero_shot(
    texts: List[str],
    source_lang: str,
    target_lang: str,
) -> List[str]:
    """Translate texts using zero-shot prompting with batching."""
    system_prompt = (
        f"You are a professional translator from {LANG_NAMES[source_lang]} "
        f"to {LANG_NAMES[target_lang]}. "
        f"Translate each numbered line and return ONLY the translations, "
        f"one per line, preserving the numbering."
    )

    translations: List[str] = []

    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start: start + BATCH_SIZE]
        numbered = "\n".join(f"{j + 1}. {t}" for j, t in enumerate(batch))

        raw = _call_with_retry([
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": numbered},
        ])

        translations.extend(parse_numbered(raw, len(batch)))

    return translations


def translate_en_sk(texts: List[str]) -> List[str]:
    return translate_zero_shot(texts, "en", "sk")


def translate_sk_en(texts: List[str]) -> List[str]:
    return translate_zero_shot(texts, "sk", "en")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    print("=" * 60)
    print("Zero-shot LLM Translation Evaluation")
    print("=" * 60)

    print("\n[1/2] Evaluating EN → SK...")
    en_sk_results = evaluate_files(translate_en_sk, "EN_SK", "EN→SK")

    print("\n[2/2] Evaluating SK → EN...")
    sk_en_results = evaluate_files(translate_sk_en, "SK_EN", "SK→EN")

    save_results("01_zero_shot_llm", en_sk_results, sk_en_results)

    print("\n" + "=" * 60)
    print("Evaluation Complete!")
    print("=" * 60)
    for label, res in [("EN→SK", en_sk_results), ("SK→EN", sk_en_results)]:
        comet_str = ""
        if res.get("avg_comet") is not None:
            comet_str = f", COMET={res['avg_comet']:.2f}"
        print(f"{label}: BLEU={res.get('avg_bleu', 0):.2f}, "
              f"chrF={res.get('avg_chrf', 0):.2f}{comet_str}")


if __name__ == "__main__":
    main()
