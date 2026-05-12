"""
Few-shot LLM Translation Evaluation
Uses OpenAI GPT-4o-mini with example translation pairs as context.
"""
import time
from typing import List, Tuple, Optional

from openai import OpenAI
from utils import evaluate_files, save_results, get_api_key, parse_numbered

# ── Config ────────────────────────────────────────────────────────────────────
MODEL      = "gpt-4o-mini"
BATCH_SIZE = 10
MAX_RETRIES = 3
RETRY_DELAY = 2.0

LANG_NAMES = {"en": "English", "sk": "Slovak"}

FEW_SHOT_EXAMPLES_EN_SK: List[Tuple[str, str]] = [
    ("Hello, how are you?",              "Ahoj, ako sa máš?"),
    ("The weather is nice today.",       "Dnes je pekné počasie."),
    ("I would like to order a coffee.",  "Chcel by som si objednať kávu."),
]

FEW_SHOT_EXAMPLES_SK_EN: List[Tuple[str, str]] = [
    ("Ahoj, ako sa máš?",               "Hello, how are you?"),
    ("Dnes je pekné počasie.",           "The weather is nice today."),
    ("Chcel by som si objednať kávu.",   "I would like to order a coffee."),
]

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


# ── Message builder ───────────────────────────────────────────────────────────
def _build_few_shot_messages(
    source_lang: str,
    target_lang: str,
    examples: List[Tuple[str, str]],
    user_text: str,
) -> List[dict]:
    """Build messages with examples as user/assistant turns."""
    system_prompt = (
        f"You are a professional translator from {LANG_NAMES[source_lang]} "
        f"to {LANG_NAMES[target_lang]}. "
        f"Translate each numbered line and return ONLY the translations, "
        f"one per line, preserving the numbering."
    )
    messages: List[dict] = [{"role": "system", "content": system_prompt}]
    for src, tgt in examples:
        messages.append({"role": "user",      "content": src})
        messages.append({"role": "assistant", "content": tgt})
    messages.append({"role": "user", "content": user_text})
    return messages


# ── Translation ───────────────────────────────────────────────────────────────
def translate_few_shot(
    texts: List[str],
    source_lang: str,
    target_lang: str,
    examples: List[Tuple[str, str]],
) -> List[str]:
    """Translate texts using few-shot prompting with batching."""
    translations: List[str] = []

    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start: start + BATCH_SIZE]
        numbered = "\n".join(f"{j + 1}. {t}" for j, t in enumerate(batch))
        messages = _build_few_shot_messages(source_lang, target_lang, examples, numbered)
        raw = _call_with_retry(messages)
        translations.extend(parse_numbered(raw, len(batch)))

    return translations


def translate_en_sk(texts: List[str]) -> List[str]:
    return translate_few_shot(texts, "en", "sk", FEW_SHOT_EXAMPLES_EN_SK)


def translate_sk_en(texts: List[str]) -> List[str]:
    return translate_few_shot(texts, "sk", "en", FEW_SHOT_EXAMPLES_SK_EN)


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    print("=" * 60)
    print("Few-shot LLM Translation Evaluation")
    print("=" * 60)

    print("\n[1/2] Evaluating EN → SK...")
    en_sk_results = evaluate_files(translate_en_sk, "EN_SK", "EN→SK")

    print("\n[2/2] Evaluating SK → EN...")
    sk_en_results = evaluate_files(translate_sk_en, "SK_EN", "SK→EN")

    save_results("02_few_shot_llm", en_sk_results, sk_en_results)

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
