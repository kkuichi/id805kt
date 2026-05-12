"""
MarianMT Translation Evaluation
Uses Helsinki-NLP MarianMT models for neural machine translation
"""
from typing import List, Optional

import torch
from transformers import MarianMTModel, MarianTokenizer
from utils import evaluate_files, save_results

# SK-specific models preferred; fallback to general Slavic
MODELS = {
    "en-sk": [
        "Helsinki-NLP/opus-mt-en-sk",     # direct EN→SK (if available)
        "Helsinki-NLP/opus-mt-en-sla",     # fallback: EN→Slavic
    ],
    "sk-en": [
        "Helsinki-NLP/opus-mt-sk-en",     # direct SK→EN (if available)
        "Helsinki-NLP/opus-mt-sla-en",     # fallback: Slavic→EN
    ],
}

BATCH_SIZE = 16


class MarianTranslator:
    def __init__(self, model_candidates: List[str]):
        self.model: Optional[MarianMTModel] = None
        self.tokenizer: Optional[MarianTokenizer] = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        for name in model_candidates:
            try:
                print(f"  Trying model: {name} ...")
                self.tokenizer = MarianTokenizer.from_pretrained(name)
                self.model = MarianMTModel.from_pretrained(name)
                self.model.to(self.device).eval()
                print(f"  ✅ Loaded {name} on {self.device}")
                self.model_name = name
                return
            except Exception as e:
                print(f"  ⚠️  {name} not available: {e}")

        raise RuntimeError(
            f"None of the candidate models could be loaded: {model_candidates}"
        )

    def translate(self, texts: List[str], batch_size: int = BATCH_SIZE) -> List[str]:
        """Translate texts in batches with OOM protection."""
        translations: List[str] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            try:
                translated = self._translate_batch(batch)
                translations.extend(translated)
            except RuntimeError as e:
                if "out of memory" in str(e).lower() and batch_size > 1:
                    print(f"  OOM at batch_size={batch_size}, retrying with batch_size=1")
                    torch.cuda.empty_cache()
                    for single in batch:
                        try:
                            translations.extend(self._translate_batch([single]))
                        except Exception:
                            translations.append("")
                else:
                    raise

        return translations

    def _translate_batch(self, batch: List[str]) -> List[str]:
        inputs = self.tokenizer(
            batch, return_tensors="pt", padding=True,
            truncation=True, max_length=512,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, max_length=512)

        return self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)


# ─── Lazy singletons ─────────────────────────────────────────────────────────
_translator_en_sk: Optional[MarianTranslator] = None
_translator_sk_en: Optional[MarianTranslator] = None


def translate_en_sk(texts: List[str]) -> List[str]:
    global _translator_en_sk
    if _translator_en_sk is None:
        _translator_en_sk = MarianTranslator(MODELS["en-sk"])
    return _translator_en_sk.translate(texts)


def translate_sk_en(texts: List[str]) -> List[str]:
    global _translator_sk_en
    if _translator_sk_en is None:
        _translator_sk_en = MarianTranslator(MODELS["sk-en"])
    return _translator_sk_en.translate(texts)


def main() -> None:
    print("=" * 60)
    print("MarianMT Translation Evaluation")
    print("=" * 60)

    print("\n[1/2] Evaluating EN → SK...")
    en_sk_results = evaluate_files(translate_en_sk, "EN_SK", "EN→SK")

    print("\n[2/2] Evaluating SK → EN...")
    sk_en_results = evaluate_files(translate_sk_en, "SK_EN", "SK→EN")

    save_results("03_marianmt", en_sk_results, sk_en_results)

    # Free GPU memory
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

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