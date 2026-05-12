"""
Backtranslation Translation Evaluation
Translates text, then translates back to source language to verify quality.
Selects the best candidate based on backtranslation similarity.
"""
from typing import List, Tuple, Optional

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from difflib import SequenceMatcher
from utils import evaluate_files, save_results, get_forced_bos_id

MODEL_NAME = "facebook/nllb-200-distilled-600M"

LANG_CODES = {
    "en": "eng_Latn",
    "sk": "slk_Latn",
}

BATCH_SIZE = 4
NUM_CANDIDATES = 3


class BacktranslationTranslator:
    def __init__(self):
        print(f"Loading translation model: {MODEL_NAME}...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device).eval()
        print(f"Model loaded on {self.device}")

        # Pre-compute forced BOS ids
        self._forced_bos = {
            lang: get_forced_bos_id(self.tokenizer, code)
            for lang, code in LANG_CODES.items()
        }

    def _translate_batch_raw(
        self,
        texts: List[str],
        source_lang: str,
        target_lang: str,
        num_beams: int = 5,
        num_return: int = 1,
    ) -> List[str]:
        """Translate a single batch (no sub-batching). Returns flat list."""
        self.tokenizer.src_lang = LANG_CODES[source_lang]

        inputs = self.tokenizer(
            texts, return_tensors="pt", padding=True,
            truncation=True, max_length=512,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                forced_bos_token_id=self._forced_bos[target_lang],
                max_length=512,
                num_beams=num_beams,
                num_return_sequences=num_return,
            )

        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

    def translate_batch(
        self,
        texts: List[str],
        source_lang: str,
        target_lang: str,
        num_beams: int = 5,
        num_return: int = 1,
        batch_size: int = BATCH_SIZE,
    ) -> List[List[str]]:
        """Translate texts in safe sub-batches. Returns grouped candidates."""
        all_grouped: List[List[str]] = []

        for i in range(0, len(texts), batch_size):
            sub = texts[i : i + batch_size]
            try:
                flat = self._translate_batch_raw(
                    sub, source_lang, target_lang, num_beams, num_return,
                )
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"  OOM at batch_size={len(sub)}, retrying one-by-one")
                    torch.cuda.empty_cache()
                    flat = []
                    for single in sub:
                        try:
                            flat.extend(self._translate_batch_raw(
                                [single], source_lang, target_lang, num_beams, num_return,
                            ))
                        except Exception:
                            flat.extend([""] * num_return)
                else:
                    raise

            # Group by input text
            for j in range(len(sub)):
                group = flat[j * num_return : (j + 1) * num_return]
                all_grouped.append(group)

        return all_grouped

    @staticmethod
    def calculate_similarity(text1: str, text2: str) -> float:
        """Calculate similarity between two texts using SequenceMatcher."""
        if not text1 or not text2:
            return 0.0
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

    def backtranslate_and_score(
        self,
        original: str,
        candidates: List[str],
        source_lang: str,
        target_lang: str,
    ) -> Tuple[str, float]:
        """Backtranslate candidates and select the best one."""
        if not candidates:
            return "", 0.0

        # Filter empty candidates
        valid = [(idx, c) for idx, c in enumerate(candidates) if c.strip()]
        if not valid:
            return candidates[0], 0.0

        valid_texts = [c for _, c in valid]

        # Backtranslate: target_lang → source_lang
        backtranslations = self.translate_batch(
            valid_texts, target_lang, source_lang, num_beams=3, num_return=1,
        )
        bt_flat = [bt[0] if bt else "" for bt in backtranslations]

        # Calculate similarity scores
        similarities = [
            self.calculate_similarity(original, bt) for bt in bt_flat
        ]

        best_local_idx = max(range(len(similarities)), key=lambda k: similarities[k])
        best_translation = valid_texts[best_local_idx]
        best_score = similarities[best_local_idx]

        return best_translation, best_score

    def translate_with_backtranslation(
        self,
        texts: List[str],
        source_lang: str,
        target_lang: str,
    ) -> List[str]:
        """Translate texts using backtranslation for quality control."""
        # Step 1: generate candidates for all texts
        candidates_grouped = self.translate_batch(
            texts, source_lang, target_lang,
            num_beams=5, num_return=NUM_CANDIDATES,
        )

        # Step 2: select best via backtranslation
        translations: List[str] = []
        for original, candidates in zip(texts, candidates_grouped):
            best, _ = self.backtranslate_and_score(
                original, candidates, source_lang, target_lang,
            )
            translations.append(best)

        return translations


# ─── Lazy singleton ──────────────────────────────────────────────────────────
_translator: Optional[BacktranslationTranslator] = None


def _get_translator() -> BacktranslationTranslator:
    global _translator
    if _translator is None:
        _translator = BacktranslationTranslator()
    return _translator


def translate_en_sk(texts: List[str]) -> List[str]:
    return _get_translator().translate_with_backtranslation(texts, "en", "sk")


def translate_sk_en(texts: List[str]) -> List[str]:
    return _get_translator().translate_with_backtranslation(texts, "sk", "en")


def main() -> None:
    print("=" * 60)
    print("Backtranslation Translation Evaluation")
    print("=" * 60)
    print("\nNote: Generates multiple candidates and selects the best one")
    print("based on backtranslation similarity. Slower but higher quality.\n")

    print("[1/2] Evaluating EN → SK...")
    en_sk_results = evaluate_files(translate_en_sk, "EN_SK", "EN→SK")

    print("\n[2/2] Evaluating SK → EN...")
    sk_en_results = evaluate_files(translate_sk_en, "SK_EN", "SK→EN")

    save_results("07_backtranslation", en_sk_results, sk_en_results)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

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