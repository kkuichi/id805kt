"""
NLLB / M2M100 Translation Evaluation
Uses Meta's NLLB (No Language Left Behind) and M2M100 models
"""
from typing import List, Dict, Optional

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from utils import evaluate_files, save_results, get_forced_bos_id

MODELS = {
    "nllb": "facebook/nllb-200-distilled-600M",
    "m2m100": "facebook/m2m100_418M",
}

LANG_CODES: Dict[str, Dict[str, str]] = {
    "nllb": {"en": "eng_Latn", "sk": "slk_Latn"},
    "m2m100": {"en": "en", "sk": "sk"},
}

BATCH_SIZE = 8


class MultilingualTranslator:
    def __init__(self, model_type: str = "nllb"):
        if model_type not in MODELS:
            raise ValueError(f"Unknown model_type '{model_type}'. Choose from: {list(MODELS)}")

        self.model_type = model_type
        model_name = MODELS[model_type]

        print(f"Loading {model_type.upper()} model: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device).eval()
        print(f"Model loaded on {self.device}")

    def translate(
        self,
        texts: List[str],
        source_lang: str,
        target_lang: str,
        batch_size: int = BATCH_SIZE,
    ) -> List[str]:
        """Translate texts with OOM protection."""
        src_code = LANG_CODES[self.model_type][source_lang]
        tgt_code = LANG_CODES[self.model_type][target_lang]
        forced_bos = get_forced_bos_id(self.tokenizer, tgt_code)

        translations: List[str] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            try:
                translated = self._translate_batch(batch, src_code, forced_bos)
                translations.extend(translated)
            except RuntimeError as e:
                if "out of memory" in str(e).lower() and batch_size > 1:
                    print(f"  OOM at batch_size={batch_size}, retrying one-by-one")
                    torch.cuda.empty_cache()
                    for single in batch:
                        try:
                            translations.extend(
                                self._translate_batch([single], src_code, forced_bos)
                            )
                        except Exception:
                            translations.append("")
                else:
                    raise

        return translations

    def _translate_batch(
        self, batch: List[str], src_code: str, forced_bos: int,
    ) -> List[str]:
        self.tokenizer.src_lang = src_code

        inputs = self.tokenizer(
            batch, return_tensors="pt", padding=True,
            truncation=True, max_length=512,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                forced_bos_token_id=forced_bos,
                max_length=512,
            )

        return self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)


# ─── Lazy singleton ──────────────────────────────────────────────────────────
_translator: Optional[MultilingualTranslator] = None


def _get_translator() -> MultilingualTranslator:
    global _translator
    if _translator is None:
        _translator = MultilingualTranslator("nllb")  # Change to "m2m100" if needed
    return _translator


def translate_en_sk(texts: List[str]) -> List[str]:
    return _get_translator().translate(texts, "en", "sk")


def translate_sk_en(texts: List[str]) -> List[str]:
    return _get_translator().translate(texts, "sk", "en")


def main() -> None:
    print("=" * 60)
    print("NLLB/M2M100 Translation Evaluation")
    print("=" * 60)

    print("\n[1/2] Evaluating EN → SK...")
    en_sk_results = evaluate_files(translate_en_sk, "EN_SK", "EN→SK")

    print("\n[2/2] Evaluating SK → EN...")
    sk_en_results = evaluate_files(translate_sk_en, "SK_EN", "SK→EN")

    save_results("05_nllb_m2m100", en_sk_results, sk_en_results)

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