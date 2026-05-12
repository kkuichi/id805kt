"""
T5 NMT Translation Evaluation
Uses T5/mT5 models for translation tasks

NOTE: mt5-base without fine-tuning on EN↔SK parallel data will produce
weak results. Consider replacing MODEL_NAME with a fine-tuned checkpoint
(e.g. from Hugging Face Hub) for meaningful evaluation.
"""
from typing import List, Optional

import torch
from transformers import T5ForConditionalGeneration, AutoTokenizer
from utils import evaluate_files, save_results

MODEL_NAME = "google/mt5-base"
BATCH_SIZE = 4


class T5Translator:
    def __init__(self, model_name: str = MODEL_NAME):
        print(f"Loading T5 model: {model_name}...")
        # AutoTokenizer is preferred over the deprecated T5Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)

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
        """Translate texts using T5 with OOM protection."""
        prefix = f"translate {source_lang} to {target_lang}: "
        translations: List[str] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            try:
                translated = self._translate_batch(batch, prefix)
                translations.extend(translated)
            except RuntimeError as e:
                if "out of memory" in str(e).lower() and batch_size > 1:
                    print(f"  OOM at batch_size={batch_size}, retrying one-by-one")
                    torch.cuda.empty_cache()
                    for single in batch:
                        try:
                            translations.extend(self._translate_batch([single], prefix))
                        except Exception:
                            translations.append("")
                else:
                    raise

        return translations

    def _translate_batch(self, batch: List[str], prefix: str) -> List[str]:
        inputs_text = [prefix + text for text in batch]
        inputs = self.tokenizer(
            inputs_text, return_tensors="pt", padding=True,
            truncation=True, max_length=512,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_length=512,
                num_beams=4,
            )

        return self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)


# ─── Lazy singleton ──────────────────────────────────────────────────────────
_translator: Optional[T5Translator] = None


def _get_translator() -> T5Translator:
    global _translator
    if _translator is None:
        _translator = T5Translator()
    return _translator


def translate_en_sk(texts: List[str]) -> List[str]:
    return _get_translator().translate(texts, "English", "Slovak")


def translate_sk_en(texts: List[str]) -> List[str]:
    return _get_translator().translate(texts, "Slovak", "English")


def main() -> None:
    print("=" * 60)
    print("T5 NMT Translation Evaluation")
    print("=" * 60)

    print("\n[1/2] Evaluating EN → SK...")
    en_sk_results = evaluate_files(translate_en_sk, "EN_SK", "EN→SK")

    print("\n[2/2] Evaluating SK → EN...")
    sk_en_results = evaluate_files(translate_sk_en, "SK_EN", "SK→EN")

    save_results("04_t5_nmt", en_sk_results, sk_en_results)

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