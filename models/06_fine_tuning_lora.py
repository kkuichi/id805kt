"""
Fine-tuning with LoRA Translation Evaluation
Uses LoRA (Low-Rank Adaptation) fine-tuned models on NLLB base.

If LoRA adapters are not found, falls back to the base model.
"""
import os
from pathlib import Path
from typing import List, Dict, Optional

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from utils import evaluate_files, save_results, get_forced_bos_id

_MODELS_DIR = Path(__file__).resolve().parent
LORA_ADAPTERS = {
    "en-sk": str(_MODELS_DIR / "lora_adapters" / "en_sk"),
    "sk-en": str(_MODELS_DIR / "lora_adapters" / "sk_en"),
}

BASE_MODEL = "facebook/nllb-200-distilled-600M"

NLLB_LANG_CODES: Dict[str, str] = {
    "en": "eng_Latn",
    "sk": "slk_Latn",
}

BATCH_SIZE = 8


class LoRATranslator:
    def __init__(self, adapter_path: str, source_lang: str, target_lang: str):
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.has_lora = False

        print(f"Loading base model: {BASE_MODEL}...")
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        base_model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)

        if os.path.exists(adapter_path):
            try:
                from peft import PeftModel
                print(f"Loading LoRA adapter from: {adapter_path}")
                self.model = PeftModel.from_pretrained(base_model, adapter_path)
                self.has_lora = True
                print("  ✅ LoRA adapter loaded successfully")
            except ImportError:
                print("  ⚠️  'peft' not installed. Run: pip install peft")
                print("  Falling back to base model.")
                self.model = base_model
            except Exception as e:
                print(f"  ⚠️  Error loading LoRA adapter: {e}")
                print("  Falling back to base model.")
                self.model = base_model
        else:
            print(f"  ⚠️  LoRA adapter not found at: {adapter_path}")
            print("  Using base model. Train adapters first for better results.")
            self.model = base_model

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device).eval()
        print(f"  Model on {self.device} (LoRA: {self.has_lora})")

        # Pre-compute forced BOS
        tgt_code = NLLB_LANG_CODES[self.target_lang]
        self._forced_bos = get_forced_bos_id(self.tokenizer, tgt_code)
        self._src_code = NLLB_LANG_CODES[self.source_lang]

    def translate(self, texts: List[str], batch_size: int = BATCH_SIZE) -> List[str]:
        """Translate texts with OOM protection."""
        translations: List[str] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            try:
                translated = self._translate_batch(batch)
                translations.extend(translated)
            except RuntimeError as e:
                if "out of memory" in str(e).lower() and batch_size > 1:
                    print(f"  OOM at batch_size={batch_size}, retrying one-by-one")
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
        self.tokenizer.src_lang = self._src_code

        inputs = self.tokenizer(
            batch, return_tensors="pt", padding=True,
            truncation=True, max_length=512,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                forced_bos_token_id=self._forced_bos,
                max_length=512,
                num_beams=5,
            )

        return self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)


# ─── Lazy singletons ─────────────────────────────────────────────────────────
_translator_en_sk: Optional[LoRATranslator] = None
_translator_sk_en: Optional[LoRATranslator] = None


def translate_en_sk(texts: List[str]) -> List[str]:
    global _translator_en_sk
    if _translator_en_sk is None:
        _translator_en_sk = LoRATranslator(LORA_ADAPTERS["en-sk"], "en", "sk")
    return _translator_en_sk.translate(texts)


def translate_sk_en(texts: List[str]) -> List[str]:
    global _translator_sk_en
    if _translator_sk_en is None:
        _translator_sk_en = LoRATranslator(LORA_ADAPTERS["sk-en"], "sk", "en")
    return _translator_sk_en.translate(texts)


def print_training_instructions() -> None:
    print("""
╔══════════════════════════════════════════════════════════╗
║              LoRA Fine-tuning Instructions               ║
╠══════════════════════════════════════════════════════════╣
║                                                          ║
║  1. pip install peft datasets                            ║
║                                                          ║
║  2. Prepare parallel data (source TAB target)            ║
║                                                          ║
║  3. Training script:                                     ║
║     from peft import get_peft_model, LoraConfig, TaskType║
║                                                          ║
║     lora_config = LoraConfig(                            ║
║         r=16, lora_alpha=32,                             ║
║         target_modules=["q_proj", "v_proj"],             ║
║         lora_dropout=0.05,                               ║
║         task_type=TaskType.SEQ_2_SEQ_LM,                 ║
║     )                                                    ║
║     model = get_peft_model(base_model, lora_config)      ║
║     # ... train ...                                      ║
║     model.save_pretrained("./lora_adapters/en_sk")       ║
║                                                          ║
║  4. Update LORA_ADAPTERS paths in this script.           ║
╚══════════════════════════════════════════════════════════╝
""")


def main() -> None:
    print("=" * 60)
    print("Fine-tuning (LoRA) Translation Evaluation")
    print("=" * 60)

    has_any_adapter = any(os.path.exists(p) for p in LORA_ADAPTERS.values())
    if not has_any_adapter:
        print("\n⚠️  No LoRA adapters found. Using base model for evaluation.")
        print_training_instructions()

    print("\n[1/2] Evaluating EN → SK...")
    en_sk_results = evaluate_files(translate_en_sk, "EN_SK", "EN→SK")

    print("\n[2/2] Evaluating SK → EN...")
    sk_en_results = evaluate_files(translate_sk_en, "SK_EN", "SK→EN")

    save_results("06_fine_tuning_lora", en_sk_results, sk_en_results)

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