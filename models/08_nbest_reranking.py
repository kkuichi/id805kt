"""
N-best Reranking Translation Evaluation
Generates N-best translation candidates and reranks them using
length, diversity, and beam-position heuristics.
"""
from typing import List, Optional

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from utils import evaluate_files, save_results, get_forced_bos_id

MODEL_NAME = "facebook/nllb-200-distilled-600M"

LANG_CODES = {
    "en": "eng_Latn",
    "sk": "slk_Latn",
}

N_BEST = 5
BATCH_SIZE = 4


class NbestReranker:
    def __init__(self, n_best: int = N_BEST):
        self.n_best = n_best

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

    def _generate_nbest_batch(
        self, texts: List[str], source_lang: str, target_lang: str,
    ) -> List[List[str]]:
        """Generate N-best candidates for a single batch."""
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
                num_beams=self.n_best * 2,
                num_return_sequences=self.n_best,
                output_scores=True,
                return_dict_in_generate=True,
            )

        translations = self.tokenizer.batch_decode(
            outputs.sequences, skip_special_tokens=True,
        )

        # Group by input text
        grouped = [
            translations[i * self.n_best : (i + 1) * self.n_best]
            for i in range(len(texts))
        ]
        return grouped

    def generate_nbest(
        self,
        texts: List[str],
        source_lang: str,
        target_lang: str,
        batch_size: int = BATCH_SIZE,
    ) -> List[List[str]]:
        """Generate N-best candidates with sub-batching and OOM protection."""
        all_grouped: List[List[str]] = []

        for i in range(0, len(texts), batch_size):
            sub = texts[i : i + batch_size]
            try:
                grouped = self._generate_nbest_batch(sub, source_lang, target_lang)
                all_grouped.extend(grouped)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"  OOM at batch_size={len(sub)}, retrying one-by-one")
                    torch.cuda.empty_cache()
                    for single in sub:
                        try:
                            grouped = self._generate_nbest_batch(
                                [single], source_lang, target_lang,
                            )
                            all_grouped.extend(grouped)
                        except Exception:
                            all_grouped.append([""])
                else:
                    raise

        return all_grouped

    @staticmethod
    def _length_penalty(source: str, translation: str) -> float:
        """Prefer translations with similar word count to source."""
        src_len = len(source.split())
        tgt_len = len(translation.split())
        if src_len == 0:
            return 0.0
        ratio = tgt_len / src_len
        return max(0.0, 1.0 - abs(1.0 - ratio))

    @staticmethod
    def _diversity_score(translation: str) -> float:
        """Prefer translations with good vocabulary diversity."""
        words = translation.split()
        if not words:
            return 0.0
        return len(set(words)) / len(words)

    @staticmethod
    def _length_adequacy(translation: str, source: str) -> float:
        """Prefer translations that are not trivially short."""
        src_words = len(source.split())
        tgt_words = len(translation.split())
        if src_words == 0:
            return 0.0
        # Normalise against source length instead of magic constant
        return min(1.0, tgt_words / max(src_words, 1))

    def rerank_candidates(self, source: str, candidates: List[str]) -> str:
        """Rerank candidates using multiple heuristics."""
        if not candidates:
            return ""
        if len(candidates) == 1:
            return candidates[0]

        scores: List[float] = []
        n = len(candidates)

        for rank, candidate in enumerate(candidates):
            length_sc = self._length_penalty(source, candidate)
            diversity_sc = self._diversity_score(candidate)
            adequacy_sc = self._length_adequacy(candidate, source)
            # Beam position: first candidate = highest beam score
            position_sc = 1.0 - (rank / n)

            total = (
                0.3 * length_sc
                + 0.2 * diversity_sc
                + 0.2 * adequacy_sc
                + 0.3 * position_sc
            )
            scores.append(total)

        best_idx = max(range(n), key=lambda k: scores[k])
        return candidates[best_idx]

    def translate_with_reranking(
        self,
        texts: List[str],
        source_lang: str,
        target_lang: str,
    ) -> List[str]:
        """Translate with N-best reranking."""
        candidates_list = self.generate_nbest(texts, source_lang, target_lang)

        translations: List[str] = []
        for source, candidates in zip(texts, candidates_list):
            best = self.rerank_candidates(source, candidates)
            translations.append(best)

        return translations


# ─── Lazy singleton ──────────────────────────────────────────────────────────
_reranker: Optional[NbestReranker] = None


def _get_reranker() -> NbestReranker:
    global _reranker
    if _reranker is None:
        _reranker = NbestReranker(n_best=N_BEST)
    return _reranker


def translate_en_sk(texts: List[str]) -> List[str]:
    return _get_reranker().translate_with_reranking(texts, "en", "sk")


def translate_sk_en(texts: List[str]) -> List[str]:
    return _get_reranker().translate_with_reranking(texts, "sk", "en")


def main() -> None:
    print("=" * 60)
    print("N-best Reranking Translation Evaluation")
    print("=" * 60)
    print("\nNote: Generates multiple candidates and reranks them using")
    print("heuristic scoring (length, diversity, position). Slower but")
    print("may produce better quality.\n")

    print("[1/2] Evaluating EN → SK...")
    en_sk_results = evaluate_files(translate_en_sk, "EN_SK", "EN→SK")

    print("\n[2/2] Evaluating SK → EN...")
    sk_en_results = evaluate_files(translate_sk_en, "SK_EN", "SK→EN")

    save_results("08_nbest_reranking", en_sk_results, sk_en_results)

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