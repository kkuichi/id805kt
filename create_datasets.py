# create_datasets.py
# Vytvorí dataset_SK_EN.tsv a dataset_EN_SK.tsv
# z 5 paralelných korpusov (po 20 000 riadkov z každého = 100 000 celkovo)

import random
import unicodedata
from pathlib import Path
from typing import List, Tuple

SEED = 42
PER_CORPUS = 20_000
DATASETS_DIR = Path("datasets")

CORPORA = [
    ("CCMatrix",       "CCMatrix.en-sk"),
    ("Europarl",       "Europarl.en-sk"),
    ("OpenSubtitles",  "OpenSubtitles.en-sk"),
    ("ParaCrawl",      "ParaCrawl.en-sk"),
    ("WikiMatrix",     "WikiMatrix.en-sk"),
]


def load_parallel(corpus_dir: Path, prefix: str) -> List[Tuple[str, str]]:
    """Načíta .en a .sk súbory, vráti zoznam (en, sk) párov."""
    en_path = corpus_dir / f"{prefix}.en"
    sk_path = corpus_dir / f"{prefix}.sk"

    if not en_path.exists():
        raise FileNotFoundError(f"Chýba: {en_path}")
    if not sk_path.exists():
        raise FileNotFoundError(f"Chýba: {sk_path}")

    with open(en_path, "r", encoding="utf-8") as f_en, \
         open(sk_path, "r", encoding="utf-8") as f_sk:
        en_lines = f_en.readlines()
        sk_lines = f_sk.readlines()

    if len(en_lines) != len(sk_lines):
        print(f"  ⚠️  {prefix}: rôzny počet riadkov EN={len(en_lines)}, SK={len(sk_lines)}")
        min_len = min(len(en_lines), len(sk_lines))
        en_lines = en_lines[:min_len]
        sk_lines = sk_lines[:min_len]

    pairs = []
    for en, sk in zip(en_lines, sk_lines):
        en = unicodedata.normalize("NFC", en.strip())
        sk = unicodedata.normalize("NFC", sk.strip())
        if en and sk:
            pairs.append((en, sk))

    return pairs


def sample_pairs(
    pairs: List[Tuple[str, str]], n: int, rng: random.Random
) -> List[Tuple[str, str]]:
    """Vyberie n náhodných párov. Ak je menej, vráti všetky."""
    if len(pairs) <= n:
        print(f"  ℹ️  Málo riadkov ({len(pairs)}), beriem všetky")
        return pairs
    return rng.sample(pairs, n)


def main() -> None:
    print("=" * 60)
    print("📦 VYTVORENIE DATASETOV (5 × 20 000 = 100 000 párov)")
    print("=" * 60)

    rng = random.Random(SEED)

    all_pairs: List[Tuple[str, str, str]] = []  # (en, sk, korpus)

    for corpus_name, prefix in CORPORA:
        corpus_dir = DATASETS_DIR / corpus_name
        print(f"\n📥 {corpus_name}:")

        pairs = load_parallel(corpus_dir, prefix)
        print(f"  Celkovo platných párov: {len(pairs):,}")

        selected = sample_pairs(pairs, PER_CORPUS, rng)
        print(f"  Vybraných: {len(selected):,}")

        for en, sk in selected:
            all_pairs.append((en, sk, corpus_name))

    # Zamiešať
    rng.shuffle(all_pairs)

    print(f"\n📊 Celkovo párov: {len(all_pairs):,}")

    # Uložiť dataset_SK_EN.tsv (sk → en, stĺpce: sk\ten\tkorpus)
    sk_en_path = Path("dataset_SK_EN.tsv")
    with open(sk_en_path, "w", encoding="utf-8") as f:
        for en, sk, corpus in all_pairs:
            f.write(f"{sk}\t{en}\t{corpus}\n")
    print(f"✅ Uložené: {sk_en_path} ({len(all_pairs):,} riadkov)")

    # Uložiť dataset_EN_SK.tsv (en → sk, stĺpce: en\tsk\tkorpus)
    en_sk_path = Path("dataset_EN_SK.tsv")
    with open(en_sk_path, "w", encoding="utf-8") as f:
        for en, sk, corpus in all_pairs:
            f.write(f"{en}\t{sk}\t{corpus}\n")
    print(f"✅ Uložené: {en_sk_path} ({len(all_pairs):,} riadkov)")

    print("\n" + "=" * 60)
    print("✅ HOTOVO! Teraz môžeš spustiť:")
    print("   python stats_datasets100k.py")
    print("=" * 60)


if __name__ == "__main__":
    main()