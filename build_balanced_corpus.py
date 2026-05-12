# build_balanced_corpus.py
# Krok 3: Filtrovanie a vyvažovanie korpusu
# Vstup:  dataset_SK_EN.tsv / dataset_EN_SK.tsv
# Výstup: filtered_SK_EN.tsv / filtered_EN_SK.tsv

import unicodedata
from pathlib import Path
from typing import List, Tuple, Dict
from collections import Counter

# ── Konfigurácia ──────────────────────────────────────────────────────────────
IN_SK_EN  = Path("dataset_SK_EN.tsv")
IN_EN_SK  = Path("dataset_EN_SK.tsv")
OUT_SK_EN = Path("filtered_SK_EN.tsv")
OUT_EN_SK = Path("filtered_EN_SK.tsv")

# Filtre dĺžky (počet slov)
MIN_WORDS = 3
MAX_WORDS = 80

# Maximálny pomer dĺžok src/tgt (napr. 3.0 = tgt max 3× dlhší ako src)
MAX_LENGTH_RATIO = 3.0

# Vyváženie: maximálny počet párov na korpus (None = bez limitu)
MAX_PER_CORPUS = None  # napr. 15_000 pre striktné vyváženie


# ── Pomocné funkcie ───────────────────────────────────────────────────────────
def load_tsv(path: Path) -> List[Tuple[str, str, str]]:
    """Načíta TSV s kolonami (col0, col1, korpus). Vráti zoznam trojíc."""
    if not path.exists():
        raise FileNotFoundError(f"Súbor neexistuje: {path}")

    rows: List[Tuple[str, str, str]] = []
    skipped = 0

    with open(path, "r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, 1):
            line = unicodedata.normalize("NFC", raw.strip())
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                skipped += 1
                continue

            col0 = parts[0].strip()
            col1 = parts[1].strip()
            corpus = parts[2].strip() if len(parts) >= 3 else "unknown"

            if col0 and col1:
                rows.append((col0, col1, corpus))
            else:
                skipped += 1

    if skipped:
        print(f"  ⚠️  Preskočené riadky (chybné): {skipped}")
    return rows


def word_count(text: str) -> int:
    return len(text.split())


def is_valid_pair(src: str, tgt: str) -> bool:
    """Vráti True ak pár spĺňa všetky filtrovacie kritériá."""
    src_w = word_count(src)
    tgt_w = word_count(tgt)

    # Filter dĺžky
    if src_w < MIN_WORDS or src_w > MAX_WORDS:
        return False
    if tgt_w < MIN_WORDS or tgt_w > MAX_WORDS:
        return False

    # Filter pomeru dĺžok
    if src_w > 0 and tgt_w > 0:
        ratio = max(src_w, tgt_w) / min(src_w, tgt_w)
        if ratio > MAX_LENGTH_RATIO:
            return False

    return True


def filter_and_balance(
    rows: List[Tuple[str, str, str]],
    max_per_corpus: int | None = None,
) -> Tuple[List[Tuple[str, str, str]], Dict]:
    """
    1. Filtruje páry podľa kritérií dĺžky.
    2. Voliteľne limituje počet párov na korpus (vyváženie).
    Vracia (filtered_rows, stats).
    """
    corpus_counters: Counter = Counter()
    filtered: List[Tuple[str, str, str]] = []

    removed_length = 0
    removed_ratio  = 0
    removed_balance = 0

    for src, tgt, corpus in rows:
        src_w = word_count(src)
        tgt_w = word_count(tgt)

        # Dĺžkový filter
        if src_w < MIN_WORDS or src_w > MAX_WORDS or tgt_w < MIN_WORDS or tgt_w > MAX_WORDS:
            removed_length += 1
            continue

        # Pomer dĺžok
        ratio = max(src_w, tgt_w) / min(src_w, tgt_w) if min(src_w, tgt_w) > 0 else 999
        if ratio > MAX_LENGTH_RATIO:
            removed_ratio += 1
            continue

        # Vyváženie (voliteľné)
        if max_per_corpus is not None and corpus_counters[corpus] >= max_per_corpus:
            removed_balance += 1
            continue

        corpus_counters[corpus] += 1
        filtered.append((src, tgt, corpus))

    stats = {
        "total_in":       len(rows),
        "total_out":      len(filtered),
        "removed_length": removed_length,
        "removed_ratio":  removed_ratio,
        "removed_balance": removed_balance,
        "per_corpus":     dict(corpus_counters),
    }
    return filtered, stats


def save_tsv(rows: List[Tuple[str, str, str]], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for col0, col1, corpus in rows:
            f.write(f"{col0}\t{col1}\t{corpus}\n")


def print_stats(label: str, stats: Dict) -> None:
    print(f"\n📊 {label}")
    print(f"  Vstupných párov:          {stats['total_in']:>10,}")
    print(f"  Vyfiltrovaných (dĺžka):   {stats['removed_length']:>10,}")
    print(f"  Vyfiltrovaných (pomer):   {stats['removed_ratio']:>10,}")
    print(f"  Vyfiltrovaných (balance): {stats['removed_balance']:>10,}")
    print(f"  Výstupných párov:         {stats['total_out']:>10,}")
    if stats["per_corpus"]:
        print("  Rozloženie po korpusoch:")
        for corpus, cnt in sorted(stats["per_corpus"].items()):
            print(f"    {corpus:<20} {cnt:>8,}")


# ── Hlavná funkcia ────────────────────────────────────────────────────────────
def main() -> None:
    print("=" * 60)
    print("🔧 FILTROVANIE A VYVAŽOVANIE KORPUSU")
    print(f"   Min slov: {MIN_WORDS}  |  Max slov: {MAX_WORDS}")
    print(f"   Max pomer dĺžok: {MAX_LENGTH_RATIO}")
    if MAX_PER_CORPUS:
        print(f"   Max párov na korpus: {MAX_PER_CORPUS:,}")
    print("=" * 60)

    # ── SK_EN ──
    print("\n📥 Načítavam dataset_SK_EN.tsv ...")
    sk_en_rows = load_tsv(IN_SK_EN)
    print(f"  Načítaných párov: {len(sk_en_rows):,}")

    filtered_sk_en, stats_sk_en = filter_and_balance(sk_en_rows, MAX_PER_CORPUS)
    print_stats("SK → EN", stats_sk_en)

    save_tsv(filtered_sk_en, OUT_SK_EN)
    print(f"\n✅ Uložené: {OUT_SK_EN}  ({len(filtered_sk_en):,} párov)")

    # ── EN_SK ──
    print("\n📥 Načítavam dataset_EN_SK.tsv ...")
    en_sk_rows = load_tsv(IN_EN_SK)
    print(f"  Načítaných párov: {len(en_sk_rows):,}")

    filtered_en_sk, stats_en_sk = filter_and_balance(en_sk_rows, MAX_PER_CORPUS)
    print_stats("EN → SK", stats_en_sk)

    save_tsv(filtered_en_sk, OUT_EN_SK)
    print(f"\n✅ Uložené: {OUT_EN_SK}  ({len(filtered_en_sk):,} párov)")

    print("\n" + "=" * 60)
    print("✅ HOTOVO! Teraz môžeš spustiť:")
    print("   python make_eval_samples.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
