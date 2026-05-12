# make_eval_samples.py
# Vytvorenie evaluačných vzoriek z paralelných TSV datasetov
# Autor: Ilarion (BP)

import random
import unicodedata
from pathlib import Path
from typing import List, Dict, Optional

N_SAMPLES = 10
SAMPLE_SIZE = 100
RANDOM_SEED = 42

IN_SK_EN_FILE = Path("dataset_SK_EN.tsv")
IN_EN_SK_FILE = Path("dataset_EN_SK.tsv")
OUT_DIR = Path("eval_samples")


def load_tsv(path: Path) -> List[str]:
    """Načíta TSV súbor, NFC normalizácia, preskočí prázdne riadky."""
    if not path.exists():
        raise FileNotFoundError(f"Súbor neexistuje: {path}")

    lines: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = unicodedata.normalize("NFC", raw.strip())
            if not line:
                continue
            # Overenie, že riadok má aspoň 2 stĺpce (source + target)
            parts = line.split("\t")
            if len(parts) < 2 or not parts[0].strip() or not parts[1].strip():
                continue
            lines.append(line)

    if not lines:
        raise ValueError(f"Súbor {path} neobsahuje žiadne platné riadky.")

    return lines


def make_samples(
    lines: List[str],
    prefix: str,
    n_samples: int = N_SAMPLES,
    sample_size: int = SAMPLE_SIZE,
    seed: int = RANDOM_SEED,
    out_dir: Optional[Path] = None,
) -> List[Dict]:
    """Vytvorí n_samples vzoriek po sample_size viet z lines.

    Ak je riadkov menej než n_samples * sample_size, vytvorí toľko
    vzoriek, koľko sa zmestí (posledná môže byť menšia).
    """
    if out_dir is None:
        out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    all_indices = list(range(len(lines)))
    rng.shuffle(all_indices)

    total_needed = n_samples * sample_size
    if len(all_indices) < total_needed:
        actual_samples = max(1, len(all_indices) // sample_size)
        print(
            f"  ⚠️  {prefix}: k dispozícii len {len(all_indices)} riadkov "
            f"(treba {total_needed}). Vytvorím {actual_samples} vzoriek."
        )
        n_samples = actual_samples

    logs: List[Dict] = []

    for i in range(n_samples):
        start = i * sample_size
        end = min(start + sample_size, len(all_indices))
        idxs = all_indices[start:end]

        if not idxs:
            break

        out_file = out_dir / f"{prefix}_sample_{i + 1:02d}.tsv"
        with open(out_file, "w", encoding="utf-8") as f:
            for idx in idxs:
                f.write(lines[idx] + "\n")

        logs.append({
            "dataset": prefix,
            "sample": i + 1,
            "pocet_viet": len(idxs),
            "subor": out_file.name,
        })

    return logs


def main(
    sk_en_file: Optional[Path] = None,
    en_sk_file: Optional[Path] = None,
) -> None:
    sk_en_path = sk_en_file if sk_en_file is not None else IN_SK_EN_FILE
    en_sk_path = en_sk_file if en_sk_file is not None else IN_EN_SK_FILE

    print("📥 Načítavam datasety...")
    sk_en_lines = load_tsv(sk_en_path)
    print(f"  SK_EN: {len(sk_en_lines):,} platných riadkov")

    en_sk_lines = load_tsv(en_sk_path)
    print(f"  EN_SK: {len(en_sk_lines):,} platných riadkov")

    log: List[Dict] = []
    print("\n🔀 Vytváram evaluačné vzorky...")
    log += make_samples(sk_en_lines, "SK_EN")
    log += make_samples(en_sk_lines, "EN_SK")

    print("\n✅ Evaluačné vzorky vytvorené:")
    for row in log:
        print(f"  {row['subor']:>25s}  —  {row['pocet_viet']} viet")

    print(f"\n📁 Výstupný priečinok: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()