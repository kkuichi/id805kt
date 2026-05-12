# stats_samples_terminal.py
# Štatistika evaluačných vzoriek – výpis do terminálu
# Autor: Ilarion (BP)

import unicodedata
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

SAMPLES_DIR = Path("eval_samples")
SEP = "\t"


def read_tsv(path: Path) -> pd.DataFrame:
    """Načíta TSV vzorku, NFC normalizácia, preskočí zlé riadky."""
    try:
        df = pd.read_csv(
            path, sep=SEP, header=None, usecols=[0, 1], names=["src", "tgt"],
            encoding="utf-8", dtype=str, keep_default_na=False,
            on_bad_lines="skip",
        )
    except Exception as e:
        raise RuntimeError(f"Chyba pri čítaní {path}: {e}")

    for col in ("src", "tgt"):
        df[col] = df[col].apply(
            lambda x: unicodedata.normalize("NFC", str(x).strip())
        )

    # Odstrániť riadky s prázdnym src alebo tgt
    df = df[(df["src"].str.len() > 0) & (df["tgt"].str.len() > 0)].reset_index(drop=True)

    return df


def n_words(s: str) -> int:
    return len(s.split()) if s else 0


def n_chars(s: str) -> int:
    return len(s) if s else 0


def basic_stats(series: pd.Series) -> Dict[str, float]:
    if series.empty:
        return {"min": 0, "median": 0.0, "mean": 0.0, "max": 0}
    return {
        "min": int(series.min()),
        "median": float(series.median()),
        "mean": float(series.mean()),
        "max": int(series.max()),
    }


def fmt_stats(label: str, st: Dict[str, float]) -> str:
    return (
        f"{label:<12} | min={st['min']:<3}  "
        f"median={st['median']:.2f}  "
        f"mean={st['mean']:.2f}  "
        f"max={st['max']:<3}"
    )


def dataset_name(filename: str) -> str:
    if filename.startswith("SK_EN_"):
        return "SK→EN"
    if filename.startswith("EN_SK_"):
        return "EN→SK"
    return "UNKNOWN"


def line() -> None:
    print("─" * 78)


def main(samples_dir: Optional[Path] = None) -> None:
    sdir = samples_dir if samples_dir is not None else SAMPLES_DIR

    if not sdir.exists():
        raise FileNotFoundError(f"Chýba priečinok: {sdir.resolve()}")

    files = sorted(sdir.glob("*.tsv"))
    if not files:
        raise FileNotFoundError(f"V {sdir.resolve()} nie sú žiadne .tsv súbory")

    print("\n📊 ŠTATISTIKA PRE EVAL SAMPLES (zrozumiteľne v termináli)")
    print("Vysvetlenie: min/median/mean/max = minimum / medián / priemer / maximum")
    line()

    total_files = 0
    total_sentences = 0

    for path in files:
        df = read_tsv(path)

        if df.empty:
            print(f"\n⚠️  {path.name} — prázdny súbor, preskakujem")
            line()
            continue

        total_files += 1
        total_sentences += len(df)

        src_words = df["src"].map(n_words)
        tgt_words = df["tgt"].map(n_words)
        src_chars = df["src"].map(n_chars)
        tgt_chars = df["tgt"].map(n_chars)

        st_src_w = basic_stats(src_words)
        st_tgt_w = basic_stats(tgt_words)
        st_src_c = basic_stats(src_chars)
        st_tgt_c = basic_stats(tgt_chars)

        print(f"\n🧾 {dataset_name(path.name)} | {path.name} | viet: {len(df)}")
        print("SLOVÁ:")
        print("  " + fmt_stats("SRC slová", st_src_w))
        print("  " + fmt_stats("TGT slová", st_tgt_w))
        print("ZNAKY:")
        print("  " + fmt_stats("SRC znaky", st_src_c))
        print("  " + fmt_stats("TGT znaky", st_tgt_c))
        line()

    print(f"\n📁 Spracované súbory: {total_files}")
    print(f"📝 Celkový počet viet: {total_sentences:,}")
    print("✅ Hotovo.\n")


if __name__ == "__main__":
    main()