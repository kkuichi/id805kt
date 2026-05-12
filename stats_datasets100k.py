# stats_datasets100k_sk.py
# Štatistiky + vizualizácie pre 2 paralelné TSV datasety (SK→EN a EN→SK)
# Autor: Ilarion (BP)
# Dátum: 2026-01-29

import os
import re
import sys
import unicodedata
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Regexy (základná kontrola kvality)
# -----------------------------
URL_RE = re.compile(r'https?://|www\.', re.IGNORECASE)
EMAIL_RE = re.compile(r'\b[\w.+-]+@[\w-]+\.[\w.-]+\b', re.IGNORECASE)
NON_ASCII_RE = re.compile(r'[^\x00-\x7F]')
# Veľké aj malé písmená (SK + EN) na začiatku + koniec .?!
SENT_RE = re.compile(
    r'^[A-Za-zÁÄČĎÉÍĹĽŇÓÔŔŠŤÚÝŽáäčďéíĺľňóôŕšťúýž].*[.!?]$'
)

RNG = np.random.RandomState(42)  # reprodukovateľnosť scatter subsamplingu


def is_full_sentence(t: str) -> bool:
    if not t:
        return False
    return bool(SENT_RE.match(t.strip()))


def safe_split_words(t: str) -> int:
    t = (t or "").strip()
    if not t:
        return 0
    return len(t.split())


def digits_signature(t: str) -> str:
    return "".join(re.findall(r"\d", t or ""))


# -----------------------------
# Načítanie TSV
# -----------------------------
def load_tsv(path: str) -> pd.DataFrame:
    """
    Podporuje TSV s:
    - 2 stĺpcami: source<TAB>target
    - 3 stĺpcami: source<TAB>target<TAB>corpus (voliteľné)
    """
    path = str(path)
    if not Path(path).exists():
        raise FileNotFoundError(f"Nenašiel sa súbor: {path}")

    df = pd.read_csv(
        path, sep="\t", header=None, quoting=3,
        encoding="utf-8", on_bad_lines="skip",
    )
    if df.shape[1] < 2:
        raise ValueError(f"Súbor {path} nemá aspoň 2 stĺpce (source/target).")

    cols = ["zdrojovy_text", "cielovy_text"]
    if df.shape[1] >= 3:
        cols.append("korpus")
        df = df.iloc[:, :3]
    else:
        df = df.iloc[:, :2]

    df.columns = cols

    # NFC normalizácia + čistenie NaN
    for col in ("zdrojovy_text", "cielovy_text"):
        df[col] = df[col].fillna("").astype(str).apply(
            lambda x: unicodedata.normalize("NFC", x.strip())
        )
    if "korpus" in df.columns:
        df["korpus"] = df["korpus"].fillna("neznamy").astype(str)

    # Odstrániť prázdne riadky
    df = df[
        (df["zdrojovy_text"].str.len() > 0)
        & (df["cielovy_text"].str.len() > 0)
    ].reset_index(drop=True)

    return df


# -----------------------------
# Pomocná funkcia: štatistiky poľa
# -----------------------------
def _array_stats(arr: np.ndarray) -> Dict[str, float]:
    """Vypočíta základné štatistiky raz pre celé pole."""
    if len(arr) == 0:
        return {
            "min": 0, "p25": 0.0, "median": 0.0,
            "p75": 0.0, "p95": 0.0, "max": 0,
            "mean": 0.0, "std": 0.0,
        }
    return {
        "min": int(np.min(arr)),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.median(arr)),
        "p75": float(np.percentile(arr, 75)),
        "p95": float(np.percentile(arr, 95)),
        "max": int(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
    }


# -----------------------------
# Výpočet štatistík
# -----------------------------
def compute_stats(df: pd.DataFrame, nazov: str) -> Dict:
    src = df["zdrojovy_text"].astype(str)
    tgt = df["cielovy_text"].astype(str)

    src_chars = src.str.len().to_numpy()
    tgt_chars = tgt.str.len().to_numpy()

    src_words = src.apply(safe_split_words).to_numpy()
    tgt_words = tgt.apply(safe_split_words).to_numpy()

    ratio_words = src_words / np.maximum(1, tgt_words)
    ratio_chars = src_chars / np.maximum(1, tgt_chars)

    # Štatistiky — vypočítané raz
    sc = _array_stats(src_chars)
    tc = _array_stats(tgt_chars)
    sw = _array_stats(src_words)
    tw = _array_stats(tgt_words)

    # Kvalitatívne flagy
    src_has_url = src.str.contains(URL_RE)
    tgt_has_url = tgt.str.contains(URL_RE)
    src_has_email = src.str.contains(EMAIL_RE)
    tgt_has_email = tgt.str.contains(EMAIL_RE)
    src_has_ellipsis = src.str.contains(r"\.\.\.", regex=True)
    tgt_has_ellipsis = tgt.str.contains(r"\.\.\.", regex=True)

    tgt_non_ascii = tgt.str.contains(NON_ASCII_RE)

    src_full_sent = src.apply(is_full_sentence)
    tgt_full_sent = tgt.apply(is_full_sentence)

    # Digitový podpis (alignment check)
    src_digits = src.apply(digits_signature)
    tgt_digits = tgt.apply(digits_signature)
    digits_mismatch = (src_digits != tgt_digits) & (src_digits != "") & (tgt_digits != "")

    def pct(x: float) -> float:
        return round(float(x) * 100.0, 2)

    out: Dict = {
        "nazov_datasetu": nazov,
        "pocet_parov": int(len(df)),

        # znaky — zdroj
        "src_min_znakov": sc["min"],
        "src_p25_znakov": round(sc["p25"], 2),
        "src_median_znakov": round(sc["median"], 2),
        "src_p75_znakov": round(sc["p75"], 2),
        "src_p95_znakov": round(sc["p95"], 2),
        "src_max_znakov": sc["max"],
        "src_priemer_znakov": round(sc["mean"], 2),
        "src_std_znakov": round(sc["std"], 2),

        # znaky — cieľ
        "tgt_min_znakov": tc["min"],
        "tgt_p25_znakov": round(tc["p25"], 2),
        "tgt_median_znakov": round(tc["median"], 2),
        "tgt_p75_znakov": round(tc["p75"], 2),
        "tgt_p95_znakov": round(tc["p95"], 2),
        "tgt_max_znakov": tc["max"],
        "tgt_priemer_znakov": round(tc["mean"], 2),
        "tgt_std_znakov": round(tc["std"], 2),

        # slová — zdroj
        "src_min_slov": sw["min"],
        "src_median_slov": round(sw["median"], 2),
        "src_priemer_slov": round(sw["mean"], 2),
        "src_p95_slov": round(sw["p95"], 2),
        "src_max_slov": sw["max"],

        # slová — cieľ
        "tgt_min_slov": tw["min"],
        "tgt_median_slov": round(tw["median"], 2),
        "tgt_priemer_slov": round(tw["mean"], 2),
        "tgt_p95_slov": round(tw["p95"], 2),
        "tgt_max_slov": tw["max"],

        # pomery
        "pomer_min_slova": round(float(np.min(ratio_words)), 4),
        "pomer_median_slova": round(float(np.median(ratio_words)), 4),
        "pomer_priemer_slova": round(float(np.mean(ratio_words)), 4),
        "pomer_p95_slova": round(float(np.percentile(ratio_words, 95)), 4),
        "pomer_max_slova": round(float(np.max(ratio_words)), 4),

        "pomer_priemer_znaky": round(float(np.mean(ratio_chars)), 4),

        # kvalita (percentá)
        "percento_src_plna_veta": pct(src_full_sent.mean()),
        "percento_tgt_plna_veta": pct(tgt_full_sent.mean()),
        "percento_url": pct((src_has_url | tgt_has_url).mean()),
        "percento_email": pct((src_has_email | tgt_has_email).mean()),
        "percento_trikrat_bodka": pct((src_has_ellipsis | tgt_has_ellipsis).mean()),
        "percento_neascii_v_target": pct(tgt_non_ascii.mean()),
        "percento_nezhoda_cisla": pct(digits_mismatch.mean()),
    }

    # Rozdelenie podľa korpusu (ak je stĺpec)
    if "korpus" in df.columns:
        vc = df["korpus"].value_counts()
        for k, v in vc.items():
            out[f"korpus_{k}_pocet"] = int(v)
            out[f"korpus_{k}_percento"] = round(int(v) / len(df) * 100.0, 2)

    return out


# -----------------------------
# Vizualizácie
# -----------------------------
def save_hist(
    arr: np.ndarray,
    title: str,
    xlabel: str,
    out_path: Path,
    bins: int = 60,
    vline: Optional[float] = None,
) -> None:
    plt.figure(figsize=(10, 6))
    plt.hist(arr, bins=bins, edgecolor="black", alpha=0.85)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Počet viet")
    if vline is not None:
        plt.axvline(vline, linestyle="--")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_boxplot(
    data_list: List[np.ndarray],
    labels: List[str],
    title: str,
    ylabel: str,
    out_path: Path,
) -> None:
    plt.figure(figsize=(10, 6))
    plt.boxplot(data_list, labels=labels, showfliers=True)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_scatter(
    x: np.ndarray,
    y: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
    out_path: Path,
    max_points: int = 20000,
) -> None:
    n = len(x)
    if n > max_points:
        idx = RNG.choice(n, size=max_points, replace=False)
        x = x[idx]
        y = y[idx]

    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, s=6, alpha=0.35)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_cdf(
    arr: np.ndarray,
    title: str,
    xlabel: str,
    out_path: Path,
) -> None:
    arr = np.sort(np.asarray(arr))
    y = np.linspace(0, 1, len(arr), endpoint=True)

    plt.figure(figsize=(10, 6))
    plt.plot(arr, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Kumulatívny podiel")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def create_plots(df: pd.DataFrame, nazov: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    src = df["zdrojovy_text"].astype(str)
    tgt = df["cielovy_text"].astype(str)

    src_chars = src.str.len().to_numpy()
    tgt_chars = tgt.str.len().to_numpy()
    src_words = src.apply(safe_split_words).to_numpy()
    tgt_words = tgt.apply(safe_split_words).to_numpy()
    ratio_words = src_words / np.maximum(1, tgt_words)

    # 1) Histogram: slová src/tgt
    save_hist(
        src_words,
        f"{nazov}: Rozdelenie dĺžky viet (zdroj) – slová",
        "Počet slov",
        out_dir / f"{nazov}_hist_src_slova.png",
        vline=float(np.mean(src_words)),
    )
    save_hist(
        tgt_words,
        f"{nazov}: Rozdelenie dĺžky viet (cieľ) – slová",
        "Počet slov",
        out_dir / f"{nazov}_hist_tgt_slova.png",
        vline=float(np.mean(tgt_words)),
    )

    # 2) Histogram: znaky src/tgt
    save_hist(
        src_chars,
        f"{nazov}: Rozdelenie dĺžky viet (zdroj) – znaky",
        "Počet znakov",
        out_dir / f"{nazov}_hist_src_znaky.png",
        vline=float(np.mean(src_chars)),
    )
    save_hist(
        tgt_chars,
        f"{nazov}: Rozdelenie dĺžky viet (cieľ) – znaky",
        "Počet znakov",
        out_dir / f"{nazov}_hist_tgt_znaky.png",
        vline=float(np.mean(tgt_chars)),
    )

    # 3) Histogram: pomer slov
    save_hist(
        ratio_words,
        f"{nazov}: Rozdelenie pomeru dĺžok (slová) – zdroj/cieľ",
        "Pomer (src_words / tgt_words)",
        out_dir / f"{nazov}_hist_pomer_slova.png",
        vline=1.0,
        bins=80,
    )

    # 4) Boxplot: porovnanie src vs tgt (slová)
    save_boxplot(
        [src_words, tgt_words],
        ["Zdrojový jazyk", "Cieľový jazyk"],
        f"{nazov}: Boxplot dĺžky viet (slová)",
        "Počet slov",
        out_dir / f"{nazov}_box_slova.png",
    )

    # 5) Scatter: slová src vs tgt
    save_scatter(
        src_words, tgt_words,
        f"{nazov}: Vzťah dĺžky viet (slová) – zdroj vs cieľ",
        "Zdroj (počet slov)", "Cieľ (počet slov)",
        out_dir / f"{nazov}_scatter_slova.png",
    )

    # 6) CDF: dĺžka viet v slovách
    save_cdf(
        src_words,
        f"{nazov}: CDF dĺžky viet (zdroj) – slová",
        "Počet slov",
        out_dir / f"{nazov}_cdf_src_slova.png",
    )
    save_cdf(
        tgt_words,
        f"{nazov}: CDF dĺžky viet (cieľ) – slová",
        "Počet slov",
        out_dir / f"{nazov}_cdf_tgt_slova.png",
    )

    # 7) Bar chart podľa korpusu
    if "korpus" in df.columns:
        vc = df["korpus"].value_counts()
        plt.figure(figsize=(10, 6))
        plt.bar(vc.index.astype(str), vc.values, edgecolor="black", alpha=0.9)
        plt.title(f"{nazov}: Rozdelenie viet podľa zdroja (korpus)")
        plt.xlabel("Korpus")
        plt.ylabel("Počet viet")
        plt.xticks(rotation=35, ha="right")
        plt.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        plt.savefig(out_dir / f"{nazov}_bar_korpusy.png", dpi=200)
        plt.close()


def compare_plots(
    df_a: pd.DataFrame,
    name_a: str,
    df_b: pd.DataFrame,
    name_b: str,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    def get_arrays(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        src = df["zdrojovy_text"].astype(str)
        tgt = df["cielovy_text"].astype(str)
        src_words = src.apply(safe_split_words).to_numpy()
        tgt_words = tgt.apply(safe_split_words).to_numpy()
        ratio_words = src_words / np.maximum(1, tgt_words)
        return src_words, tgt_words, ratio_words

    a_src_w, a_tgt_w, a_ratio = get_arrays(df_a)
    b_src_w, b_tgt_w, b_ratio = get_arrays(df_b)

    # 1) Boxplot porovnanie pomeru
    save_boxplot(
        [a_ratio, b_ratio],
        [name_a, name_b],
        "Porovnanie: Boxplot pomeru dĺžok (slová) – zdroj/cieľ",
        "Pomer (src_words / tgt_words)",
        out_dir / "POROVNANIE_box_pomer_slova.png",
    )

    # 2) Histogram porovnanie ratio
    plt.figure(figsize=(10, 6))
    plt.hist(a_ratio, bins=80, alpha=0.55, edgecolor="black", label=name_a)
    plt.hist(b_ratio, bins=80, alpha=0.55, edgecolor="black", label=name_b)
    plt.title("Porovnanie: Rozdelenie pomeru dĺžok (slová)")
    plt.xlabel("Pomer (src_words / tgt_words)")
    plt.ylabel("Počet viet")
    plt.axvline(1.0, linestyle="--")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "POROVNANIE_hist_pomer_slova.png", dpi=200)
    plt.close()

    # 3) Priemer slov (bar)
    means = [
        float(np.mean(a_src_w)),
        float(np.mean(a_tgt_w)),
        float(np.mean(b_src_w)),
        float(np.mean(b_tgt_w)),
    ]
    labels = [
        f"{name_a} src",
        f"{name_a} tgt",
        f"{name_b} src",
        f"{name_b} tgt",
    ]

    plt.figure(figsize=(10, 6))
    plt.bar(labels, means, edgecolor="black", alpha=0.9)
    plt.title("Porovnanie: Priemerná dĺžka viet (slová)")
    plt.ylabel("Priemer (počet slov)")
    plt.xticks(rotation=20, ha="right")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / "POROVNANIE_bar_priemer_slova.png", dpi=200)
    plt.close()


# -----------------------------
# Uloženie štatistík do CSV
# -----------------------------
def save_stats_csv(stats_list: List[Dict], out_path: str) -> pd.DataFrame:
    df = pd.DataFrame(stats_list)

    front = ["nazov_datasetu", "pocet_parov"]
    cols = front + [c for c in df.columns if c not in front]
    df = df[cols]

    df.to_csv(out_path, index=False, encoding="utf-8")
    return df


# -----------------------------
# MAIN
# -----------------------------
def main(
    file_a: str = "dataset_SK_EN.tsv",
    file_b: str = "dataset_EN_SK.tsv",
    out_stats: str = "statistika_korpusu_100k.csv",
    out_dir: str = "grafy_100k",
) -> None:
    print("==============================================")
    print("📊 ŠTATISTIKA A VIZUALIZÁCIE – PARALELNÝ KORPUS")
    print("==============================================")

    print(f"📥 Načítavam: {file_a}")
    df_a = load_tsv(file_a)
    print(f"✅ Načítané páry: {len(df_a):,}")

    print(f"📥 Načítavam: {file_b}")
    df_b = load_tsv(file_b)
    print(f"✅ Načítané páry: {len(df_b):,}")

    # Výpočet štatistík
    print("\n🧮 Počítam štatistiky...")
    stats_a = compute_stats(df_a, "SK→EN")
    stats_b = compute_stats(df_b, "EN→SK")

    # Uloženie CSV
    save_stats_csv([stats_a, stats_b], out_stats)
    print(f"💾 Štatistiky uložené: {out_stats}")

    # Grafy
    plot_dir = Path(out_dir)
    print("\n📈 Vytváram grafy (samostatne pre každý dataset)...")
    create_plots(df_a, "SK_EN", plot_dir)
    create_plots(df_b, "EN_SK", plot_dir)
    print(f"✅ Grafy uložené v priečinku: {plot_dir}")

    print("\n📉 Vytváram porovnávacie grafy...")
    compare_plots(df_a, "SK→EN", df_b, "EN→SK", plot_dir)
    print("✅ Porovnávacie grafy hotové.")

    print("\n==============================================")
    print("✅ HOTOVO: CSV + grafy pripravené pre bakalársku prácu.")
    print("==============================================")


if __name__ == "__main__":
    main()