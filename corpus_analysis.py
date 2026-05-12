# corpus_analysis.py
# Komplexná analýza paralelných korpusov pre bakalársku prácu (EN ↔ SK)
#
# Pokrýva:
#   1. Celková štatistika 100k datasetov (EN→SK, SK→EN)
#   2. Breakdown podľa zdrojového korpusu (CCMatrix, Europarl, …)
#   3. Processing štatistiky (koľko párov bolo dostupných / vybraných)
#   4. Korelácie src-tgt dĺžok (Pearson + Spearman)
#   5. Slovníkové štatistiky (unique tokeny, TTR)
#   6. Porovnanie oboch smerov
#   7. Grafické výstupy + TXT report + CSV súhrn
#
# Spustenie (z koreňa projektu):
#   python corpus_analysis.py

import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:
    from scipy import stats as scipy_stats
    _SCIPY = True
except ImportError:
    scipy_stats = None
    _SCIPY = False

# ── Konfigurácia ──────────────────────────────────────────────────────────────
DATASET_SK_EN = Path("dataset_SK_EN.tsv")
DATASET_EN_SK = Path("dataset_EN_SK.tsv")
DATASETS_DIR  = Path("datasets")
OUT_DIR       = Path("grafy_analyza")
REPORT_FILE   = Path("corpus_analysis_report.txt")
CSV_FILE      = Path("corpus_analysis_stats.csv")
SEED          = 42

CORPORA = ["CCMatrix", "Europarl", "OpenSubtitles", "ParaCrawl", "WikiMatrix"]
CORPUS_COLORS = {
    "CCMatrix":       "#4C72B0",
    "Europarl":       "#DD8452",
    "OpenSubtitles":  "#55A868",
    "ParaCrawl":      "#C44E52",
    "WikiMatrix":     "#8172B2",
}

RNG = np.random.default_rng(SEED)

# ── Načítanie TSV ─────────────────────────────────────────────────────────────
def load_tsv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Súbor neexistuje: {path}")

    df = pd.read_csv(
        path, sep="\t", header=None, quoting=3,
        encoding="utf-8", on_bad_lines="skip", dtype=str,
    )

    if df.shape[1] < 2:
        raise ValueError(f"{path}: menej ako 2 stĺpce")

    cols = ["src", "tgt"] + (["corpus"] if df.shape[1] >= 3 else [])
    df = df.iloc[:, :len(cols)].copy()
    df.columns = cols

    for col in ("src", "tgt"):
        df[col] = df[col].fillna("").astype(str).apply(
            lambda x: unicodedata.normalize("NFC", x.strip())
        )
    if "corpus" in df.columns:
        df["corpus"] = df["corpus"].fillna("unknown").astype(str).str.strip()

    df = df[(df["src"].str.len() > 0) & (df["tgt"].str.len() > 0)].reset_index(drop=True)
    return df


def load_raw_corpus(corpus_name: str, prefix: str) -> Optional[pd.DataFrame]:
    """Načíta surový paralelný korpus z datasets/."""
    en_p = DATASETS_DIR / corpus_name / f"{prefix}.en"
    sk_p = DATASETS_DIR / corpus_name / f"{prefix}.sk"
    if not en_p.exists() or not sk_p.exists():
        return None
    en_lines = en_p.read_text("utf-8").splitlines()
    sk_lines = sk_p.read_text("utf-8").splitlines()
    n = min(len(en_lines), len(sk_lines))
    pairs = [
        (unicodedata.normalize("NFC", e.strip()), unicodedata.normalize("NFC", s.strip()))
        for e, s in zip(en_lines[:n], sk_lines[:n])
        if e.strip() and s.strip()
    ]
    return pd.DataFrame(pairs, columns=["src", "tgt"])


# ── Výpočet metrík ────────────────────────────────────────────────────────────
def word_counts(series: pd.Series) -> np.ndarray:
    return series.apply(lambda x: len(x.split()) if x else 0).to_numpy()


def char_counts(series: pd.Series) -> np.ndarray:
    return series.str.len().to_numpy()


def pct_stats(arr: np.ndarray) -> Dict:
    if len(arr) == 0:
        return {k: 0 for k in ("min","p25","median","mean","p75","p95","max","std")}
    return {
        "min":    int(np.min(arr)),
        "p25":    round(float(np.percentile(arr, 25)), 2),
        "median": round(float(np.median(arr)), 2),
        "mean":   round(float(np.mean(arr)), 2),
        "p75":    round(float(np.percentile(arr, 75)), 2),
        "p95":    round(float(np.percentile(arr, 95)), 2),
        "max":    int(np.max(arr)),
        "std":    round(float(np.std(arr)), 2),
    }


def ttr(series: pd.Series) -> float:
    """Type-Token Ratio: unique_tokens / total_tokens."""
    tokens = " ".join(series.tolist()).lower().split()
    if not tokens:
        return 0.0
    return round(len(set(tokens)) / len(tokens), 4)


def top_tokens(series: pd.Series, n: int = 10) -> List[Tuple[str, int]]:
    tokens = " ".join(series.tolist()).lower().split()
    return Counter(tokens).most_common(n)


def compute_correlations(src_arr: np.ndarray, tgt_arr: np.ndarray) -> Dict:
    if len(src_arr) < 3 or not _SCIPY:
        # Fallback: numpy Pearson bez p-hodnoty
        if len(src_arr) >= 3:
            pr = float(np.corrcoef(src_arr, tgt_arr)[0, 1])
        else:
            pr = None
        return {"pearson_r": round(pr, 4) if pr is not None else None,
                "pearson_p": None, "spearman_r": None, "spearman_p": None}
    pr, pp = scipy_stats.pearsonr(src_arr, tgt_arr)
    sr, sp = scipy_stats.spearmanr(src_arr, tgt_arr)
    return {
        "pearson_r":  round(float(pr), 4),
        "pearson_p":  float(pp),
        "spearman_r": round(float(sr), 4),
        "spearman_p": float(sp),
    }


def full_analysis(df: pd.DataFrame, direction: str) -> Dict:
    """Kompletná analýza jedného datasetu."""
    src_w = word_counts(df["src"])
    tgt_w = word_counts(df["tgt"])
    src_c = char_counts(df["src"])
    tgt_c = char_counts(df["tgt"])
    ratio_w = src_w / np.maximum(1, tgt_w)
    ratio_c = src_c / np.maximum(1, tgt_c)

    n = len(df)
    n_unique_src = df["src"].nunique()
    n_unique_tgt = df["tgt"].nunique()
    n_dup = n - n_unique_src

    corr_words = compute_correlations(src_w, tgt_w)
    corr_chars = compute_correlations(src_c, tgt_c)

    src_ttr = ttr(df["src"])
    tgt_ttr = ttr(df["tgt"])
    src_top = top_tokens(df["src"])
    tgt_top = top_tokens(df["tgt"])

    result = {
        "direction": direction,
        "n_pairs":      n,
        "n_unique_src": n_unique_src,
        "n_unique_tgt": n_unique_tgt,
        "n_duplicates": n_dup,
        "dup_pct":      round(n_dup / n * 100, 2) if n else 0,

        "src_words":  pct_stats(src_w),
        "tgt_words":  pct_stats(tgt_w),
        "src_chars":  pct_stats(src_c),
        "tgt_chars":  pct_stats(tgt_c),
        "ratio_words": pct_stats(ratio_w),
        "ratio_chars": pct_stats(ratio_c),

        "corr_words": corr_words,
        "corr_chars": corr_chars,

        "src_ttr": src_ttr,
        "tgt_ttr": tgt_ttr,
        "src_top_tokens": src_top,
        "tgt_top_tokens": tgt_top,

        "src_w_arr": src_w,
        "tgt_w_arr": tgt_w,
        "src_c_arr": src_c,
        "tgt_c_arr": tgt_c,
        "ratio_w_arr": ratio_w,
    }

    # Per-corpus breakdown
    if "corpus" in df.columns:
        corpus_stats = {}
        for corp in df["corpus"].unique():
            sub = df[df["corpus"] == corp]
            sw = word_counts(sub["src"])
            tw = word_counts(sub["tgt"])
            corpus_stats[corp] = {
                "count":        len(sub),
                "pct":          round(len(sub) / n * 100, 2),
                "src_mean_w":   round(float(sw.mean()), 2) if len(sw) else 0,
                "tgt_mean_w":   round(float(tw.mean()), 2) if len(tw) else 0,
                "src_median_w": round(float(np.median(sw)), 2) if len(sw) else 0,
                "tgt_median_w": round(float(np.median(tw)), 2) if len(tw) else 0,
                "src_std_w":    round(float(sw.std()), 2) if len(sw) else 0,
                "tgt_std_w":    round(float(tw.std()), 2) if len(tw) else 0,
            }
        result["corpus_stats"] = corpus_stats

    return result


# ── Načítanie surových korpusov (processing stats) ────────────────────────────
RAW_CORPORA_CONFIG = [
    ("CCMatrix",      "CCMatrix.en-sk"),
    ("Europarl",      "Europarl.en-sk"),
    ("OpenSubtitles", "OpenSubtitles.en-sk"),
    ("ParaCrawl",     "ParaCrawl.en-sk"),
    ("WikiMatrix",    "WikiMatrix.en-sk"),
]


def compute_processing_stats(df_100k: pd.DataFrame) -> List[Dict]:
    """Porovnanie: koľko párova bolo dostupných vs. vybraných v 100k datasete."""
    rows = []
    for corpus_name, prefix in RAW_CORPORA_CONFIG:
        selected = 0
        if "corpus" in df_100k.columns:
            selected = int((df_100k["corpus"] == corpus_name).sum())

        raw_df = load_raw_corpus(corpus_name, prefix)
        available = len(raw_df) if raw_df is not None else None

        rows.append({
            "corpus":       corpus_name,
            "available":    available,
            "selected":     selected,
            "sampling_pct": round(selected / available * 100, 2) if available else None,
            "raw_present":  raw_df is not None,
        })
    return rows


# ── Grafické výstupy ──────────────────────────────────────────────────────────
def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_hist_compare(arr_a: np.ndarray, arr_b: np.ndarray,
                      label_a: str, label_b: str,
                      xlabel: str, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(arr_a, bins=60, alpha=0.55, label=label_a, color="#4C72B0", edgecolor="none")
    ax.hist(arr_b, bins=60, alpha=0.55, label=label_b, color="#DD8452", edgecolor="none")
    ax.axvline(np.mean(arr_a), color="#4C72B0", linestyle="--", lw=1.5,
               label=f"μ {label_a}={np.mean(arr_a):.1f}")
    ax.axvline(np.mean(arr_b), color="#DD8452", linestyle="--", lw=1.5,
               label=f"μ {label_b}={np.mean(arr_b):.1f}")
    ax.set_xlabel(xlabel); ax.set_ylabel("Počet viet")
    ax.set_title(title); ax.legend(); ax.grid(axis="y", alpha=0.25)
    _save(fig, path)


def plot_boxplot_corpora(df: pd.DataFrame, direction: str, path: Path) -> None:
    if "corpus" not in df.columns:
        return
    corpora = [c for c in CORPORA if c in df["corpus"].unique()]
    data = [word_counts(df[df["corpus"] == c]["src"]) for c in corpora]
    colors = [CORPUS_COLORS.get(c, "#888") for c in corpora]

    fig, ax = plt.subplots(figsize=(11, 6))
    bp = ax.boxplot(data, labels=corpora, patch_artist=True, showfliers=False)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    ax.set_title(f"{direction}: Dĺžka viet podľa korpusu (slová, src)")
    ax.set_ylabel("Počet slov"); ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=20, ha="right")
    _save(fig, path)


def plot_scatter_corr(src_arr: np.ndarray, tgt_arr: np.ndarray,
                      direction: str, path: Path, max_pts: int = 15000) -> None:
    n = len(src_arr)
    if n > max_pts:
        idx = RNG.choice(n, size=max_pts, replace=False)
        x, y = src_arr[idx], tgt_arr[idx]
    else:
        x, y = src_arr, tgt_arr

    pr = float(np.corrcoef(src_arr, tgt_arr)[0, 1])
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(x, y, s=5, alpha=0.25, color="#4C72B0")
    # Fit line
    m, b = np.polyfit(src_arr, tgt_arr, 1)
    xs = np.linspace(src_arr.min(), src_arr.max(), 200)
    ax.plot(xs, m * xs + b, color="#C44E52", lw=1.5, label=f"fit (r={pr:.3f})")
    ax.set_xlabel("Src – počet slov"); ax.set_ylabel("Tgt – počet slov")
    ax.set_title(f"{direction}: Korelácia dĺžok src↔tgt (slová)")
    ax.legend(); ax.grid(alpha=0.2)
    _save(fig, path)


def plot_cdf_compare(arr_a: np.ndarray, arr_b: np.ndarray,
                     label_a: str, label_b: str,
                     xlabel: str, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    for arr, label, color in [(arr_a, label_a, "#4C72B0"), (arr_b, label_b, "#DD8452")]:
        s = np.sort(arr)
        y = np.linspace(0, 1, len(s))
        ax.plot(s, y, label=label, color=color, lw=1.8)
    ax.set_xlabel(xlabel); ax.set_ylabel("Kumulatívny podiel")
    ax.set_title(title); ax.legend(); ax.grid(alpha=0.25)
    _save(fig, path)


def plot_corpus_bar(df: pd.DataFrame, direction: str, path: Path) -> None:
    if "corpus" not in df.columns:
        return
    vc = df["corpus"].value_counts().reindex(CORPORA, fill_value=0)
    colors = [CORPUS_COLORS.get(c, "#888") for c in vc.index]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(vc.index, vc.values, color=colors, edgecolor="white", alpha=0.9)
    for bar, val in zip(bars, vc.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 200,
                f"{val:,}", ha="center", va="bottom", fontsize=9)
    ax.set_title(f"{direction}: Počet viet podľa korpusu")
    ax.set_ylabel("Počet viet"); ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=15, ha="right")
    _save(fig, path)


def plot_ratio_hist(ratio_arr: np.ndarray, direction: str, path: Path) -> None:
    clipped = np.clip(ratio_arr, 0, 5)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(clipped, bins=80, color="#55A868", edgecolor="none", alpha=0.85)
    ax.axvline(1.0, color="red", linestyle="--", lw=1.5, label="pomer = 1.0")
    ax.axvline(float(np.median(ratio_arr)), color="#4C72B0", linestyle="-.", lw=1.5,
               label=f"medián = {np.median(ratio_arr):.2f}")
    ax.set_xlabel("src_words / tgt_words (orezané na 5)")
    ax.set_ylabel("Počet viet")
    ax.set_title(f"{direction}: Rozdelenie pomeru dĺžok src/tgt (slová)")
    ax.legend(); ax.grid(axis="y", alpha=0.25)
    _save(fig, path)


def plot_mean_per_corpus_compare(df_sk_en: pd.DataFrame, df_en_sk: pd.DataFrame,
                                  path: Path) -> None:
    """Priemerná dĺžka src a tgt slov pre každý korpus, oba smery vedľa seba."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)

    for ax, df, direction in [
        (axes[0], df_sk_en, "SK→EN"),
        (axes[1], df_en_sk, "EN→SK"),
    ]:
        if "corpus" not in df.columns:
            continue
        corpora = [c for c in CORPORA if c in df["corpus"].unique()]
        src_means = [word_counts(df[df["corpus"] == c]["src"]).mean() for c in corpora]
        tgt_means = [word_counts(df[df["corpus"] == c]["tgt"]).mean() for c in corpora]
        x = np.arange(len(corpora))
        w = 0.35
        ax.bar(x - w/2, src_means, w, label="src", alpha=0.85, color="#4C72B0")
        ax.bar(x + w/2, tgt_means, w, label="tgt", alpha=0.85, color="#DD8452")
        ax.set_xticks(x); ax.set_xticklabels(corpora, rotation=20, ha="right")
        ax.set_title(f"{direction}: Priemerná dĺžka viet (slová)")
        ax.set_ylabel("Počet slov"); ax.legend(); ax.grid(axis="y", alpha=0.25)

    fig.suptitle("Priemerná dĺžka viet podľa korpusu — porovnanie smerov", fontsize=14)
    plt.tight_layout()
    _save(fig, path)


def plot_processing_stats(proc_stats: List[Dict], path: Path) -> None:
    corpora = [r["corpus"] for r in proc_stats]
    selected = [r["selected"] for r in proc_stats]
    available = [r["available"] if r["available"] is not None else 0 for r in proc_stats]
    colors = [CORPUS_COLORS.get(c, "#888") for c in corpora]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(corpora))
    w = 0.35
    if any(a > 0 for a in available):
        ax.bar(x - w/2, available, w, label="Dostupných párov (surový korpus)",
               color="#8172B2", alpha=0.75)
    ax.bar(x + w/2 if any(a > 0 for a in available) else x,
           selected, w, label="Vybraných do 100k datasetu",
           color=colors, alpha=0.9)
    ax.set_xticks(x); ax.set_xticklabels(corpora, rotation=15, ha="right")
    ax.set_title("Processing štatistiky: dostupné vs. vybraté páry")
    ax.set_ylabel("Počet párov"); ax.legend(); ax.grid(axis="y", alpha=0.25)
    _save(fig, path)


# ── Textový report ────────────────────────────────────────────────────────────
def _fmt_stats(label: str, st: Dict, unit: str = "slov") -> List[str]:
    return [
        f"    {label}:",
        f"      min={st['min']}  p25={st['p25']}  medián={st['median']}  "
        f"μ={st['mean']}  p75={st['p75']}  p95={st['p95']}  max={st['max']}  σ={st['std']}  [{unit}]",
    ]


def write_report(
    res_sk_en: Dict,
    res_en_sk: Dict,
    proc_stats: List[Dict],
    path: Path,
) -> None:
    lines: List[str] = []
    sep = "=" * 72
    sub = "-" * 72

    def section(title: str) -> None:
        lines.extend(["", sep, f"  {title}", sep])

    def sub_section(title: str) -> None:
        lines.extend(["", sub, f"  {title}", sub])

    lines.append("CORPUS ANALYSIS REPORT — EN ↔ SK Parallel Dataset")
    lines.append("Bakalárska práca — Strojový preklad")
    lines.append("")

    for res in [res_sk_en, res_en_sk]:
        d = res["direction"]
        section(f"DATASET: {d}")

        lines.append(f"  Celkový počet párov:      {res['n_pairs']:>10,}")
        lines.append(f"  Unikátnych src viet:       {res['n_unique_src']:>10,}")
        lines.append(f"  Unikátnych tgt viet:       {res['n_unique_tgt']:>10,}")
        lines.append(f"  Duplikáty (src):           {res['n_duplicates']:>10,}  ({res['dup_pct']}%)")

        sub_section("Dĺžka viet – SLOVÁ")
        lines.extend(_fmt_stats("SRC", res["src_words"], "slov"))
        lines.extend(_fmt_stats("TGT", res["tgt_words"], "slov"))

        sub_section("Dĺžka viet – ZNAKY")
        lines.extend(_fmt_stats("SRC", res["src_chars"], "znakov"))
        lines.extend(_fmt_stats("TGT", res["tgt_chars"], "znakov"))

        sub_section("Pomer dĺžok (src/tgt)")
        lines.extend(_fmt_stats("slová src/tgt", res["ratio_words"], "pomer"))
        lines.extend(_fmt_stats("znaky src/tgt", res["ratio_chars"], "pomer"))

        sub_section("Korelácia dĺžok src ↔ tgt")
        cw = res["corr_words"]
        cc = res["corr_chars"]
        lines.append(f"    Slová — Pearson r={cw['pearson_r']}  "
                     f"Spearman r={cw['spearman_r']}")
        lines.append(f"    Znaky — Pearson r={cc['pearson_r']}  "
                     f"Spearman r={cc['spearman_r']}")

        sub_section("Slovníková diverzita (Type-Token Ratio)")
        lines.append(f"    TTR src: {res['src_ttr']}  "
                     f"(nižšie = väčší vocab reuse)")
        lines.append(f"    TTR tgt: {res['tgt_ttr']}")
        lines.append(f"    Top 10 src tokenov: "
                     f"{', '.join(f'{w}({c})' for w, c in res['src_top_tokens'])}")
        lines.append(f"    Top 10 tgt tokenov: "
                     f"{', '.join(f'{w}({c})' for w, c in res['tgt_top_tokens'])}")

        if "corpus_stats" in res:
            sub_section("Breakdown podľa korpusu")
            lines.append(
                f"    {'Korpus':<18} {'Párov':>8} {'%':>6} "
                f"{'μ src slov':>12} {'μ tgt slov':>12} "
                f"{'med src':>8} {'med tgt':>8}"
            )
            lines.append("    " + "-" * 68)
            for corp in CORPORA:
                if corp not in res["corpus_stats"]:
                    continue
                cs = res["corpus_stats"][corp]
                lines.append(
                    f"    {corp:<18} {cs['count']:>8,} {cs['pct']:>6.1f}% "
                    f"{cs['src_mean_w']:>12.2f} {cs['tgt_mean_w']:>12.2f} "
                    f"{cs['src_median_w']:>8.1f} {cs['tgt_median_w']:>8.1f}"
                )

    section("PROCESSING ŠTATISTIKY (surový korpus → 100k dataset)")
    lines.append(
        f"    {'Korpus':<18} {'Dostupných':>12} {'Vybraných':>11} "
        f"{'Sampling %':>11} {'Raw k dispozícii':>18}"
    )
    lines.append("    " + "-" * 72)
    for r in proc_stats:
        avail_str = f"{r['available']:,}" if r["available"] is not None else "N/A (prázdny)"
        samp_str  = f"{r['sampling_pct']:.1f}%" if r["sampling_pct"] is not None else "—"
        lines.append(
            f"    {r['corpus']:<18} {avail_str:>12} {r['selected']:>11,} "
            f"{samp_str:>11} {'✅' if r['raw_present'] else '❌ (datasets/ prázdny)':>18}"
        )

    section("POROVNANIE SMEROV (SK→EN vs EN→SK)")
    a, b = res_sk_en, res_en_sk
    lines.append(f"  {'Metrika':<30} {'SK→EN':>12} {'EN→SK':>12}")
    lines.append("  " + "-" * 54)
    metrics = [
        ("Počet párov",          f"{a['n_pairs']:,}",             f"{b['n_pairs']:,}"),
        ("SRC medián slov",      str(a['src_words']['median']),   str(b['src_words']['median'])),
        ("TGT medián slov",      str(a['tgt_words']['median']),   str(b['tgt_words']['median'])),
        ("SRC μ slov",           str(a['src_words']['mean']),     str(b['src_words']['mean'])),
        ("TGT μ slov",           str(a['tgt_words']['mean']),     str(b['tgt_words']['mean'])),
        ("SRC μ znakov",         str(a['src_chars']['mean']),     str(b['src_chars']['mean'])),
        ("TGT μ znakov",         str(a['tgt_chars']['mean']),     str(b['tgt_chars']['mean'])),
        ("Pomer slov medián",    str(a['ratio_words']['median']), str(b['ratio_words']['median'])),
        ("Pearson r (slová)",    str(a['corr_words']['pearson_r']),str(b['corr_words']['pearson_r'])),
        ("Spearman r (slová)",   str(a['corr_words']['spearman_r']),str(b['corr_words']['spearman_r'])),
        ("TTR src",              str(a['src_ttr']),               str(b['src_ttr'])),
        ("TTR tgt",              str(a['tgt_ttr']),               str(b['tgt_ttr'])),
        ("Duplikáty src (%)",    f"{a['dup_pct']}%",             f"{b['dup_pct']}%"),
    ]
    for name, va, vb in metrics:
        lines.append(f"  {name:<30} {va:>12} {vb:>12}")

    path.write_text("\n".join(lines), encoding="utf-8")


# ── CSV súhrn ─────────────────────────────────────────────────────────────────
def write_csv(res_sk_en: Dict, res_en_sk: Dict, proc_stats: List[Dict]) -> None:
    rows = []
    for res in [res_sk_en, res_en_sk]:
        d = res["direction"]
        base = {
            "direction": d,
            "n_pairs": res["n_pairs"],
            "n_unique_src": res["n_unique_src"],
            "n_unique_tgt": res["n_unique_tgt"],
            "n_duplicates": res["n_duplicates"],
            "dup_pct": res["dup_pct"],
        }
        for key, st in [("src_words", res["src_words"]), ("tgt_words", res["tgt_words"]),
                         ("src_chars", res["src_chars"]), ("tgt_chars", res["tgt_chars"]),
                         ("ratio_words", res["ratio_words"])]:
            for stat_k, stat_v in st.items():
                base[f"{key}_{stat_k}"] = stat_v
        for key, corr in [("corr_words", res["corr_words"]), ("corr_chars", res["corr_chars"])]:
            for ck, cv in corr.items():
                base[f"{key}_{ck}"] = cv
        base["src_ttr"] = res["src_ttr"]
        base["tgt_ttr"] = res["tgt_ttr"]
        rows.append(base)

    df_main = pd.DataFrame(rows)
    df_proc = pd.DataFrame(proc_stats)

    with pd.ExcelWriter(CSV_FILE.with_suffix(".xlsx"), engine="openpyxl") as writer:
        df_main.to_excel(writer, sheet_name="Celkova_statistika", index=False)
        df_proc.to_excel(writer, sheet_name="Processing_stats", index=False)
        # per-corpus sheets
        for res in [res_sk_en, res_en_sk]:
            if "corpus_stats" not in res:
                continue
            corp_rows = []
            for corp, cs in res["corpus_stats"].items():
                corp_rows.append({"corpus": corp, **cs})
            pd.DataFrame(corp_rows).to_excel(
                writer,
                sheet_name=f"Corpora_{res['direction'].replace('→','_')}",
                index=False,
            )
    # Fallback CSV pre prípad bez openpyxl
    df_main.to_csv(CSV_FILE, index=False, encoding="utf-8")


# ── Terminálový výpis ─────────────────────────────────────────────────────────
def print_summary(res_sk_en: Dict, res_en_sk: Dict, proc_stats: List[Dict]) -> None:
    sep = "=" * 72
    print(f"\n{sep}")
    print("  CORPUS ANALYSIS — EN ↔ SK")
    print(sep)

    for res in [res_sk_en, res_en_sk]:
        d = res["direction"]
        sw, tw = res["src_words"], res["tgt_words"]
        cw = res["corr_words"]
        print(f"\n{'─'*72}")
        print(f"  {d}")
        print(f"{'─'*72}")
        print(f"  Párov:      {res['n_pairs']:>10,}  |  Duplikáty src: {res['n_duplicates']:,} ({res['dup_pct']}%)")
        print(f"  SRC slová:  min={sw['min']}  med={sw['median']}  μ={sw['mean']}  max={sw['max']}  σ={sw['std']}")
        print(f"  TGT slová:  min={tw['min']}  med={tw['median']}  μ={tw['mean']}  max={tw['max']}  σ={tw['std']}")
        print(f"  Korelácia slov:  Pearson r={cw['pearson_r']}  Spearman r={cw['spearman_r']}")
        print(f"  TTR src={res['src_ttr']}  TTR tgt={res['tgt_ttr']}")

        if "corpus_stats" in res:
            print(f"\n  {'Korpus':<18} {'Párov':>8} {'%':>6} {'μ src':>8} {'μ tgt':>8}")
            print(f"  {'─'*50}")
            for corp in CORPORA:
                if corp not in res["corpus_stats"]:
                    continue
                cs = res["corpus_stats"][corp]
                print(f"  {corp:<18} {cs['count']:>8,} {cs['pct']:>6.1f}%"
                      f" {cs['src_mean_w']:>8.1f} {cs['tgt_mean_w']:>8.1f}")

    print(f"\n{sep}")
    print("  PROCESSING STATS")
    print(sep)
    print(f"  {'Korpus':<18} {'Dostupných':>13} {'Vybraných':>11} {'Sampling':>10}")
    print(f"  {'─'*55}")
    for r in proc_stats:
        avail = f"{r['available']:,}" if r["available"] is not None else "N/A"
        samp  = f"{r['sampling_pct']:.1f}%" if r["sampling_pct"] is not None else "—"
        print(f"  {r['corpus']:<18} {avail:>13} {r['selected']:>11,} {samp:>10}")


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main() -> None:
    print("=" * 72)
    print("  KOMPLEXNÁ ANALÝZA KORPUSU — EN ↔ SK")
    print("=" * 72)

    # 1. Načítanie datasetov
    print(f"\n📥 Načítavam {DATASET_SK_EN} ...")
    df_sk_en = load_tsv(DATASET_SK_EN)
    print(f"   ✅ {len(df_sk_en):,} párov")

    print(f"📥 Načítavam {DATASET_EN_SK} ...")
    df_en_sk = load_tsv(DATASET_EN_SK)
    print(f"   ✅ {len(df_en_sk):,} párov")

    # 2. Analýza
    print("\n🧮 Počítam štatistiky...")
    res_sk_en = full_analysis(df_sk_en, "SK→EN")
    res_en_sk = full_analysis(df_en_sk, "EN→SK")

    # 3. Processing stats
    print("🔍 Zisťujem processing štatistiky...")
    proc_stats = compute_processing_stats(df_sk_en)

    # 4. Terminál
    print_summary(res_sk_en, res_en_sk, proc_stats)

    # 5. Report + CSV
    print(f"\n💾 Ukladám textový report → {REPORT_FILE}")
    write_report(res_sk_en, res_en_sk, proc_stats, REPORT_FILE)

    print(f"💾 Ukladám CSV/XLSX → {CSV_FILE} / {CSV_FILE.with_suffix('.xlsx')}")
    try:
        write_csv(res_sk_en, res_en_sk, proc_stats)
    except ImportError:
        df_main = pd.DataFrame([
            {"direction": r["direction"], "n_pairs": r["n_pairs"]}
            for r in [res_sk_en, res_en_sk]
        ])
        df_main.to_csv(CSV_FILE, index=False, encoding="utf-8")

    # 6. Grafy
    print(f"\n📈 Generujem grafy → {OUT_DIR}/")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Histogram slov: SK→EN vs EN→SK
    plot_hist_compare(
        res_sk_en["src_w_arr"], res_en_sk["src_w_arr"],
        "SK→EN (src)", "EN→SK (src)",
        "Počet slov", "Porovnanie dĺžky src viet (slová)",
        OUT_DIR / "compare_hist_src_words.png",
    )
    plot_hist_compare(
        res_sk_en["tgt_w_arr"], res_en_sk["tgt_w_arr"],
        "SK→EN (tgt)", "EN→SK (tgt)",
        "Počet slov", "Porovnanie dĺžky tgt viet (slová)",
        OUT_DIR / "compare_hist_tgt_words.png",
    )

    # CDF slov
    plot_cdf_compare(
        res_sk_en["src_w_arr"], res_en_sk["src_w_arr"],
        "SK→EN", "EN→SK",
        "Počet slov", "CDF dĺžky src viet (slová)",
        OUT_DIR / "compare_cdf_src_words.png",
    )

    # Korelácia src-tgt
    plot_scatter_corr(res_sk_en["src_w_arr"], res_sk_en["tgt_w_arr"],
                      "SK→EN", OUT_DIR / "corr_sk_en.png")
    plot_scatter_corr(res_en_sk["src_w_arr"], res_en_sk["tgt_w_arr"],
                      "EN→SK", OUT_DIR / "corr_en_sk.png")

    # Pomer dĺžok
    plot_ratio_hist(res_sk_en["ratio_w_arr"], "SK→EN",
                    OUT_DIR / "ratio_sk_en.png")
    plot_ratio_hist(res_en_sk["ratio_w_arr"], "EN→SK",
                    OUT_DIR / "ratio_en_sk.png")

    # Per-corpus boxplot
    plot_boxplot_corpora(df_sk_en, "SK→EN",
                         OUT_DIR / "boxplot_corpora_sk_en.png")
    plot_boxplot_corpora(df_en_sk, "EN→SK",
                         OUT_DIR / "boxplot_corpora_en_sk.png")

    # Bar chart korpusov
    plot_corpus_bar(df_sk_en, "SK→EN", OUT_DIR / "bar_corpus_sk_en.png")
    plot_corpus_bar(df_en_sk, "EN→SK", OUT_DIR / "bar_corpus_en_sk.png")

    # Priemerná dĺžka per-corpus, oba smery
    plot_mean_per_corpus_compare(df_sk_en, df_en_sk,
                                 OUT_DIR / "mean_words_per_corpus.png")

    # Processing stats
    plot_processing_stats(proc_stats, OUT_DIR / "processing_stats.png")

    print(f"\n✅ Hotovo!")
    print(f"   📄 Report:  {REPORT_FILE}")
    print(f"   📊 CSV:     {CSV_FILE}")
    print(f"   🖼  Grafy:   {OUT_DIR}/  ({len(list(OUT_DIR.glob('*.png')))} súborov)")
    print("=" * 72)


if __name__ == "__main__":
    main()
