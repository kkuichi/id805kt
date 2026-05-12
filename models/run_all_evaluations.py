"""
Run all translation evaluation methods
Executes all 8 methods sequentially and generates a comparison report
"""
import sys
import time
import importlib
from pathlib import Path
from typing import Optional, Dict, List
import pandas as pd
import matplotlib.pyplot as plt

# Ensure the models/ directory is on sys.path so sub-modules are importable
# regardless of the working directory from which this script is launched.
_MODELS_DIR = str(Path(__file__).resolve().parent)
if _MODELS_DIR not in sys.path:
    sys.path.insert(0, _MODELS_DIR)

# List of all evaluation modules (without .py)
EVALUATION_MODULES = [
    "01_zero_shot_llm",
    "02_few_shot_llm",
    "03_marianmt",
    "04_t5_nmt",
    "05_nllb_m2m100",
    "06_fine_tuning_lora",
    "07_backtranslation",
    "08_nbest_reranking",
]

METHOD_NAMES = [
    "Zero-shot LLM",
    "Few-shot LLM",
    "MarianMT",
    "T5 NMT",
    "NLLB/M2M100",
    "Fine-tuning (LoRA)",
    "Backtranslation",
    "N-best Reranking",
]


def run_evaluation(module_name: str, method_name: str) -> Dict:
    """Run a single evaluation module via importlib"""
    print("\n" + "=" * 80)
    print(f"  Running: {method_name}  ({module_name})")
    print("=" * 80)

    start_time = time.time()

    try:
        # Import (or re-import) module; reload only if already cached to avoid
        # stale state from a previous evaluation run in the same process.
        if module_name in sys.modules:
            mod = importlib.reload(sys.modules[module_name])
        else:
            mod = importlib.import_module(module_name)

        if hasattr(mod, "main"):
            mod.main()
        else:
            print(f"  Warning: {module_name} has no main() — skipped execution.")

        elapsed = time.time() - start_time
        return {
            'method': method_name,
            'status': 'success',
            'time': elapsed,
        }

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"  Error in {module_name}: {e}")
        return {
            'method': method_name,
            'status': 'error',
            'error': str(e),
            'time': elapsed,
        }


def parse_results_file(results_dir: Path, method_prefix: str) -> Optional[Dict]:
    """Parse the most recent results file for a method.

    Reads BLEU, chrF, COMET scores.
    """
    pattern = f"{method_prefix}_*.txt"
    files = sorted(results_dir.glob(pattern), reverse=True)

    if not files:
        return None

    results: Dict = {
        'en_sk': {},
        'sk_en': {},
    }
    current_direction: Optional[str] = None

    with open(files[0], 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            if 'EN -> SK' in line or 'EN → SK' in line:
                current_direction = 'en_sk'
            elif 'SK -> EN' in line or 'SK → EN' in line:
                current_direction = 'sk_en'
            elif current_direction:
                try:
                    if 'Average BLEU:' in line:
                        results[current_direction]['bleu'] = float(line.split(':')[1].strip())
                    elif 'Average chrF:' in line:
                        results[current_direction]['chrf'] = float(line.split(':')[1].strip())
                    elif 'Average COMET:' in line:
                        results[current_direction]['comet'] = float(line.split(':')[1].strip())
                    elif 'Total Time:' in line:
                        time_str = line.split(':')[1].strip().replace('s', '')
                        results[current_direction]['time'] = float(time_str)
                except (ValueError, IndexError):
                    pass  # Skip malformed lines gracefully

    return results


def generate_comparison_report(results_dir: Path) -> None:
    """Generate a comparison report of all methods (BLEU, chrF, COMET)"""
    print("\n" + "=" * 80)
    print("  Generating Comparison Report")
    print("=" * 80)

    all_results: List[Dict] = []
    has_comet = False

    for module, method in zip(EVALUATION_MODULES, METHOD_NAMES):
        results = parse_results_file(results_dir, module)
        if results is None:
            continue

        en_comet = results['en_sk'].get('comet')
        sk_comet = results['sk_en'].get('comet')
        if en_comet is not None or sk_comet is not None:
            has_comet = True

        row = {
            'Method': method,
            'EN->SK BLEU': results['en_sk'].get('bleu', 0),
            'EN->SK chrF': results['en_sk'].get('chrf', 0),
            'EN->SK COMET': en_comet if en_comet is not None else 0,
            'EN->SK Time': results['en_sk'].get('time', 0),
            'SK->EN BLEU': results['sk_en'].get('bleu', 0),
            'SK->EN chrF': results['sk_en'].get('chrf', 0),
            'SK->EN COMET': sk_comet if sk_comet is not None else 0,
            'SK->EN Time': results['sk_en'].get('time', 0),
        }
        all_results.append(row)

    if not all_results:
        print("  No results found to compare.")
        return

    df = pd.DataFrame(all_results)

    ts = time.strftime('%Y%m%d_%H%M%S')

    # CSV
    csv_file = results_dir / f"comparison_report_{ts}.csv"
    df.to_csv(csv_file, index=False)
    print(f"\nComparison CSV saved to: {csv_file}")

    # TXT
    txt_file = results_dir / f"comparison_report_{ts}.txt"
    with open(txt_file, 'w', encoding='utf-8') as f:
        f.write("=" * 85 + "\n")
        f.write("  Translation Methods — Comparison Report\n")
        f.write("=" * 85 + "\n\n")
        f.write(df.to_string(index=False))
        f.write("\n\n")

        f.write("=" * 85 + "\n")
        f.write("  Rankings\n")
        f.write("=" * 85 + "\n\n")

        ranking_cols = [
            'EN->SK BLEU',
            'SK->EN BLEU',
            'EN->SK chrF',
            'SK->EN chrF',
        ]
        if has_comet:
            ranking_cols.extend(['EN->SK COMET', 'SK->EN COMET'])

        for col in ranking_cols:
            f.write(f"{col} (higher is better):\n")
            ranked = df.sort_values(col, ascending=False)
            for rank, (_, row) in enumerate(ranked.iterrows(), 1):
                f.write(f"  {rank}. {row['Method']}: {row[col]:.2f}\n")
            f.write("\n")

    print(f"Comparison report saved to: {txt_file}")

    generate_comparison_plots(df, results_dir, has_comet, ts)


def generate_comparison_plots(
    df: pd.DataFrame,
    results_dir: Path,
    has_comet: bool,
    timestamp: str,
) -> None:
    """Generate 3×3 comparison plots (9 subplots).

    Layout:
      [0,0] EN→SK overlay  — BLEU + chrF + COMET na jednom grafe
      [0,1] SK→EN overlay  — BLEU + chrF + COMET na jednom grafe
      [0,2] Combined ranking — priemerné skóre (BLEU+chrF) zoradené
      [1,0] BLEU comparison  — grouped bar EN→SK vs SK→EN
      [1,1] chrF comparison  — grouped bar
      [1,2] COMET comparison — grouped bar (alebo info ak nie je k dispozícii)
      [2,0] Processing time  — grouped bar
      [2,1] Quality vs Speed — scatter BLEU vs čas
      [2,2] Metrics heatmap  — metódy × metriky (farebná tabuľka)
    """
    import numpy as np

    methods  = df['Method'].tolist()
    n        = len(methods)
    x_pos    = list(range(n))
    width    = 0.35

    COLORS = {
        'EN->SK': '#4C72B0',
        'SK->EN': '#DD8452',
        'BLEU':   '#4C72B0',
        'chrF':   '#55A868',
        'COMET':  '#C44E52',
    }

    fig, axes = plt.subplots(3, 3, figsize=(22, 18))
    fig.suptitle('Translation Methods — Comparison (3×3)', fontsize=20, fontweight='bold', y=1.01)

    # ── Helper: grouped bar (EN→SK + SK→EN) ──────────────────────────────────
    def _bar_pair(ax, col_en: str, col_sk: str, ylabel: str, title: str) -> None:
        ax.bar([p - width / 2 for p in x_pos], df[col_en], width,
               label='EN→SK', color=COLORS['EN->SK'], alpha=0.85)
        ax.bar([p + width / 2 for p in x_pos], df[col_sk], width,
               label='SK→EN', color=COLORS['SK->EN'], alpha=0.85)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(methods, rotation=40, ha='right', fontsize=9)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.25, axis='y')

    # ── Helper: overlay linechart (BLEU + chrF + COMET pre jeden smer) ───────
    def _overlay(ax, direction_prefix: str, title: str) -> None:
        """
        Nakresli BLEU, chrF a COMET ako samostatné línie pre daný smer.
        Každá metrika má svoju ľavú os (spoločnú) keďže sú na rovnakej škále.
        """
        bleu_col  = f'{direction_prefix} BLEU'
        chrf_col  = f'{direction_prefix} chrF'
        comet_col = f'{direction_prefix} COMET'

        x = range(n)

        ln1 = ax.plot(x, df[bleu_col], marker='o', linewidth=2.2,
                      color=COLORS['BLEU'], label='BLEU', zorder=3)
        ln2 = ax.plot(x, df[chrf_col], marker='s', linewidth=2.2,
                      color=COLORS['chrF'], label='chrF', zorder=3)

        lines = ln1 + ln2
        labels = ['BLEU', 'chrF']

        if has_comet and df[comet_col].sum() > 0:
            ln3 = ax.plot(x, df[comet_col], marker='^', linewidth=2.2,
                          color=COLORS['COMET'], label='COMET', zorder=3)
            lines += ln3
            labels.append('COMET')

        # Farebné pásy pre lepšiu čitateľnosť metód
        for i in range(n):
            ax.axvspan(i - 0.5, i + 0.5, alpha=0.04,
                       color='gray' if i % 2 == 0 else 'white')

        # Hodnoty na bodoch pre BLEU
        for i, val in enumerate(df[bleu_col]):
            ax.annotate(f'{val:.1f}', (i, val), textcoords='offset points',
                        xytext=(0, 6), ha='center', fontsize=7, color=COLORS['BLEU'])

        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_ylabel('Skóre', fontsize=11)
        ax.set_xticks(range(n))
        ax.set_xticklabels(methods, rotation=40, ha='right', fontsize=9)
        ax.legend(lines, labels, fontsize=9, loc='upper left')
        ax.grid(True, alpha=0.25, axis='y')
        ax.set_ylim(bottom=0)

    # ── [0,0] EN→SK overlay ───────────────────────────────────────────────────
    _overlay(axes[0, 0], 'EN->SK', 'EN→SK: BLEU + chrF + COMET (overlay)')

    # ── [0,1] SK→EN overlay ───────────────────────────────────────────────────
    _overlay(axes[0, 1], 'SK->EN', 'SK→EN: BLEU + chrF + COMET (overlay)')

    # ── [0,2] Combined ranking (BLEU+chrF avg, zoradené) ─────────────────────
    ax_rank = axes[0, 2]
    avg_score = (df['EN->SK BLEU'] + df['SK->EN BLEU'] +
                 df['EN->SK chrF']  + df['SK->EN chrF']) / 4
    ranked = avg_score.sort_values(ascending=True)
    ranked_methods = [methods[i] for i in ranked.index]
    colors_rank = plt.cm.RdYlGn(  # červená→zelená podľa skóre
        (ranked.values - ranked.values.min()) /
        max(ranked.values.max() - ranked.values.min(), 1e-6)
    )
    bars = ax_rank.barh(range(len(ranked)), ranked.values,
                        color=colors_rank, edgecolor='white', alpha=0.9)
    for bar, val in zip(bars, ranked.values):
        ax_rank.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                     f'{val:.2f}', va='center', fontsize=9)
    ax_rank.set_yticks(range(len(ranked)))
    ax_rank.set_yticklabels(ranked_methods, fontsize=9)
    ax_rank.set_xlabel('Priemerné skóre (BLEU+chrF / oba smery)', fontsize=10)
    ax_rank.set_title('Celkový rebríček metód', fontsize=13, fontweight='bold')
    ax_rank.grid(True, alpha=0.25, axis='x')

    # ── [1,0] BLEU comparison ─────────────────────────────────────────────────
    _bar_pair(axes[1, 0], 'EN->SK BLEU', 'SK->EN BLEU', 'BLEU Score', 'BLEU — porovnanie metód')

    # ── [1,1] chrF comparison ─────────────────────────────────────────────────
    _bar_pair(axes[1, 1], 'EN->SK chrF', 'SK->EN chrF', 'chrF Score', 'chrF — porovnanie metód')

    # ── [1,2] COMET comparison (alebo info text) ──────────────────────────────
    ax_comet = axes[1, 2]
    if has_comet and (df['EN->SK COMET'].sum() > 0 or df['SK->EN COMET'].sum() > 0):
        ax_comet.bar([p - width / 2 for p in x_pos], df['EN->SK COMET'], width,
                     label='EN→SK', color=COLORS['EN->SK'], alpha=0.85)
        ax_comet.bar([p + width / 2 for p in x_pos], df['SK->EN COMET'], width,
                     label='SK→EN', color=COLORS['SK->EN'], alpha=0.85)
        ax_comet.set_ylabel('COMET Score', fontsize=11)
        ax_comet.set_title('COMET — porovnanie metód', fontsize=13, fontweight='bold')
        ax_comet.set_xticks(x_pos)
        ax_comet.set_xticklabels(methods, rotation=40, ha='right', fontsize=9)
        ax_comet.legend(fontsize=9)
        ax_comet.grid(True, alpha=0.25, axis='y')
    else:
        ax_comet.text(0.5, 0.5,
                      'COMET nie je k dispozícii\n(nainštalujte unbabel-comet)',
                      ha='center', va='center', fontsize=12,
                      transform=ax_comet.transAxes,
                      bbox=dict(boxstyle='round', facecolor='#FFF3CD', alpha=0.8))
        ax_comet.set_title('COMET — porovnanie metód', fontsize=13, fontweight='bold')
        ax_comet.axis('off')

    # ── [2,0] Processing time ─────────────────────────────────────────────────
    _bar_pair(axes[2, 0], 'EN->SK Time', 'SK->EN Time', 'Čas (s)', 'Čas spracovania — porovnanie')

    # ── [2,1] Quality vs Speed scatter ────────────────────────────────────────
    ax_scatter = axes[2, 1]
    avg_bleu_vals  = (df['EN->SK BLEU'] + df['SK->EN BLEU']) / 2
    avg_chrf_vals  = (df['EN->SK chrF']  + df['SK->EN chrF'])  / 2
    avg_time_vals  = (df['EN->SK Time']  + df['SK->EN Time'])  / 2

    scatter_colors = plt.cm.tab10(range(n))
    for i, method in enumerate(methods):
        ax_scatter.scatter(avg_time_vals.iloc[i], avg_bleu_vals.iloc[i],
                           s=220, color=scatter_colors[i], zorder=3,
                           label=method, alpha=0.85)
        ax_scatter.annotate(method,
                            (avg_time_vals.iloc[i], avg_bleu_vals.iloc[i]),
                            textcoords='offset points', xytext=(5, 4),
                            fontsize=7.5)
    ax_scatter.set_xlabel('Priemerný čas (s)', fontsize=11)
    ax_scatter.set_ylabel('Priemerné BLEU', fontsize=11)
    ax_scatter.set_title('Kvalita vs. Rýchlosť', fontsize=13, fontweight='bold')
    ax_scatter.grid(True, alpha=0.25)

    # ── [2,2] Metrics heatmap (metódy × metriky) ──────────────────────────────
    ax_heat = axes[2, 2]
    heat_cols = ['EN->SK BLEU', 'SK->EN BLEU', 'EN->SK chrF', 'SK->EN chrF']
    heat_labels = ['EN→SK\nBLEU', 'SK→EN\nBLEU', 'EN→SK\nchrF', 'SK→EN\nchrF']
    if has_comet and df['EN->SK COMET'].sum() > 0:
        heat_cols  += ['EN->SK COMET', 'SK->EN COMET']
        heat_labels += ['EN→SK\nCOMET', 'SK→EN\nCOMET']

    heat_data = df[heat_cols].values.astype(float)
    # Normalizácia stĺpcov na [0,1] pre farebné porovnanie
    col_min = heat_data.min(axis=0)
    col_max = heat_data.max(axis=0)
    heat_norm = (heat_data - col_min) / (col_max - col_min + 1e-9)

    im = ax_heat.imshow(heat_norm.T, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
    ax_heat.set_xticks(range(n))
    ax_heat.set_xticklabels(methods, rotation=40, ha='right', fontsize=8)
    ax_heat.set_yticks(range(len(heat_labels)))
    ax_heat.set_yticklabels(heat_labels, fontsize=9)
    ax_heat.set_title('Heatmapa metrík\n(normalizované, zelená = lepšie)',
                      fontsize=12, fontweight='bold')

    # Hodnoty v bunkách
    for row_i in range(len(heat_labels)):
        for col_i in range(n):
            val = heat_data[col_i, row_i]
            text_color = 'black' if heat_norm[col_i, row_i] > 0.4 else 'white'
            ax_heat.text(col_i, row_i, f'{val:.1f}',
                         ha='center', va='center', fontsize=7.5,
                         color=text_color, fontweight='bold')

    plt.colorbar(im, ax=ax_heat, shrink=0.8, label='Normalizované skóre')

    plt.tight_layout()
    plot_file = results_dir / f"comparison_plot_{timestamp}.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Comparison plot saved to: {plot_file}")


def main() -> None:
    print("=" * 80)
    print("  Running All Translation Evaluation Methods")
    print("=" * 80)
    print(f"\n  Methods to evaluate: {len(EVALUATION_MODULES)}")
    print(f"  Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(exist_ok=True)

    execution_results: List[Dict] = []

    for module, method in zip(EVALUATION_MODULES, METHOD_NAMES):
        result = run_evaluation(module, method)
        execution_results.append(result)

        if result['status'] == 'success':
            print(f"\n  OK: {method} — {result['time']:.2f}s")
        else:
            print(f"\n  FAIL: {method} — {result.get('error', 'Unknown error')}")

    generate_comparison_report(results_dir)

    print("\n" + "=" * 80)
    print("  All Evaluations Complete!")
    print("=" * 80)

    successful = sum(1 for r in execution_results if r['status'] == 'success')
    total_time = sum(r['time'] for r in execution_results)

    print(f"\n  Successful: {successful}/{len(execution_results)}")
    print(f"  Failed:     {len(execution_results) - successful}/{len(execution_results)}")
    print(f"  Total time: {total_time:.2f}s ({total_time / 60:.1f} min)")
    print(f"  Results in: {results_dir}")
    print(f"  End time:   {time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()