"""
Shared utility functions for translation evaluation.

Exports:
    load_tsv_file       — load eval sample TSV
    calculate_metrics   — BLEU / chrF / COMET
    evaluate_files      — run model on all 10 sample files
    save_results        — write TXT report + plots
    parse_numbered      — parse numbered LLM output
    get_forced_bos_id   — resolve forced BOS token for NLLB/M2M100
    get_api_key         — read API key from .env
"""
import math
import os
import time
import unicodedata
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Callable

import pandas as pd
from sacrebleu.metrics import BLEU, CHRF
import matplotlib.pyplot as plt
from dotenv import load_dotenv

load_dotenv()

# ── COMET lazy-load ──────────────────────────────────────────────────────────
_comet_model = None


def _get_comet_model():
    """Lazy-load COMET model (downloads ~1.8 GB on first run)."""
    global _comet_model
    if _comet_model is None:
        try:
            from comet import download_model, load_from_checkpoint
            model_path = download_model("Unbabel/wmt22-comet-da")
            _comet_model = load_from_checkpoint(model_path)
            print("COMET model loaded successfully.")
        except ImportError:
            print("WARNING: unbabel-comet not installed. Run: pip install unbabel-comet")
            _comet_model = None
        except Exception as e:
            print(f"WARNING: Could not load COMET model: {e}")
            _comet_model = None
    return _comet_model


# ── Resolve eval_samples directory ───────────────────────────────────────────
def _resolve_eval_dir() -> Path:
    """Find eval_samples/ regardless of working directory."""
    candidate = Path(__file__).resolve().parent.parent / "eval_samples"
    if candidate.is_dir():
        return candidate
    candidate = Path("../eval_samples").resolve()
    if candidate.is_dir():
        return candidate
    return Path(__file__).resolve().parent.parent / "eval_samples"


# ── File loading ─────────────────────────────────────────────────────────────
def load_tsv_file(filepath: str) -> Tuple[List[str], List[str]]:
    """Load a TSV eval-sample and return (sources, targets).

    Skips bad lines, normalises to NFC, drops empty rows.
    """
    try:
        df = pd.read_csv(
            filepath, sep='\t', header=None,
            usecols=[0, 1], names=['source', 'target'],
            dtype=str, keep_default_na=False,
            on_bad_lines='skip',
        )
    except Exception as e:
        raise RuntimeError(f"Failed to read {filepath}: {e}")

    for col in ('source', 'target'):
        df[col] = df[col].apply(
            lambda x: unicodedata.normalize('NFC', str(x).strip())
        )

    df = df[(df['source'].str.len() > 0) & (df['target'].str.len() > 0)]

    if df.empty:
        raise ValueError(f"No valid sentence pairs found in {filepath}")

    return df['source'].tolist(), df['target'].tolist()


# ── Metrics ──────────────────────────────────────────────────────────────────
def calculate_metrics(
    predictions: List[str],
    references: List[str],
    sources: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Calculate BLEU, chrF, and optionally COMET scores.

    Args:
        predictions: Model translations.
        references:  Reference translations (single-ref list).
        sources:     Source sentences (required for COMET).

    Returns:
        Dict with keys 'bleu', 'chrf', and optionally 'comet'.
        COMET is stored on a 0-100 scale (×100 from raw model output).
    """
    bleu_metric = BLEU()
    chrf_metric = CHRF()

    bleu_score = bleu_metric.corpus_score(predictions, [references]).score
    chrf_score = chrf_metric.corpus_score(predictions, [references]).score

    result: Dict[str, float] = {
        'bleu': round(float(bleu_score), 4),
        'chrf': round(float(chrf_score), 4),
    }

    if sources is not None:
        model = _get_comet_model()
        if model is not None:
            try:
                data = [
                    {"src": s, "mt": m, "ref": r}
                    for s, m, r in zip(sources, predictions, references)
                ]
                comet_output = model.predict(data, batch_size=64, gpus=0, num_workers=1)

                # unbabel-comet >= 2.0 returns a Prediction object
                if hasattr(comet_output, 'system_score'):
                    comet_score = comet_output.system_score
                # older versions return (sentence_scores, system_score)
                elif isinstance(comet_output, (list, tuple)) and len(comet_output) >= 2:
                    comet_score = comet_output[1]
                else:
                    raise ValueError(f"Unexpected COMET output: {type(comet_output)}")

                result['comet'] = round(float(comet_score) * 100, 2)
            except Exception as e:
                print(f"WARNING: COMET scoring failed: {e}")

    return result


# ── Evaluation loop ──────────────────────────────────────────────────────────
def evaluate_files(
    translate_func: Callable[[List[str]], List[str]],
    file_pattern: str,
    direction: str,
) -> Dict:
    """Run translation evaluation on sample files 01–10.

    Args:
        translate_func: Function(List[str]) -> List[str].
        file_pattern:   e.g. "EN_SK" or "SK_EN".
        direction:      e.g. "EN→SK" or "SK→EN".

    Returns:
        Dict with bleu_scores, chrf_scores, comet_scores, times, averages.
    """
    eval_dir = _resolve_eval_dir()
    results: Dict = {
        'direction':     direction,
        'files':         [],
        'bleu_scores':   [],
        'chrf_scores':   [],
        'comet_scores':  [],
        'times':         [],
        'total_samples': 0,
    }

    for i in range(1, 11):
        filename = f"{file_pattern}_sample_{i:02d}.tsv"
        filepath = eval_dir / filename

        if not filepath.exists():
            print(f"  Warning: {filename} not found, skipping...")
            continue

        print(f"  Processing {filename}...")

        try:
            sources, targets = load_tsv_file(str(filepath))
        except (RuntimeError, ValueError) as e:
            print(f"  Warning: Skipping {filename}: {e}")
            continue

        results['total_samples'] += len(sources)

        start = time.time()
        predictions = translate_func(sources)
        elapsed = time.time() - start

        # Guard: length mismatch
        if len(predictions) != len(sources):
            print(f"  WARNING: predictions ({len(predictions)}) != sources ({len(sources)}), padding.")
            if len(predictions) < len(sources):
                predictions.extend([''] * (len(sources) - len(predictions)))
            else:
                predictions = predictions[:len(sources)]

        metrics = calculate_metrics(predictions, targets, sources=sources)

        results['files'].append(filename)
        results['bleu_scores'].append(metrics['bleu'])
        results['chrf_scores'].append(metrics['chrf'])
        results['comet_scores'].append(metrics.get('comet', None))
        results['times'].append(elapsed)

        comet_str = ""
        if 'comet' in metrics:
            comet_str = f", COMET: {metrics['comet']:.2f}"
        print(f"    BLEU: {metrics['bleu']:.2f}  "
              f"chrF: {metrics['chrf']:.2f}{comet_str}  "
              f"Time: {elapsed:.2f}s")

    n = len(results['bleu_scores'])
    if n:
        results['avg_bleu']   = round(sum(results['bleu_scores']) / n, 4)
        results['avg_chrf']   = round(sum(results['chrf_scores']) / n, 4)
        results['total_time'] = round(sum(results['times']), 2)
        comet_vals = [c for c in results['comet_scores'] if c is not None]
        if comet_vals:
            results['avg_comet'] = round(sum(comet_vals) / len(comet_vals), 4)

    return results


# ── Save results ─────────────────────────────────────────────────────────────
def save_results(method_name: str, en_sk_results: Dict, sk_en_results: Dict) -> None:
    """Write TXT report and generate plots to models/results/."""
    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")

    txt_file = results_dir / f"{method_name}_{timestamp}.txt"
    with open(txt_file, 'w', encoding='utf-8') as f:
        f.write(f"{'=' * 65}\n")
        f.write(f"  Translation Evaluation Results: {method_name}\n")
        f.write(f"{'=' * 65}\n\n")

        for label, res in [("EN -> SK", en_sk_results), ("SK -> EN", sk_en_results)]:
            f.write(f"Direction: {label}\n")
            f.write(f"{'-' * 65}\n")
            if res.get('bleu_scores'):
                f.write(f"  Average BLEU:    {res['avg_bleu']:.2f}\n")
                f.write(f"  Average chrF:    {res['avg_chrf']:.2f}\n")
                if 'avg_comet' in res:
                    f.write(f"  Average COMET:   {res['avg_comet']:.2f}\n")
                f.write(f"  Total Time:      {res['total_time']:.2f}s\n")
                f.write(f"  Total Samples:   {res['total_samples']}\n")
                f.write(f"  Avg Time/Sample: "
                        f"{res['total_time'] / max(res['total_samples'], 1):.3f}s\n\n")
                f.write("  Per-file results:\n")
                for j, fname in enumerate(res['files']):
                    comet_part = ""
                    if res['comet_scores'][j] is not None:
                        comet_part = f", COMET={res['comet_scores'][j]:.2f}"
                    f.write(
                        f"    {fname}: "
                        f"BLEU={res['bleu_scores'][j]:.2f}, "
                        f"chrF={res['chrf_scores'][j]:.2f}{comet_part}, "
                        f"Time={res['times'][j]:.2f}s\n"
                    )
            else:
                f.write("  No results.\n")
            f.write("\n")

    print(f"\nResults saved to: {txt_file}")
    generate_plots(method_name, en_sk_results, sk_en_results, timestamp)


# ── Plots (per-method) ────────────────────────────────────────────────────────
def generate_plots(
    method_name: str,
    en_sk_results: Dict,
    sk_en_results: Dict,
    timestamp: str,
) -> None:
    """Generate per-method evaluation plots (2×2 or 3×2 when COMET present)."""
    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(exist_ok=True)

    has_comet = (
        any(c is not None for c in en_sk_results.get('comet_scores', []))
        or any(c is not None for c in sk_en_results.get('comet_scores', []))
    )

    rows = 3 if has_comet else 2
    fig, axes = plt.subplots(rows, 2, figsize=(14, 5 * rows))
    fig.suptitle(f'{method_name} — Evaluation Results', fontsize=16, fontweight='bold')

    x = range(1, 11)

    def _plot_line(ax, en_data, sk_data, ylabel, title):
        if en_data:
            ax.plot(x[:len(en_data)], en_data, marker='o', label='EN→SK', linewidth=2)
        if sk_data:
            ax.plot(x[:len(sk_data)], sk_data, marker='s', label='SK→EN', linewidth=2)
        ax.set_xlabel('File Number')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    _plot_line(axes[0, 0],
               en_sk_results.get('bleu_scores', []),
               sk_en_results.get('bleu_scores', []),
               'BLEU Score', 'BLEU Scores per File')
    _plot_line(axes[0, 1],
               en_sk_results.get('chrf_scores', []),
               sk_en_results.get('chrf_scores', []),
               'chrF Score', 'chrF Scores per File')

    # Average metrics bar chart
    ax_bar = axes[1, 0]
    metrics_labels = ['BLEU', 'chrF']
    en_avgs = [en_sk_results.get('avg_bleu', 0), en_sk_results.get('avg_chrf', 0)]
    sk_avgs = [sk_en_results.get('avg_bleu', 0), sk_en_results.get('avg_chrf', 0)]
    if has_comet:
        metrics_labels.append('COMET')
        en_avgs.append(en_sk_results.get('avg_comet', 0))
        sk_avgs.append(sk_en_results.get('avg_comet', 0))

    x_pos = list(range(len(metrics_labels)))
    w = 0.35
    ax_bar.bar([p - w / 2 for p in x_pos], en_avgs, w, label='EN→SK', alpha=0.8)
    ax_bar.bar([p + w / 2 for p in x_pos], sk_avgs, w, label='SK→EN', alpha=0.8)
    ax_bar.set_ylabel('Score')
    ax_bar.set_title('Average Metrics Comparison')
    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels(metrics_labels)
    ax_bar.legend()
    ax_bar.grid(True, alpha=0.3, axis='y')

    _plot_line(axes[1, 1],
               en_sk_results.get('times', []),
               sk_en_results.get('times', []),
               'Time (seconds)', 'Processing Time per File')

    if has_comet:
        # Use NaN for missing COMET values to avoid distorting the plot
        en_comet = [c if c is not None else math.nan
                    for c in en_sk_results.get('comet_scores', [])]
        sk_comet = [c if c is not None else math.nan
                    for c in sk_en_results.get('comet_scores', [])]
        _plot_line(axes[2, 0], en_comet, sk_comet,
                   'COMET Score', 'COMET Scores per File')
        axes[2, 1].axis('off')

    plt.tight_layout()
    plot_file = results_dir / f"{method_name}_{timestamp}.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to: {plot_file}")


# ── Shared LLM helpers ────────────────────────────────────────────────────────
def parse_numbered(raw: str, expected: int) -> List[str]:
    """Parse LLM output of the form '1. text\\n2. text\\n...'.

    Falls back to plain line-split when numbering is absent.
    Always returns exactly *expected* strings (padded with '' or truncated).
    """
    lines = [ln.strip() for ln in raw.strip().splitlines() if ln.strip()]
    result: List[str] = []

    for line in lines:
        stripped = line
        for sep in (". ", ") "):
            idx = line.find(sep)
            if idx != -1 and line[:idx].strip().isdigit():
                stripped = line[idx + len(sep):]
                break
        result.append(stripped)

    if len(result) < expected:
        result.extend([""] * (expected - len(result)))
    return result[:expected]


# ── Shared NMT helpers ────────────────────────────────────────────────────────
def get_forced_bos_id(tokenizer, tgt_code: str) -> int:
    """Resolve forced BOS token id for NLLB / M2M100 tokenizers.

    Handles API differences across transformers versions:
    - tokenizer.lang_code_to_id  (dict mapping)
    - tokenizer.get_lang_id()    (method)
    - tokenizer.convert_tokens_to_ids()  (fallback)
    """
    if hasattr(tokenizer, "lang_code_to_id"):
        mapping = getattr(tokenizer, "lang_code_to_id")
        if isinstance(mapping, dict) and tgt_code in mapping:
            return mapping[tgt_code]
    if hasattr(tokenizer, "get_lang_id"):
        return tokenizer.get_lang_id(tgt_code)
    token_id = tokenizer.convert_tokens_to_ids(tgt_code)
    if token_id == tokenizer.unk_token_id:
        raise ValueError(
            f"Cannot resolve forced_bos_token_id for '{tgt_code}'. "
            f"Check LANG_CODES for this model type."
        )
    return token_id


# ── API key helper ────────────────────────────────────────────────────────────
def get_api_key(key_name: str = "OPENAI_API_KEY") -> str:
    """Read API key from .env / environment variables."""
    api_key = os.getenv(key_name)
    if not api_key:
        raise ValueError(
            f"{key_name} not found. Add it to your .env file:\n"
            f"  {key_name}=your_key_here"
        )
    return api_key
