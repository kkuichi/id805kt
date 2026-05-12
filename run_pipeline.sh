#!/usr/bin/env bash
# =============================================================================
#  run_pipeline.sh — Spustí celý pipeline od začiatku do konca
#
#  Použitie:
#    chmod +x run_pipeline.sh      (iba prvýkrát)
#    ./run_pipeline.sh             (spustí všetko)
#    ./run_pipeline.sh --skip-data (preskočí kroky 1-5, spustí iba modely)
# =============================================================================

set -e  # zastav sa pri akejkoľvek chybe

# ── Farby pre výpis ──────────────────────────────────────────────────────────
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'  # reset

ok()   { echo -e "${GREEN}  ✅ $1${NC}"; }
info() { echo -e "${BLUE}  ➜  $1${NC}"; }
warn() { echo -e "${YELLOW}  ⚠️  $1${NC}"; }
fail() { echo -e "${RED}  ❌ $1${NC}"; exit 1; }

header() {
  echo ""
  echo -e "${BLUE}============================================================${NC}"
  echo -e "${BLUE}  $1${NC}"
  echo -e "${BLUE}============================================================${NC}"
}

# ── Prejdi do priečinka kde leží tento skript ────────────────────────────────
cd "$(dirname "$0")"

header "Machine Translation Pipeline — EN ↔ SK"
echo "  Štart: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Priečinok: $(pwd)"

# ── Skontroluj Python ────────────────────────────────────────────────────────
if ! command -v python &> /dev/null; then
  fail "Python nenájdený. Aktivuj virtuálne prostredie: source .venv/bin/activate"
fi
info "Python: $(python --version)"

# ── Argument --skip-data ─────────────────────────────────────────────────────
SKIP_DATA=false
if [[ "$1" == "--skip-data" ]]; then
  SKIP_DATA=true
  warn "Preskakujem prípravu dát (--skip-data), spúšťam iba modely."
fi

# =============================================================================
#  FÁZA 1 — PRÍPRAVA DÁT
# =============================================================================

if [[ "$SKIP_DATA" == false ]]; then

  # ── Krok 1: Kombinovanie korpusov → 100k párov ──────────────────────────
  header "Krok 1/6 — Kombinovanie korpusov (create_datasets.py)"

  if [[ ! -d "datasets" ]] || [[ -z "$(ls -A datasets 2>/dev/null)" ]]; then
    warn "Priečinok datasets/ je prázdny alebo neexistuje."
    warn "Stiahni korpusy do datasets/ (CCMatrix, Europarl, OpenSubtitles, ParaCrawl, WikiMatrix)"
    warn "Preskakujem krok 1 a ďalšie kroky závislé od surových dát."
  else
    info "Spúšťam create_datasets.py ..."
    python create_datasets.py
    ok "dataset_SK_EN.tsv a dataset_EN_SK.tsv vytvorené"
  fi

  # ── Krok 2: Štatistiky datasetu ─────────────────────────────────────────
  header "Krok 2/6 — Štatistiky datasetu (stats_datasets100k.py)"

  if [[ -f "dataset_EN_SK.tsv" && -f "dataset_SK_EN.tsv" ]]; then
    info "Spúšťam stats_datasets100k.py ..."
    python stats_datasets100k.py
    ok "Štatistiky uložené → statistika_korpusu_100k.csv + grafy_100k/"
  else
    warn "dataset_EN_SK.tsv / dataset_SK_EN.tsv chýbajú — preskakujem"
  fi

  # ── Krok 3: Filtrovanie korpusu ─────────────────────────────────────────
  header "Krok 3/6 — Filtrovanie korpusu (build_balanced_corpus.py)"

  if [[ -f "dataset_EN_SK.tsv" && -f "dataset_SK_EN.tsv" ]]; then
    info "Spúšťam build_balanced_corpus.py ..."
    python build_balanced_corpus.py
    ok "filtered_SK_EN.tsv a filtered_EN_SK.tsv vytvorené"
  else
    warn "dataset TSV súbory chýbajú — preskakujem"
  fi

  # ── Krok 4: Evaluačné vzorky ─────────────────────────────────────────────
  header "Krok 4/6 — Evaluačné vzorky (make_eval_samples.py)"

  if [[ -f "dataset_EN_SK.tsv" && -f "dataset_SK_EN.tsv" ]]; then
    info "Spúšťam make_eval_samples.py ..."
    python make_eval_samples.py
    ok "20 eval vzoriek vytvorených v eval_samples/"
  else
    warn "dataset TSV súbory chýbajú — preskakujem"
  fi

  # ── Krok 5: Štatistiky eval vzoriek ─────────────────────────────────────
  header "Krok 5/6 — Štatistiky eval vzoriek (eval_samples_stats.py)"

  if [[ -d "eval_samples" ]] && ls eval_samples/*.tsv &>/dev/null; then
    info "Spúšťam eval_samples_stats.py ..."
    python eval_samples_stats.py
    ok "Štatistiky eval vzoriek hotové"
  else
    warn "eval_samples/ je prázdny — preskakujem"
  fi

  # ── Krok 6: Hĺbková analýza korpusu (voliteľné) ─────────────────────────
  header "Krok 6/6 — Hĺbková analýza korpusu (corpus_analysis.py)"

  if [[ -f "dataset_EN_SK.tsv" && -f "dataset_SK_EN.tsv" ]]; then
    info "Spúšťam corpus_analysis.py ..."
    python corpus_analysis.py
    ok "Analýza uložená → corpus_analysis_report.txt + grafy_analyza/"
  else
    warn "dataset TSV súbory chýbajú — preskakujem"
  fi

fi  # END skip-data

# =============================================================================
#  FÁZA 2 — EVALUÁCIA MODELOV
# =============================================================================

header "Fáza 2 — Evaluácia všetkých 8 modelov (run_all_evaluations.py)"

# Skontroluj eval_samples
EVAL_OK=false
if [[ -d "eval_samples" ]] && ls eval_samples/*.tsv &>/dev/null; then
  EVAL_OK=true
elif [[ -d "models/eval_samples" ]] && ls models/eval_samples/*.tsv &>/dev/null; then
  EVAL_OK=true
fi

if [[ "$EVAL_OK" == false ]]; then
  fail "eval_samples/ neobsahuje žiadne TSV súbory. Spusti najprv make_eval_samples.py"
fi

# Skontroluj .env
if [[ ! -f "models/.env" ]]; then
  warn "models/.env neexistuje — LLM metódy (01, 02) nebudú fungovať bez OPENAI_API_KEY"
  warn "Vytvor ho: echo 'OPENAI_API_KEY=sk-...' > models/.env"
fi

info "Spúšťam run_all_evaluations.py ..."
python models/run_all_evaluations.py

# =============================================================================
#  HOTOVO
# =============================================================================

echo ""
echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}  ✅ Celý pipeline úspešne dokončený!${NC}"
echo -e "${GREEN}============================================================${NC}"
echo "  Koniec: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "  Výsledky:"
echo "  📊  models/results/comparison_report_*.txt"
echo "  📈  models/results/comparison_plot_*.png"
echo "  📁  models/results/ (všetky detailné výsledky)"
echo ""