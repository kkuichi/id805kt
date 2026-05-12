# Machine Translation EN ↔ SK — Bakalárska práca

> Porovnanie 8 metód strojového prekladu medzi angličtinou a slovenčinou.
> Vyhodnotenie pomocou BLEU, chrF a COMET metrík.

---

## ⚡ Spustenie jedným príkazom

```bash
cd /Users/ilariondub/PycharmProjects/BP
source .venv/bin/activate
./run_pipeline.sh
```

Skript **sám zistí** čo už existuje a začne od správneho kroku.

---

## Ako pipeline funguje

```
datasets/                     (surové korpusy)
    │
    ▼ create_datasets.py
dataset_EN_SK.tsv             (100 000 párov EN→SK)
dataset_SK_EN.tsv             (100 000 párov SK→EN)
    │
    ├──▶ stats_datasets100k.py     →  statistika_korpusu_100k.csv + grafy_100k/
    │
    ▼ build_balanced_corpus.py
filtered_EN_SK.tsv            (vyfiltrovaný korpus)
filtered_SK_EN.tsv
    │
    ▼ make_eval_samples.py
eval_samples/                 (20 × 100-vetových vzoriek)
    │
    ├──▶ eval_samples_stats.py     →  štatistiky vzoriek
    ├──▶ corpus_analysis.py        →  hĺbková analýza + grafy
    │
    ▼ models/run_all_evaluations.py
models/results/               (výsledky všetkých 8 modelov + grafy)
```

---

## Kroky pipelineu

### Krok 0 — Príprava prostredia (iba raz)

```bash
# Vytvor virtuálne prostredie
python3 -m venv .venv

# Aktivuj ho
source .venv/bin/activate

# Nainštaluj závislosti
pip install -r models/requirements.txt

# Voliteľne: COMET metrika (~1.8 GB, stiahne sa automaticky pri prvom použití)
pip install unbabel-comet

# Voliteľne: LoRA fine-tuning (len pre metódu 6)
pip install peft accelerate datasets

# Nastav OpenAI API kľúč
echo "OPENAI_API_KEY=sk-tvoj-kluc-sem" > models/.env
```

---

### Krok 1 — Kombinovanie korpusov → 100k párov

**Skript:** `create_datasets.py`

**Vstup:** `datasets/` so 5 paralelnými korpusmi:
```
datasets/
├── CCMatrix/       CCMatrix.en-sk.en   CCMatrix.en-sk.sk   CCMatrix.en-sk.scores
├── Europarl/       Europarl.en-sk.en   Europarl.en-sk.sk
├── OpenSubtitles/  OpenSubtitles.en-sk.en   OpenSubtitles.en-sk.sk
├── ParaCrawl/      ParaCrawl.en-sk.en   ParaCrawl.en-sk.sk
└── WikiMatrix/     WikiMatrix.en-sk.en  WikiMatrix.en-sk.sk
```

**Čo robí:**
- Z každého korpusu náhodne vyberie 20 000 párov (seed=42)
- Spája ich do jedného súboru → 5 × 20 000 = **100 000 párov**
- Generuje verziu EN→SK aj SK→EN

**Výstup:**
- `dataset_EN_SK.tsv` — 100 000 riadkov, EN zdroj | SK cieľ
- `dataset_SK_EN.tsv` — 100 000 riadkov, SK zdroj | EN cieľ

```bash
python create_datasets.py
```

---

### Krok 2 — Štatistiky datasetu

**Skript:** `stats_datasets100k.py`

**Čo robí:**
- Počíta dĺžky viet (počet slov, znakov) — min, max, priemer, medián
- Analýza distribúcie dĺžok pre EN aj SK stranu
- Pomer dĺžok zdrojových a cieľových viet
- Porovnanie medzi korpusmi (CCMatrix vs Europarl vs ...)
- Generuje vizualizácie

**Výstup:**
- `statistika_korpusu_100k.csv` — štatistiky v CSV
- `grafy_100k/` — sada PNG grafov (histogramy, box ploty, scatter, CDF)

```bash
python stats_datasets100k.py
```

---

### Krok 3 — Filtrovanie a čistenie dát

**Skript:** `build_balanced_corpus.py`

**Čo robí:**
- Odstraňuje príliš krátke vety (< 3 slová)
- Odstraňuje príliš dlhé vety (> 80 slov)
- Filtruje páry s extrémnym pomerom dĺžok (> 3.0)
- Voliteľne vyváži počet párov z každého korpusu

**Výstup:**
- `filtered_EN_SK.tsv`
- `filtered_SK_EN.tsv`

```bash
python build_balanced_corpus.py
```

---

### Krok 4 — Evaluačné vzorky

**Skript:** `make_eval_samples.py`

**Čo robí:**
- Z 100k datasetu náhodne vyberá 10 vzoriek po 100 viet (seed=42)
- Každá vzorka je iná (bez opakovania)
- Generuje pre oba smery: EN→SK aj SK→EN

**Výstup:** `eval_samples/` — 20 TSV súborov:
```
eval_samples/
├── EN_SK_sample_01.tsv  …  EN_SK_sample_10.tsv   (100 párov každý)
└── SK_EN_sample_01.tsv  …  SK_EN_sample_10.tsv   (100 párov každý)
```

```bash
python make_eval_samples.py
```

---

### Krok 5 — Štatistiky eval vzoriek

**Skript:** `eval_samples_stats.py`

**Čo robí:** Overuje kvalitu a distribúciu eval vzoriek — dĺžky, zdroje, rôznorodosť.

```bash
python eval_samples_stats.py
```

---

### Krok 6 — Hĺbková analýza korpusu (voliteľné)

**Skript:** `corpus_analysis.py`

**Čo robí:**
- TTR (Type-Token Ratio) — slovníková bohatosť
- Pearson a Spearman korelácie dĺžok párov
- Per-korpus breakdown (CCMatrix zvlášť, Europarl zvlášť, ...)
- Percentily, top-10 tokenov, duplicity
- 13+ grafov

**Výstup:**
- `corpus_analysis_report.txt`
- `corpus_analysis_stats.csv` + `.xlsx`
- `grafy_analyza/` — sada PNG grafov

```bash
python corpus_analysis.py
```

---

### Krok 7 — Evaluácia všetkých 8 modelov

**Skript:** `models/run_all_evaluations.py`

**Čo robí:**
- Spustí všetkých 8 metód postupne
- Každá metóda preloží 10 × 100 viet = 1 000 viet v oboch smeroch
- Po skončení vygeneruje porovnávaciu správu a 3×3 graf

```bash
python models/run_all_evaluations.py
```

Alebo každú metódu zvlášť:

```bash
cd models
python 01_zero_shot_llm.py
python 02_few_shot_llm.py
python 03_marianmt.py
python 04_t5_nmt.py
python 05_nllb_m2m100.py
python 06_fine_tuning_lora.py
python 07_backtranslation.py
python 08_nbest_reranking.py
cd ..
```

---

## 8 prekladových metód

| # | Metóda | Model | Stiahnutie |
|---|--------|-------|------------|
| 1 | **Zero-shot LLM** | GPT-4o-mini | API (OpenAI kľúč) |
| 2 | **Few-shot LLM** | GPT-4o-mini + príklady | API (OpenAI kľúč) |
| 3 | **MarianMT** | Helsinki-NLP/opus-mt-en-sk | ~300 MB |
| 4 | **T5 / mT5** | google/mt5-base | ~1.2 GB |
| 5 | **NLLB-200** | facebook/nllb-200-distilled-600M | ~2.5 GB |
| 6 | **LoRA Fine-tuning** | NLLB + vlastné adaptéry | ~2.5 GB + adaptéry |
| 7 | **Backtranslation** | NLLB (spätný preklad) | ~2.5 GB |
| 8 | **N-best Reranking** | NLLB (N-best + reranking) | ~2.5 GB |

---

## Metriky hodnotenia

| Metrika | Čo meria | Rozsah |
|---------|----------|--------|
| **BLEU** | Presnosť n-gramov oproti referenčnému prekladu | 0–100 ↑ |
| **chrF** | Zhoda znakových n-gramov (robustnejšia voči morfológii) | 0–100 ↑ |
| **COMET** | Neurálna metrika, citlivá na sémantiku a kontext | 0–100 ↑ |

Evaluácia: **10 vzoriek × 100 viet = 1 000 viet** pre každý smer (EN→SK a SK→EN).

---

## Výstupy

### Po krokoch 1–6 (dáta a štatistiky)
```
statistika_korpusu_100k.csv    — štatistiky 100k datasetu
grafy_100k/                    — grafy datasetu (histogramy, scatter, CDF...)
filtered_EN_SK.tsv             — vyfiltrovaný korpus
filtered_SK_EN.tsv
eval_samples/                  — 20 TSV vzoriek pre evaluáciu
corpus_analysis_report.txt     — hĺbková analýza
grafy_analyza/                 — grafy analýzy (TTR, korelácie...)
```

### Po kroku 7 (modely)
```
models/results/
├── 01_zero_shot_llm_DATUM.txt      — výsledky metódy 1
├── 01_zero_shot_llm_DATUM.png      — graf metódy 1
├── ...
├── 08_nbest_reranking_DATUM.txt
├── 08_nbest_reranking_DATUM.png
├── comparison_report_DATUM.txt     — porovnávacia tabuľka všetkých metód
├── comparison_report_DATUM.csv     — to isté v CSV
└── comparison_plot_DATUM.png       — 3×3 porovnávací graf
```

### Popis 3×3 porovnávacieho grafu
```
┌──────────────────────┬──────────────────────┬──────────────────────┐
│  EN→SK overlay       │  SK→EN overlay       │  Celkový rebríček    │
│  BLEU+chrF+COMET     │  BLEU+chrF+COMET     │  (priemer metrík)    │
├──────────────────────┼──────────────────────┼──────────────────────┤
│  BLEU porovnanie     │  chrF porovnanie     │  COMET porovnanie    │
│  (stĺpcový graf)     │  (stĺpcový graf)     │  (stĺpcový graf)     │
├──────────────────────┼──────────────────────┼──────────────────────┤
│  Čas spracovania     │  Kvalita vs Rýchlosť │  Heatmapa metrík     │
│  (stĺpcový graf)     │  (scatter plot)      │  (normalizovaná)     │
└──────────────────────┴──────────────────────┴──────────────────────┘
```

---

## Štruktúra projektu

```
BP/
├── README.md                       ← tento súbor
├── run_pipeline.sh                 ← spustí celý pipeline jedným príkazom
├── .venv/                          ← Python virtuálne prostredie
│
├── datasets/                       ← surové korpusy (prázdne v repozitári)
│   ├── CCMatrix/
│   ├── Europarl/
│   ├── OpenSubtitles/
│   ├── ParaCrawl/
│   └── WikiMatrix/
│
├── dataset_EN_SK.tsv               ← 100k párov (výstup kroku 1)
├── dataset_SK_EN.tsv               ← 100k párov (výstup kroku 1)
├── filtered_EN_SK.tsv              ← filtrovaný (výstup kroku 3)
├── filtered_SK_EN.tsv              ← filtrovaný (výstup kroku 3)
│
├── eval_samples/                   ← 20 vzoriek × 100 viet (výstup kroku 4)
│   ├── EN_SK_sample_01.tsv
│   │   ...
│   └── SK_EN_sample_10.tsv
│
├── grafy_100k/                     ← grafy štatistík datasetu
├── grafy_analyza/                  ← grafy hĺbkovej analýzy
├── statistika_korpusu_100k.csv
├── corpus_analysis_report.txt
├── corpus_analysis_stats.csv
│
├── create_datasets.py              ← Krok 1
├── stats_datasets100k.py           ← Krok 2
├── build_balanced_corpus.py        ← Krok 3
├── make_eval_samples.py            ← Krok 4
├── eval_samples_stats.py           ← Krok 5
├── corpus_analysis.py              ← Krok 6
│
└── models/
    ├── .env                        ← OPENAI_API_KEY=sk-...
    ├── requirements.txt
    ├── utils.py                    ← zdieľané utility (metriky, I/O, grafy)
    │
    ├── 01_zero_shot_llm.py
    ├── 02_few_shot_llm.py
    ├── 03_marianmt.py
    ├── 04_t5_nmt.py
    ├── 05_nllb_m2m100.py
    ├── 06_fine_tuning_lora.py
    ├── 07_backtranslation.py
    ├── 08_nbest_reranking.py
    ├── run_all_evaluations.py
    │
    ├── lora_adapters/
    │   ├── en_sk/                  ← LoRA adaptér EN→SK
    │   └── sk_en/                  ← LoRA adaptér SK→EN
    │
    └── results/                    ← výsledky evaluácií
```

---

## Spustenie pipelineu

### Automatické (odporúčané)

```bash
./run_pipeline.sh
```

Skript sám zistí čo existuje a začne od správneho kroku:

| Čo je na disku | Štart od |
|---|---|
| `datasets/` so súbormi `.en` / `.sk` | Krok 1 — plný pipeline |
| `dataset_EN_SK.tsv` + `dataset_SK_EN.tsv` | Krok 2 — štatistiky → modely |
| `filtered_EN_SK.tsv` + `filtered_SK_EN.tsv` | Krok 4 — eval vzorky → modely |
| `eval_samples/` s TSV súbormi | Krok 7 — iba modely |

### Manuálne od konkrétneho kroku

```bash
./run_pipeline.sh --from 1   # plný pipeline od začiatku
./run_pipeline.sh --from 2   # začne od štatistík
./run_pipeline.sh --from 4   # začne od eval vzoriek
./run_pipeline.sh --from 7   # iba modely
```

---

## Časté problémy

**`FileNotFoundError: dataset_SK_EN.tsv`**
```bash
# Buď spusti krok 1 (potrebuješ datasets/ s korpusmi):
python create_datasets.py

# Alebo skopíruj existujúce TSV súbory do projektu:
cp ~/Documents/Claude/BP1/dataset_SK_EN.tsv .
cp ~/Documents/Claude/BP1/dataset_EN_SK.tsv .
```

**`eval_samples/ neobsahuje žiadne TSV súbory`**
```bash
python make_eval_samples.py
```

**`OPENAI_API_KEY not found`**
```bash
echo "OPENAI_API_KEY=sk-tvoj-kluc" > models/.env
```

**`ModuleNotFoundError: No module named 'utils'`**
```bash
# Vždy spúšťaj z koreňa projektu (BP/), nie z models/
cd /Users/ilariondub/PycharmProjects/BP
python models/run_all_evaluations.py
```

**COMET nie je dostupný (iba BLEU + chrF)**
```bash
pip install unbabel-comet
# Model sa stiahne automaticky (~1.8 GB) pri prvom spustení
```

**LoRA adaptéry nenájdené (metóda 06 padá)**
```
# Metóda 06 automaticky použije základný NLLB model bez adaptérov.
# Pre vlastné adaptéry ich umiestni do:
# models/lora_adapters/en_sk/
# models/lora_adapters/sk_en/
```

**`zsh: permission denied: ./run_pipeline.sh`**
```bash
chmod +x run_pipeline.sh
./run_pipeline.sh
```# id805kt
