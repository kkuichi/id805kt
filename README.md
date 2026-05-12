# Strojový preklad EN ↔ SK – evaluačný framework

Tento repozitár obsahuje zdrojový kód experimentálneho frameworku vytvoreného v rámci bakalárskej práce **„Strojový preklad pomocou veľkých jazykových modelov“**.

Projekt sa zameriava na porovnanie vybraných prístupov ku strojovému prekladu medzi angličtinou a slovenčinou. Hodnotenie je založené na metrikách **BLEU**, **chrF**, **COMET**, meraní rýchlosti inferencie vo forme **ms/token** a analýze stability výsledkov pomocou štandardnej odchýlky naprieč evaluačnými vzorkami.

---

## Charakteristika projektu

Cieľom projektu nie je vytvoriť nový prekladový model od začiatku, ale pripraviť jednotný experimentálny rámec na porovnanie viacerých dostupných prekladových konfigurácií. Framework umožňuje spracovať paralelný korpus, vytvoriť evaluačné vzorky, spustiť jednotlivé prekladové prístupy a následne vyhodnotiť ich kvalitu, rýchlosť a stabilitu.

Experiment sa zameriava na dva smery prekladu:

- **EN → SK** – preklad z angličtiny do slovenčiny,
- **SK → EN** – preklad zo slovenčiny do angličtiny.

Slovenčina je morfologicky bohatý jazyk, pri ktorom je potrebné sledovať nielen formálnu zhodu s referenčným prekladom, ale aj významovú primeranosť, stabilitu výstupov a praktickú použiteľnosť jednotlivých modelov.

---

## Porovnávané konfigurácie

V rámci experimentu bolo porovnávaných osem prekladových konfigurácií:

| # | Konfigurácia | Typ prístupu | Úloha v experimente |
|---|---|---|---|
| 1 | Zero-shot LLM | promptovaný veľký jazykový model | preklad bez ukážkových príkladov |
| 2 | Few-shot LLM | promptovaný veľký jazykový model | preklad s príkladmi v prompte |
| 3 | MarianMT | neurónový model strojového prekladu | rýchly baseline |
| 4 | mT5 bez fine-tuningu | multilingválny text-to-text model | všeobecný baseline |
| 5 | NLLB-200 | multilingválny model strojového prekladu | hlavný multilingválny model |
| 6 | LoRA variant | adaptačný prístup | test efektívnej adaptácie |
| 7 | NLLB + Backtranslation | pomocný prekladový variant | spätná kontrola výstupu |
| 8 | NLLB + N-best Reranking | dekódovací/reranking variant | výber z viacerých kandidátnych prekladov |

---

## Použité dáta

Experimentálny korpus bol vytvorený z piatich verejne dostupných paralelných zdrojov:

- **Europarl**,
- **CCMatrix**,
- **OpenSubtitles**,
- **ParaCrawl**,
- **WikiMatrix**.

Z každého zdroja bolo použitých 20 000 vetných párov, čím vznikol základný vyvážený dataset s rozsahom **100 000 paralelných vetných párov**.

Pri spracovaní dát boli aplikované kroky čistenia a filtrovania, najmä:

- odstránenie prázdnych alebo technicky poškodených riadkov,
- odstránenie duplicít,
- kontrola dĺžky segmentov,
- kontrola pomeru dĺžok medzi jazykmi,
- odstránenie URL, HTML značiek, e-mailových adries a nadmerného technického šumu,
- kontrola potenciálne nekvalitných zarovnaní.

Surové dáta a veľké výstupné súbory nie sú súčasťou repozitára z dôvodu veľkosti a licenčných obmedzení použitých korpusov.

---

## Metodológia

Experimentálny postup pozostáva z nasledujúcich krokov:

1. vytvorenie vyváženého paralelného korpusu,
2. čistenie a filtrovanie dát,
3. štatistická analýza korpusu,
4. generovanie evaluačných vzoriek,
5. spustenie prekladových konfigurácií,
6. výpočet metrík BLEU, chrF a COMET,
7. meranie rýchlosti inferencie v ms/token,
8. výpočet priemerných hodnôt a štandardnej odchýlky,
9. generovanie výstupných tabuliek a grafov.

Pre rýchlejšie konfigurácie boli použité väčšie evaluačné vzorky. Výpočtovo náročnejšie konfigurácie boli hodnotené na menších vzorkách, čo predstavuje metodologický kompromis medzi rozsahom evaluácie a technickou realizovateľnosťou experimentu.

---

## Štruktúra repozitára

```text
.
├── build_balanced_corpus.py        # zostavenie vyváženého korpusu
├── corpus_analysis.py              # štatistická analýza korpusu
├── create_datasets.py              # príprava datasetov
├── eval_samples_stats.py           # štatistika evaluačných vzoriek
├── make_eval_samples.py            # tvorba evaluačných vzoriek
├── stats_datasets100k.py           # štatistiky 100k korpusu
├── run_pipeline.sh                 # spustenie hlavnej pipeline
├── README.md                       # dokumentácia projektu
└── models/
    ├── 01_zero_shot_llm.py         # zero-shot LLM konfigurácia
    ├── 02_few_shot_llm.py          # few-shot LLM konfigurácia
    ├── 03_marianmt.py              # MarianMT konfigurácia
    ├── 04_t5_nmt.py                # mT5 konfigurácia
    ├── 05_nllb_m2m100.py           # NLLB/M2M konfigurácie
    ├── 06_fine_tuning_lora.py      # LoRA adaptácia
    ├── 07_backtranslation.py       # backtranslation variant
    ├── 08_nbest_reranking.py       # n-best reranking
    ├── run_all_evaluations.py      # agregácia výsledkov
    ├── utils.py                    # pomocné funkcie
    └── requirements.txt            # závislosti projektu
```

---

## Súbory nezahrnuté v repozitári

Z repozitára sú zámerne vylúčené najmä:

```text
.env
datasets/
data/
*.tsv
models/results/
models/lora_adapters/
grafy_*/
__pycache__/
.idea/
.DS_Store
```

Tieto súbory sú uvedené v `.gitignore`. Repozitár preto obsahuje hlavne zdrojový kód, dokumentáciu a konfiguračné súbory potrebné na reprodukciu experimentálneho postupu, nie veľké dátové alebo modelové artefakty.

---

## Inštalácia

Odporúčané prostredie:

- Python 3.10 alebo novší,
- PyTorch,
- Hugging Face Transformers,
- sacreBLEU,
- COMET,
- pandas,
- numpy,
- matplotlib.

Postup inštalácie:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r models/requirements.txt
```

Pri použití LLM konfigurácií je potrebné lokálne nastaviť API kľúč v súbore `.env`:

```bash
OPENAI_API_KEY=your_api_key_here
```

Súbor `.env` nesmie byť súčasťou repozitára.

---

## Spustenie

Celú pipeline možno spustiť pomocou shell skriptu:

```bash
bash run_pipeline.sh
```

Jednotlivé časti pipeline je možné spúšťať aj samostatne, napríklad:

```bash
python build_balanced_corpus.py
python corpus_analysis.py
python make_eval_samples.py
python models/run_all_evaluations.py
```

Niektoré skripty predpokladajú existenciu pripravených vstupných dát v očakávanej adresárovej štruktúre.

---

## Výstupy projektu

Projekt generuje najmä:

- štatistické tabuľky korpusu,
- evaluačné vzorky pre smery EN → SK a SK → EN,
- preklady jednotlivých konfigurácií,
- hodnoty metrík BLEU, chrF a COMET,
- priemerný čas spracovania v ms/token,
- štandardnú odchýlku výsledkov medzi vzorkami,
- grafy závislosti dĺžky viet,
- boxploty metrík,
- grafy pomeru kvality a rýchlosti.

Výsledné tabuľky a grafy boli použité pri spracovaní praktickej časti bakalárskej práce.

---

## Hlavné zistenia

Experiment ukázal, že jednotlivé konfigurácie majú rozdielne silné a slabé stránky.

- **MarianMT** sa ukázal ako rýchly a stabilný neurónový baseline.
- **Zero-shot LLM** predstavuje dobrý praktický kompromis medzi sémantickou kvalitou, jednoduchosťou použitia a prijateľnou časovou náročnosťou.
- **NLLB-200** je relevantný multilingválny model vhodný aj pre menej zastúpené jazyky.
- **mT5 bez fine-tuningu** dosiahol slabšie výsledky, čo ukazuje, že všeobecný text-to-text model nemusí byť bez dodatočného prispôsobenia vhodný na priamy preklad.
- **LoRA, Backtranslation a N-best Reranking** sú metodicky zaujímavé varianty, ich praktické použitie je však obmedzené vyššou časovou náročnosťou a variabilitou výsledkov.

Z výsledkov vyplýva, že neexistuje jedna univerzálne najlepšia konfigurácia. Výber vhodného riešenia závisí od toho, či je prioritou kvalita, rýchlosť, stabilita alebo flexibilita použitia.

---

## Obmedzenia

Pri interpretácii výsledkov je potrebné zohľadniť tieto obmedzenia:

- rýchlejšie a výpočtovo náročnejšie konfigurácie boli hodnotené na evaluačných vzorkách rôznej veľkosti,
- automatické metriky nenahrádzajú úplné ľudské hodnotenie,
- výsledky závisia od kvality a domény použitých paralelných dát,
- surové dáta a modelové artefakty nie sú súčasťou repozitára z dôvodu veľkosti a licenčných obmedzení.

---

## Súvis s bakalárskou prácou

Repozitár je praktickou prílohou k bakalárskej práci:

**Strojový preklad pomocou veľkých jazykových modelov**  
Technická univerzita v Košiciach  
Fakulta elektrotechniky a informatiky  
Autor: **Ilarion Dub**  
Rok: **2026**

Repozitár slúži ako dokumentácia a zdrojový kód experimentálnej časti práce.

---

## Autor

**Ilarion Dub**  
Bakalárska práca, 2026  
Technická univerzita v Košiciach  
Fakulta elektrotechniky a informatiky

---

## Licencia a použitie

Projekt bol vytvorený ako súčasť bakalárskej práce. Použitie, úprava alebo ďalšie šírenie projektu tretími stranami je možné len so súhlasom autora, pokiaľ nie je uvedené inak.
