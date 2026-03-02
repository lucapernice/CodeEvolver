# Run Audit Report — 2026-03-02

## Scope
- Project: `NLP_project`
- Goal: eseguire una run completa dell’algoritmo evolutivo e rilevare errori/problemi/inefficienze.
- Execution mode: run reale con Nebius, budget moderato.

## Run configuration used
- Command: `/home/lucapernice/NLP_project/.venv/bin/python main.py`
- Config overrides for this run only:
  - `population_size: 6` (was 15)
  - `generations: 4` (was 15)
- Model: `meta-llama/Llama-3.3-70B-Instruct`
- Test file used at runtime: `dataset2.txt`
- Run log id: `11505a0b-2cc9-47d7-938d-82a1bccb286a`

## Artifacts produced
- CSV: `logs/11505a0b-2cc9-47d7-938d-82a1bccb286a.csv`
- LOG: `logs/11505a0b-2cc9-47d7-938d-82a1bccb286a.log`
- Evolved output: `evolved_compression.c`

## Outcome summary
- Run completed successfully (`Evolution completed!`).
- Best fitness achieved: `89.39`.
- No improvement over best individuals already present in initial population (plateau early).

### Quantitative summary (from CSV)
- Total evaluations: `30` (5 generations incl. gen 0 × 6 individuals)
- `fitness_max`: `89.39`
- `fitness_avg`: `42.19`
- Zero-fitness individuals: `14/30` (46.7%)
- Integrity success: `16/30`
- Integrity failure: `14/30` (46.7%)
- Compression ratio average: `0.951033`
- Individuals with `compression_ratio > 1.0`: `18/30`

### Trend by generation
- Gen 0: best 89.39, avg 32.46, zeros 2
- Gen 1: best 89.39, avg 14.90, zeros 5
- Gen 2: best 89.39, avg 44.41, zeros 3
- Gen 3: best 89.39, avg 59.59, zeros 2
- Gen 4: best 89.39, avg 59.59, zeros 2

Interpretation: fitness massimo stabile e non cresce dopo inizializzazione; migliorano solo stabilità media e tasso di individui validi.

---

## Errors and issues observed during execution

### 1) Frequent invalid individuals from LLM output
Observed in runtime:
- C compile failures due to inconsistent generated code (undefined `struct Node`, bad function declarations).
- Example compiler errors seen:
  - `invalid use of undefined type 'struct Node'`
  - `assignment of read-only location 'compressed_data[j]'`

Impact:
- Many offspring are discarded or scored as zero, reducing effective search quality.

Severity: **P1**

---

### 2) Aggregation fallback triggered by missing parsed fields
Observed in runtime:
- `Error calculating average fitness: 'fitness'`

Likely root cause:
- `parse_test_output()` may return partial metrics when output format is incomplete; later aggregation assumes all keys exist.
- In `compile_and_test()`, averages use direct indexing (`r['fitness']`, etc.), which throws if one key is missing.

Impact:
- Entire evaluation falls back to default zero metrics even when partial useful metrics exist.
- Can bias selection pressure toward noisy defaults.

Severity: **P1**

---

### 3) Logging observability gap with `logs: False`
Observed:
- `.log` file contains only header metadata, no runtime diagnostics.
- CSV is always written, but textual diagnostics are suppressed.

Impact:
- Harder post-mortem debugging of LLM failures/parse edge cases unless terminal output is retained.

Severity: **P2**

---

### 4) Configuration mismatch risk (runtime ignores configured test files)
Code-level issue:
- `main.py` reads `test_files`, but `CodeEvolver` constructor does not accept/use it.
- `CodeEvolver` hardcodes `self.test_files = ["dataset2.txt"]`.

Impact:
- User may think multiple datasets are used while runtime always evaluates only one file.
- Overfitting risk and misleading experimental setup.

Severity: **P1**

---

## Inefficiencies and non-optimal patterns

### A) LLM usage inefficiency
- No retry/backoff on LLM errors in mutation/crossover or initial generation calls.
- No caching/deduplication for repeated or near-duplicate prompts/parents.
- Full function bodies are repeatedly sent; no token-budget optimization strategy.

Impact:
- Higher cost/latency, fragile under transient API/rate-limit issues.

Priority: **P1**

### B) Search inefficiency in evolution loop
- High fraction of invalid offspring produces wasted compile/test cycles.
- Selection pressure is diluted by frequent default-zero fallbacks.

Impact:
- Lower effective exploration depth for the same budget.

Priority: **P1**

### C) Logging I/O pattern inefficiency
- `add_log()` and `add_csv_row()` open/write/close per call.
- For larger runs this can become unnecessary I/O overhead.

Impact:
- Added overhead and potential write amplification on long experiments.

Priority: **P2**

---

## Recommended fixes (high impact)

### P0 / P1 (first)
1. Pass `test_files` from `main.py` into `CodeEvolver` and remove hardcoded dataset list.
2. Make `compile_and_test()` aggregation robust using defaults (`dict.get`) per metric instead of hard-failing on missing keys.
3. Add LLM retry policy with exponential backoff and bounded retries for transient errors.
4. Strengthen output-contract validation from LLM before compile (reject/repair obviously broken function signatures/types).

### P2 (next)
5. Add structured run metadata to CSV (model, temperature, pop, generations, timestamp, run id).
6. Improve logging strategy:
   - keep minimal diagnostic logs even when `logs: False` (warnings/errors only),
   - buffer writes or use Python `logging` handlers for efficiency.
7. Add optional smoke mode in config for quick health checks before expensive runs.

---

## What this run suggests about current system behavior
- Pipeline is operational end-to-end.
- Main bottleneck is not orchestration, but output validity of generated C candidates and robustness of evaluation aggregation.
- Current process reaches a decent valid solution quickly, then plateaus without clear gains.

## Repro notes
- Date: 2026-03-02
- Workspace: `/home/lucapernice/NLP_project`
- Python env: `.venv` (3.12.x)

---

## Post-fix verification (same date)

### Objective
- Verificare che dopo i fix non compaia più l'errore runtime `Error calculating average fitness: 'fitness'`.

### Verification run setup
- Command: `/home/lucapernice/NLP_project/.venv/bin/python main.py`
- Temporary config for verification: `population_size: 6`, `generations: 4` (poi ripristinata a `15/15`).
- Run log id: `18df8fee-67e0-4af5-93ea-0be3be846e58`

### Verification result
- ✅ L'errore `Error calculating average fitness: 'fitness'` **non è apparso** durante la run.
- ✅ La run è terminata correttamente con `Evolution completed!`.
- ✅ CSV generato correttamente: `logs/18df8fee-67e0-4af5-93ea-0be3be846e58.csv`.

### Notes
- Restano presenti failure di integrità/timeout su alcuni individui generati dal LLM (problema noto della qualità/validità del codice C generato), ma senza crash della pipeline di aggregazione fitness.
