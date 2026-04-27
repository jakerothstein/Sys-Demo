# Paper writeup brief (for a downstream writing agent)

Use this file together with the numbered folders in `paper_writeup_package/`. The JSON files are the source of truth for numbers; restate them exactly in the paper and cite the run IDs below.

## 1. What this work is

A **LangGraph-based text-to-SQL agent** for research/demo that combines:

- Multi-stage **SQL generation** with validation and self-correction.
- **Human-in-the-loop (HITL)**: the pipeline can **pause** when ambiguity or multi-signal uncertainty is high, instead of silently returning a wrong answer.
- **Semantic / execution consistency checks** (execution entropy, semantic entropy, structural diversity) and a **calibrated quality gate** (`data/calibration.json`).

The engineering goal of the *latest* configuration was to **favor usability** (fewer unnecessary pauses) while still surfacing high-risk cases. The evaluation measures **(a)** agreement with hand labels on when to ask the user, and **(b)** recall on **AmbiQT**-style questions where *every* item is deliberately ambiguous (should usually trigger HITL).

**Do not claim** state-of-the-art on Spider/BIRD execution accuracy unless you add those experiments. The current evidence is about **interactive disambiguation behavior**, not end-to-end SQL SOTA.

---

## 2. Headline results (copy into tables; verify in JSON)

### 2.1 Hand-labeled ambiguity benchmark

- **File:** `01_source_metrics/ambiguity_benchmark_summary.json` (duplicate in `05_sealed_paper_runs/ambiguity_2026-04-24T205029Z/summary.json`)
- **Run timestamp:** `2026-04-24T205029Z`
- **N:** 67
- **Positive class:** “needs clarification” (gold `expected_hitl: true`); **prediction** = pipeline ends in `paused_hitl` (`predicted_hitl: true`).

| Metric | Value |
|--------|--------|
| Accuracy | 0.612 (61.2%) |
| Precision (HITL) | 0.559 |
| Recall (HITL) | 1.000 |
| F1 (HITL) | 0.717 |
| Specificity (answerable) | 0.235 |

**Confusion matrix counts:** TP 33, FP 26, TN 8, FN 0 (FN = 0 means **no** gold ambiguous question was “answered through” without pausing—strong safety on that axis; many FP = still pausing on answerable items).

**By database (from same summary):** `company_analytics` n=40, `sample_company` n=14, `sec_data_v2` n=13 — see JSON for per-db metrics.

**Per-question raw rows:** `02_labeled_per_question/ambiguity_benchmark_results.jsonl` (and sealed copy under `05_sealed_paper_runs/.../results.jsonl`).

**Gold questions + metadata:** `01_source_metrics/ambiguity_benchmark_gold_labeled.json` (in-repo this is `data/ambiguity_benchmark.json`).

**CLI / provenance:** `resume: true` in summary; `git_commit` in `05_sealed_paper_runs/ambiguity_2026-04-24T205029Z/manifest.json` is `62597a5b36e035600fc9aa010cadef75b8203dae`.

**Environment caveat:** the ambiguity summary’s `environment` block may show **empty** `OLLAMA_MODEL`; the AmbiQt run (below) records **`qwen2.5-coder:14b`**. For the paper, **standardize** on one line: e.g. “Local inference via Ollama, model `qwen2.5-coder:14b`” if that matches the actual run, and state that metadata logging was incomplete on one of the two summaries.

### 2.2 AmbiQT (constructed ambiguity; recall of HITL)

- **File:** `01_source_metrics/ambiqt_summary.json`
- **Run timestamp:** `2026-04-24T145816Z` (**different day/time** than the full ambiguity re-run; mention as limitation or rerun AmbiQT on the same commit for alignment)
- **Design:** 10 **col-synonyms** + 10 **tbl-synonyms** (materialized DBs; every question is **ambiguous by construction**; success = pipeline **flags HITL**)
- **Model line in file:** `qwen2.5-coder:14b`

| Metric | Value |
|--------|--------|
| Recall (flagged HITL) | 0.95 (19/20) |
| tbl-synonyms (n=10) | 1.00 |
| col-synonyms (n=10) | 0.90 |
| n_errors | 0 |

**Per-question + sealed bundle:** `05_sealed_paper_runs/ambiqt_2026-04-24T145816Z/`

---

## 3. Calibration and “usability” preset

- **File:** `03_calibration/calibration.json` (sealed copies also under `05_sealed_paper_runs/*/calibration.json`)
- The project tuned thresholds (e.g. **lower** `composite_confidence_min`, **looser** entropy caps, stricter “unanimous structural divergence” conditions) to **reduce false HITL** on clear queries, at the cost of more accepted answers. **Quantitative** before/after is best told as: earlier hand-labeled run (e.g. in git history) vs `205029Z` if you have both summaries; the **current** summary reflects the post-tuning behavior.

**Writing note:** `scripts/calibrate_threshold.py` exists for **conformal-style** refitting of `composite_confidence_min` from labeled runs; the shipped JSON may be hand-tuned + comments—read the file’s `_comment` and thresholds when describing methods.

---

## 4. Figures

- **Folder:** `04_figures/` — seven PNGs + `README.json` (figure manifest; paths inside JSON point at original repo `data/benchmark_plots/`).
- **Regenerate to match latest summaries:** from repo root, `python scripts/plot_benchmark_results.py` (requires matplotlib), then re-copy into this package if needed.

**Suggested figure captions (paper):**

- Ambiguity: overall bar metrics + confusion matrix (`00_*.png`, `00b_*.png`).
- Ambiguity by database (`01_*.png`).
- AmbiQT: caught vs missed, recall by subtype/db (`02–04`).
- Side-by-side comparison (`05_benchmark_comparison.png`) — use only if both underlying summaries are the ones you want to stand behind.

---

## 5. Codebase / architecture (for “System” section)

- **`07_project_context/README_project.md`** — overview, feature list, diagram ASCII, setup (Flask app, Ollama options).
- **`07_project_context/AGENTS.md`** — project rules (LangGraph JSON, do not modify `src/sandbox/executor.py` boundaries, benchmark when changing SQL generation logic).
- **Implementation paths (for the paper to reference without pasting code):** `src/graph/nodes.py` (e.g. disambiguation, consistency, HITL finalization), `src/graph/edges.py` (quality gate, entropy routing), `src/graph/pipeline.py`, `app.py` (Flask entry).

---

## 6. Reproducibility commands (for an appendix or footnote)

```text
# Hand-labeled ambiguity (full set; produces summary + can use --paper-run)
python scripts/run_ambiguity_benchmark.py --paper-run

# AmbiQT slice (after materializing DBs per repo README)
python scripts/run_ambiqt.py --paper-run

# Figures
python scripts/plot_benchmark_results.py

# Optional: bundle (script may exist in repo as paper_prep)
python scripts/paper_prep.py bundle
```

**Artifact layout:** `scripts/benchmark_artifacts.py` documents `write_run_bundle` (manifest + results + summary + calibration snapshot).

---

## 7. What to be careful about (limitations paragraph)

- **Sample size:** 67 + 20 is adequate for a **demo** narrative, not for tight CIs; avoid over-claiming.
- **Run alignment:** AmbiQt vs hand-labeled run **timestamps** differ; same **model** is assumed—**rerun AmbiQt** for a clean single table if reviewers push back.
- **HITL definition:** `paused_hitl` bundles **disambiguation** pauses, **consistency/entropy** pauses, and **quality-gate** pauses; the paper can briefly say “any pause requiring user input.”
- **No substitute for standard SQL execution benchmark** in this package—add Spider/BIRD subset numbers if the venue expects them.

---

## 8. Suggested paper section skeleton (for the writing agent)

1. **Abstract** — Problem: silent wrong SQL; Approach: HITL + multi-signal gate; **Two** evals: hand-labeled + AmbiQt slice; main finding: high recall for “should clarify,” remaining gap = false pauses.
2. **System** — LangGraph flow; HITL points; figure from README/pipeline.
3. **Method (signals)** — Entropy, composite confidence, calibration file.
4. **Experiments** — Table 1 hand-labeled; Table 2 AmbiQt; optional qualitative examples from JSONL.
5. **Limitations** — n, model, no Spider/BIRD table here, run alignment.
6. **Reproducibility** — commit hash, commands, `05_sealed_paper_runs/`.

---

## 9. File tree (this package)

```text
paper_writeup_package/
  MANIFEST.json
  PAPER_BRIEF_FOR_LLM.md          ← you are here
  01_source_metrics/
  02_labeled_per_question/
  03_calibration/
  04_figures/
  05_sealed_paper_runs/
  06_reproducibility/
  07_project_context/
```

**End of brief.**
