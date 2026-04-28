# Knowing When to Ask: A Human-in-the-Loop Text-to-SQL System

A LangGraph-powered Text-to-SQL pipeline with multi-signal ambiguity detection, composite confidence gating, and Human-in-the-Loop (HITL) clarification.

Companion code for the NeurIPS 2026 submission:  
**"Knowing When to Ask: A Human-in-the-Loop Text-to-SQL System with Multi-Signal Ambiguity Detection"**

---

## Quick Start

### 1. Install dependencies

```bash
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements-benchmarks.txt
```

### 2. Set your LLM backend

**Option A — OpenAI (used for paper results)**
```bash
export OPENAI_API_KEY="sk-..."
export LLM_PROVIDER=openai
```

**Option B — Local via Ollama (free, no rate limits)**
```bash
# Install Ollama from https://ollama.com, then:
ollama pull qwen2.5-coder:14b
export LLM_PROVIDER=ollama
export OLLAMA_MODEL=qwen2.5-coder:14b
```

> **Note on local models:** Ollama does not expose token log-probabilities, so the log-prob signal contributes a neutral 0.5 to composite confidence. Smaller local models also produce more false-positive HITL triggers. OpenAI GPT-4o was used for all paper benchmark results.

### 3. Run the interactive demo

```bash
python app.py
# Open http://localhost:5000
```

---

## Running the Benchmarks

All benchmark scripts live in `scripts/`. Run from the project root with the virtual environment active.

### Ambiguity benchmark (n=67, paper primary result)

```bash
python scripts/run_ambiguity_benchmark.py --paper-run
```

Outputs results to `data/ambiguity_benchmark_results.jsonl` and a summary to `data/ambiguity_benchmark_summary.json`.

### AmbiQT adversarial slice (n=20)

```bash
# First build the AmbiQT databases (one-time, requires Spider data in data/spider/)
python scripts/build_ambiqt_dbs.py

python scripts/run_ambiqt.py --paper-run
```

### Spider domain-balanced evaluation (n=194, 20 databases)

```bash
# Download Spider first (if not already present):
python scripts/download_spider.py

# Sample the balanced 194-question set (seed=42, already committed):
# data/spider/spider_val_balanced_200.jsonl

# Run:
LLM_PROVIDER=openai python scripts/run_large_benchmark.py \
  --input-file data/spider/spider_val_balanced_200.jsonl \
  --output     data/spider/spider_200_balanced_results.jsonl \
  --summary    data/spider/spider_200_balanced_summary.json \
  --paper-run  --run-name spider_200_balanced
```

### Weight sensitivity analysis

```bash
python scripts/weight_sensitivity.py
```

### Bootstrap confidence intervals

```bash
python scripts/compute_confidence_intervals.py
```

---

## Reproducing Paper Results

Sealed benchmark runs are committed under `data/benchmark_runs/`. Each run directory contains:

| File | Contents |
|---|---|
| `results.jsonl` | Per-question output with all signals |
| `summary.json` | Aggregate metrics |
| `calibration.json` | Gate thresholds used |
| `manifest.json` | Metadata and run config |

To refit the quality-gate threshold from a sealed run without re-running the LLM:

```bash
python scripts/calibrate_threshold.py \
  --results data/benchmark_runs/ambiguity_aligned_20260425/results.jsonl
```

---

## Project Structure

```
├── app.py                          # Flask demo server (SSE streaming, HITL feedback)
├── src/
│   ├── graph/
│   │   ├── pipeline.py             # LangGraph pipeline construction
│   │   ├── nodes.py                # All pipeline node implementations
│   │   ├── edges.py                # Conditional routing logic
│   │   └── state.py                # Pipeline state schema
│   ├── llm/
│   │   └── client.py               # LLM provider abstraction (OpenAI / Ollama)
│   ├── data/
│   │   ├── vector_store.py         # ChromaDB vector store for few-shot RAG
│   │   └── benchmark_store.py      # Benchmark result storage
│   └── sandbox/
│       └── executor.py             # Sandboxed SQL execution (do not modify)
├── scripts/
│   ├── run_ambiguity_benchmark.py  # Primary benchmark runner
│   ├── run_large_benchmark.py      # Spider / large-scale runner
│   ├── run_ambiqt.py               # AmbiQT adversarial benchmark
│   ├── sample_balanced_spider.py   # Domain-balanced Spider sampler
│   ├── calibrate_threshold.py      # Gate threshold refitting
│   ├── compute_confidence_intervals.py  # Bootstrap CIs (1000 resamples)
│   ├── weight_sensitivity.py       # Composite weight perturbation analysis
│   ├── reannotate_benchmark.py     # LLM-based IAA re-annotation
│   └── plot_benchmark_results.py   # Figure generation
├── data/
│   ├── ambiguity_benchmark.json    # 67-item hand-labeled benchmark
│   ├── ambiguity_benchmark_results.jsonl  # Sealed primary run results
│   ├── ambiqt_results.jsonl        # Sealed AmbiQT run results
│   ├── calibration.json            # Active gate configuration
│   ├── few_shot_examples.json      # RAG example pool
│   ├── schema_catalog.json         # Database schema registry
│   └── benchmark_runs/             # Timestamped sealed run bundles
└── SQL_Proj/
    ├── neurips_2026.tex            # Paper source
    ├── neurips_2026.sty            # NeurIPS 2026 style file
    └── 04_figures/                 # All paper figures
```

---

## Pipeline Architecture

```
disambiguate_node
      │
      ├─[ambiguous]──▶  HITL pause (pre-generation)
      │
      ▼
generate_draft_sql ──▶ retrieve_examples ──▶ generate_sql
                                                    │
                                                    ▼
                                          consistency_check
                                                    │
                                    ├─[high entropy]──▶ HITL pause
                                                    │
                                                    ▼
                                           evaluate_sql  (composite gate)
                                                    │
                                    ├─[low confidence]──▶ HITL pause
                                                    │
                                                    ▼
                                            execute_sql
                                                    │
                                         ├─[error]──▶ debug ──▶ retry (max 2)
                                                    │
                                                    ▼
                                               ✅ result
```

**Three HITL pause points:**
1. **Pre-generation** — Disambiguation node detects semantic ambiguity
2. **Consistency check** — Execution/semantic entropy exceeds threshold
3. **Quality gate** — Composite confidence score < 0.70

---

## Configuration

Gate thresholds are stored in `data/calibration.json`. Key parameters:

| Parameter | Default | Description |
|---|---|---|
| `confidence_threshold` | 0.55 | Self-reported confidence floor |
| `disambiguation_clear_confidence` | 0.70 | Composite gate minimum |
| `num_sql_variations` | 3 | Candidates for entropy calculation |
| `max_retries` | 2 | Self-correction attempts |

Composite confidence formula (Equation 1 in paper):

```
c_composite = 0.65·c_self + 0.15·c_exec + 0.10·c_sem + 0.10·c_logprob
```

---

## License

MIT
