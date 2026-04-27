# Text-to-SQL Research Pipeline

🧠 A LangGraph-powered Text-to-SQL agent with semantic disambiguation, self-correction, and Human-in-the-Loop (HITL) capabilities.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-green.svg)
![Flask](https://img.shields.io/badge/Flask-3.0+-red.svg)

---

## Overview

This project implements an advanced Text-to-SQL pipeline that converts natural language questions into executable SQL queries. Unlike traditional keyword-matching approaches, this system uses a **state machine architecture** powered by LangGraph to handle ambiguous queries, validate generated SQL, and self-correct errors.

### Key Features

- **🔄 LangGraph State Machine** — Multi-node pipeline with conditional routing
- **🤔 Semantic Disambiguation** — Detects ambiguous queries and requests clarification
- **🔧 Self-Correction** — Automatically retries failed queries with error analysis
- **👤 Human-in-the-Loop (HITL)** — Pauses for user input when confidence is low
- **📊 RAG-Enhanced Generation** — Uses few-shot examples from vector store
- **🗄️ Multi-Database Support** — Demo, BIRD-bench, Spider, and custom databases

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    LangGraph Pipeline                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────┐   ┌──────────────┐   ┌───────────────┐        │
│  │  Parse   │──▶│ Disambiguate │──▶│ Generate SQL  │        │
│  └──────────┘   └──────────────┘   └───────────────┘        │
│       │              │                    │                 │
│       │         HITL Pause                │                 │
│       │              ▼                    ▼                 │
│       │        ┌──────────┐       ┌──────────────┐          │
│       │        │  Human   │       │   Execute    │          │
│       │        │ Feedback │       │     SQL      │          │
│       │        └──────────┘       └──────────────┘          │
│       │                                   │                 │
│       │                           Error   │  Success        │
│       │                             ▼     ▼                 │
│       │                      ┌──────────────┐               │
│       └─────────────────────▶│    Debug     │◀──┐           │
│                              │   & Retry    │───┘           │
│                              └──────────────┘    (max 3)    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Pipeline Nodes

| Node | Description |
|------|-------------|
| `parse_node` | Extracts intent, tables, and entities from natural language |
| `disambiguate_node` | Detects ambiguity and decides if HITL is needed |
| `generate_sql_node` | Uses LLM to generate SQL from query + schema |
| `validate_node` | Validates SQL syntax before execution |
| `execute_node` | Runs SQL against the database safely |
| `debug_node` | Analyzes errors and prepares correction prompts |

---

## Installation

### Prerequisites

- Python 3.10+
- SQLite
- API key for OpenAI, Anthropic, or Google Gemini

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd "Sys Demo"

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install flask flask-cors langgraph langchain chromadb openai anthropic google-generativeai

# Pick an LLM backend (choose one):
export ANTHROPIC_API_KEY="your-key"   # cloud, best quality
export GOOGLE_API_KEY="your-key"      # cloud, free tier with daily quota
# ...or run a local model with Ollama (no API key, no rate limits):
#   1. Install Ollama from https://ollama.com
#   2. ollama pull qwen2.5-coder:7b   # ~4.7 GB, very strong at SQL
#   3. export LLM_PROVIDER=ollama
```

### Local model via Ollama

Use this when the cloud provider's quota is exhausted or you want to iterate
on the benchmark for free.

| Variable | Default | Purpose |
|----------|---------|---------|
| `LLM_PROVIDER` | `auto` | Set to `ollama` to force local. `auto` falls back to Ollama if no cloud key is set and a local server is running. |
| `OLLAMA_MODEL` | `qwen2.5-coder:7b` | Any tag visible in `ollama list`. `llama3.1:8b` and `granite3-dense:8b` also work. |
| `OLLAMA_HOST` | `http://localhost:11434` | Override if Ollama is on a different host/port. |
| `OLLAMA_NUM_CTX` | `8192` | Bump for very large schemas. |

**Caveats** — Ollama does not expose token-level log probabilities, so the
log-prob signal contributes neutrally (0.5) to composite confidence. Execution
entropy, semantic entropy, and self-reported confidence still drive the
ambiguity gate. Smaller models also produce weaker SQL, so expect more
false-positive HITL triggers than with Claude/Gemini — local is best for
*iterating* on the pipeline, cloud is best for *headline benchmark numbers*.

---

## Usage

### Running the Web UI

```bash
python app.py
```

Open [http://localhost:5000](http://localhost:5000) in your browser.

### Example Queries

Try these example queries to test the system:

| Query | Expected Behavior |
|-------|-------------------|
| "How many customers do we have?" | Direct SQL generation |
| "Show total sales by department" | Multi-table JOIN |
| "What are the sales?" | Triggers HITL (ambiguous) |
| "Top 5 employees by performance" | Complex aggregation |

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/query` | POST | Submit natural language query |
| `/api/feedback` | POST | Provide HITL clarification |
| `/api/databases` | GET | List available databases |
| `/api/databases/switch` | POST | Switch active database |
| `/api/databases/schema` | GET | Get current schema |
| `/api/benchmark/start` | POST | Start benchmark run |
| `/api/examples/save` | POST | Save query as few-shot example |

---

## Project Structure

```
Sys Demo/
├── app.py                  # Flask web server + API endpoints
├── src/
│   ├── graph/
│   │   ├── pipeline.py     # LangGraph pipeline construction
│   │   ├── nodes.py        # Pipeline node implementations + LLM prompts
│   │   ├── edges.py        # Conditional routing logic
│   │   └── state.py        # Pipeline state definition
│   ├── llm/
│   │   ├── client.py       # LLM client abstraction
│   │   └── providers.py    # OpenAI/Anthropic/Gemini providers
│   ├── data/
│   │   ├── schema.py       # Schema loading + formatting
│   │   ├── vector_store.py # Vector store + schema linking
│   │   └── few_shot.py     # Vector store for examples
│   └── sandbox/
│       └── executor.py     # Safe SQL execution
├── data/
│   ├── schema_catalog.json # Database schemas with FK relations
│   ├── few_shot_examples.json # Example question-SQL pairs (60+ examples)
│   ├── chroma_db/          # Vector store for RAG
│   └── custom/             # Custom databases
├── static/
│   ├── index.html          # Main UI
│   ├── style.css           # Styles
│   ├── app.js              # Frontend JavaScript
│   └── analytics.html      # Benchmark analytics dashboard
└── scripts/
    ├── create_company_db.py        # Generate test database
    └── analyze_data_for_examples.py # Generate few-shot examples from data
```

---

## Configuration

The pipeline can be configured through `PipelineConfig`:

```python
from src.graph.pipeline import PipelineConfig

config = PipelineConfig(
    confidence_threshold=0.55,  # Disambiguation back-stop; see calibration.json for quality gate
    max_retries=3,             # Self-correction attempts
    llm_provider="auto",       # auto, openai, anthropic, gemini
    use_few_shot=True,         # Enable RAG examples
    debug_mode=False           # Verbose logging
)
```

---

## Databases

### Included Databases

1. **Demo (E-Commerce)** — Customers, orders, products, order_items
2. **Company Analytics** — 10-table business database with 1,373 rows:
   - employees, departments, projects, project_assignments
   - customers, sales, products, categories
   - inventory, locations

### Adding Custom Databases

1. Place `.db` file in `data/custom/`
2. Add schema to `data/schema_catalog.json`:

```json
{
  "databases": {
    "your_database": {
      "file": "data/custom/your_database.db",
      "tables": {
        "table_name": {
          "columns": {
            "column_name": {"type": "TEXT", "description": "..."}
          }
        }
      }
    }
  }
}
```

---

## How It Works

### 1. Query Parsing
The system extracts structured information from natural language:
- **Intent**: SELECT, COUNT, SUM, etc.
- **Tables**: Which tables are relevant
- **Entities**: Specific values mentioned (dates, names, etc.)

### 2. Semantic Disambiguation
The LLM analyzes whether the query is ambiguous:
- Missing time range? ("sales" — all time or this month?)
- Unclear aggregation? ("total" — sum, count, or average?)
- Multiple interpretations? ("performance" — sales, ratings, etc.)

If confidence < threshold, the system pauses for human clarification.

### 3. SQL Generation
Uses the LLM with:
- Full database schema (including foreign keys)
- Retrieved few-shot examples (RAG)
- Disambiguation context (if provided)

**Token-Level Confidence:**
The pipeline requests token log probabilities from the LLM during generation (if supported). The log odds are converted to a linear confidence percentage using the exponential formula: `Confidence = exp(logprob)` (where `exp` is Euler's number `e` raised to the power of the log probability). This score is used in the UI to generate a confidence heatmap, where low-confidence tokens (< 95%) are highlighted in red.


### 4. Execution & Self-Correction
- Validates SQL syntax
- Executes in sandboxed environment
- On error: analyzes issue, modifies prompt, retries (up to 3x)

---

## Benchmarking

Run automated benchmarks to evaluate pipeline performance:

1. Click **🧪 Run Benchmark** in the UI
2. View results at `/static/analytics.html`

Metrics tracked:
- Success rate by difficulty
- HITL trigger frequency
- Average response time
- Error patterns

---

## Few-Shot Examples

The pipeline uses few-shot examples to improve SQL generation accuracy. Examples are stored in `data/few_shot_examples.json`.

### Saving Examples from the UI

When a query executes successfully, click the **💾 Save as Example** button in the Generated SQL card to save it as a few-shot example. This is the best way to build quality examples from real usage.

If the query required clarification (HITL), the clarification is also saved:

```json
{
    "id": "user_20260106_120530",
    "question": "What are the sales?",
    "clarification": "Show completed sales for Q1 2025",
    "sql": "SELECT * FROM sales WHERE status = 'completed' AND ...",
    "intent": "select",
    "tables": ["sales"],
    "difficulty": "medium",
    "saved_at": "2026-01-06T12:05:30.123456",
    "source": "user_saved"
}
```

### Generating Examples from Data

Run the analyzer script to generate examples based on your actual database schema:

```bash
python scripts/analyze_data_for_examples.py
```

This script:
- Auto-detects your SQLite database
- Analyzes tables and column types
- Generates realistic example queries (count, filter, aggregation, joins, etc.)
- Saves suggestions to `scripts/suggested_examples.json`

### SQLite Float Division Fix

⚠️ **Important**: When calculating discounts or percentages in SQLite, always use `100.0` (not `100`) to avoid integer division:

```sql
-- ❌ WRONG: Integer division, discount becomes 0
SELECT price * (1 - discount_percent/100) FROM sales;

-- ✅ CORRECT: Float division
SELECT price * (1 - discount_percent/100.0) FROM sales;
```

The few-shot examples include discount calculations using `100.0` to teach the LLM this pattern.

---

## Development

### Running Tests

```bash
python test_agent.py
```

### Adding New LLM Providers

Implement the `LLMProvider` interface in `src/llm/providers.py`:

```python
class MyProvider(LLMProvider):
    def generate(self, prompt: str, **kwargs) -> str:
        # Your implementation
        pass
```

---

## License

MIT License

---

## Acknowledgments

- [LangGraph](https://github.com/langchain-ai/langgraph) — State machine framework
- [BIRD-bench](https://bird-bench.github.io/) — Text-to-SQL benchmark
- [Spider](https://yale-lily.github.io/spider) — Text-to-SQL dataset
