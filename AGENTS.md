# Text-to-SQL R&D Guidelines
- **Architecture**: LangGraph state machine with LLM nodes returning strictly typed JSON.
- **Rules**: Never modify the `src/sandbox/executor.py` execution boundaries.
- **Testing**: Any change to SQL generation logic must be validated using `python scripts/run_benchmark.py`.