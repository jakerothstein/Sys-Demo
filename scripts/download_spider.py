#!/usr/bin/env python3
"""
Download Spider dataset via HuggingFace `datasets`.
Automatically formats the 1,034 validation questions into a JSONL 
format ready for our benchmark runners.
"""

import json
import os
from pathlib import Path
from datasets import load_dataset

DATA_DIR = Path(__file__).parent.parent / "data"

def main():
    print("📦 Downloading Spider validation set via HuggingFace datasets...")
    # Load only the validation split (1,034 examples)
    dataset = load_dataset("spider", split="validation")
    
    spider_dir = DATA_DIR / "spider"
    spider_dir.mkdir(parents=True, exist_ok=True)
    
    out_file = spider_dir / "spider_val.jsonl"
    
    print(f"Formatting {len(dataset)} examples...")
    
    with open(out_file, "w", encoding="utf-8") as f:
        for idx, row in enumerate(dataset):
            # Map the HF spider format to our benchmark runner format
            record = {
                "id": f"spider_val_{idx}",
                "question": row["question"],
                "database": row["db_id"],
                "expected_sql": row["query"],
                # We assume all spider validation questions are clear (answerable) 
                # as they have ground-truth SQL
                "expected_hitl": False
            }
            f.write(json.dumps(record) + "\n")
            
    print(f"✓ Saved formatted Spider dataset to: {out_file}")

if __name__ == "__main__":
    main()
