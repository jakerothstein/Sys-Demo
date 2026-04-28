import json
import random
from collections import defaultdict

def sample_balanced_spider(input_path, output_path, target_total=50, max_per_db=8, min_dbs=10):
    with open(input_path, 'r', encoding='utf-8') as f:
        all_questions = [json.loads(line) for line in f if line.strip()]
    
    # Group by database
    by_db = defaultdict(list)
    for q in all_questions:
        by_db[q['database']].append(q)
    
    print(f"Found {len(all_questions)} questions across {len(by_db)} databases.")
    
    # Filter databases with at least some questions
    db_ids = list(by_db.keys())
    random.shuffle(db_ids)
    
    selected_questions = []
    selected_dbs = set()
    
    # First pass: pick 1 question from at least min_dbs
    for db_id in db_ids:
        if len(selected_dbs) < min_dbs:
            q = random.choice(by_db[db_id])
            selected_questions.append(q)
            selected_dbs.add(db_id)
            by_db[db_id].remove(q)
    
    # Second pass: fill up to target_total while respecting max_per_db
    # We cycle through selected_dbs to keep it balanced
    while len(selected_questions) < target_total:
        added_in_round = 0
        # Shuffle dbs each round to avoid bias
        active_dbs = list(selected_dbs)
        random.shuffle(active_dbs)
        
        for db_id in active_dbs:
            if len(selected_questions) >= target_total:
                break
            
            # How many from this db?
            current_count = sum(1 for q in selected_questions if q['database'] == db_id)
            if current_count < max_per_db and by_db[db_id]:
                q = random.choice(by_db[db_id])
                selected_questions.append(q)
                by_db[db_id].remove(q)
                added_in_round += 1
        
        if added_in_round == 0:
            # If we still need more, try to add a new database
            remaining_dbs = [d for d in db_ids if d not in selected_dbs and by_db[d]]
            if remaining_dbs:
                new_db = random.choice(remaining_dbs)
                selected_dbs.add(new_db)
                continue
            else:
                break
                
    print(f"Selected {len(selected_questions)} questions across {len(selected_dbs)} databases.")
    db_counts = defaultdict(int)
    for q in selected_questions:
        db_counts[q['database']] += 1
    
    for db, count in sorted(db_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {db}: {count}")
        
    with open(output_path, 'w', encoding='utf-8') as f:
        for q in selected_questions:
            f.write(json.dumps(q) + '\n')
            
    print(f"Saved balanced sample to {output_path}")

if __name__ == "__main__":
    # Seed for reproducibility in the paper
    random.seed(42)
    sample_balanced_spider(
        'data/spider/spider_val.jsonl',
        'data/spider/spider_val_balanced_50.jsonl',
        target_total=50,
        max_per_db=8,
        min_dbs=10
    )
