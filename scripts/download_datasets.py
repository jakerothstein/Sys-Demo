#!/usr/bin/env python3
"""
Download and setup BIRD-bench and Spider 2.0 datasets.

This script helps you download the research benchmark datasets for Text-to-SQL evaluation.

Usage:
    python scripts/download_datasets.py bird     # Download BIRD-bench
    python scripts/download_datasets.py spider   # Download Spider
    python scripts/download_datasets.py all      # Download both
"""

import os
import sys
import subprocess
import zipfile
import shutil
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"


def create_directories():
    """Create necessary data directories."""
    (DATA_DIR / "bird").mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "spider").mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "custom").mkdir(parents=True, exist_ok=True)
    print("✓ Created data directories")


def download_bird():
    """
    Download BIRD-bench dataset.
    
    Source: https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/bird
    """
    print("\n📦 BIRD-bench Dataset")
    print("=" * 50)
    
    bird_dir = DATA_DIR / "bird"
    
    print("""
BIRD-bench (Big Bench for Large-Scale Database Grounded Text-to-SQL)
is a large-scale cross-domain dataset with 12,751 question-SQL pairs.

To download BIRD-bench:

1. Visit: https://bird-bench.github.io/
2. Register and download the dataset
3. Extract to: {bird_dir}

Expected structure:
{bird_dir}/
├── dev/
│   ├── dev.json                 # Development questions
│   └── databases/               # SQLite databases
│       ├── california_schools/
│       │   └── california_schools.sqlite
│       ├── card_games/
│       │   └── card_games.sqlite
│       └── ...
└── train/
    └── ...

Alternative: BIRD Mini-Dev (smaller subset):
- https://github.com/bird-bench/mini_benchmark
""".format(bird_dir=bird_dir))
    
    # Check if already exists
    existing = list(bird_dir.glob("**/*.sqlite"))
    if existing:
        print(f"✓ Found {len(existing)} existing BIRD databases")
        for db in existing[:5]:
            print(f"  - {db.name}")
        if len(existing) > 5:
            print(f"  ... and {len(existing) - 5} more")
    else:
        print("⚠ No BIRD databases found yet. Please download manually.")


def download_spider():
    """
    Download Spider dataset.
    
    Source: https://yale-lily.github.io/spider
    """
    print("\n📦 Spider Dataset")
    print("=" * 50)
    
    spider_dir = DATA_DIR / "spider"
    
    print("""
Spider is a complex cross-domain text-to-SQL dataset with
10,181 questions and 5,693 unique complex SQL queries.

To download Spider:

1. Visit: https://yale-lily.github.io/spider
2. Download the dataset (requires filling a form)
3. Extract to: {spider_dir}

Expected structure:
{spider_dir}/
├── dev.json                     # Development questions
├── train_spider.json            # Training questions  
├── tables.json                  # Schema information
└── database/                    # SQLite databases
    ├── academic/
    │   └── academic.sqlite
    ├── aircraft/
    │   └── aircraft.sqlite
    └── ...

Alternative quick download (Spider 2.0 Lite):
wget https://github.com/xlang-ai/Spider2/raw/main/spider2-lite.jsonl
""".format(spider_dir=spider_dir))
    
    # Check if already exists
    existing = list(spider_dir.glob("**/*.sqlite"))
    if existing:
        print(f"✓ Found {len(existing)} existing Spider databases")
        for db in existing[:5]:
            print(f"  - {db.name}")
        if len(existing) > 5:
            print(f"  ... and {len(existing) - 5} more")
    else:
        print("⚠ No Spider databases found yet. Please download manually.")


def create_sample_database():
    """Create a sample database for immediate testing."""
    import sqlite3
    
    sample_dir = DATA_DIR / "custom" / "sample_company"
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    db_path = sample_dir / "sample_company.sqlite"
    
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    cursor.executescript("""
        -- Employees table
        CREATE TABLE IF NOT EXISTS employees (
            employee_id INTEGER PRIMARY KEY,
            first_name TEXT NOT NULL,
            last_name TEXT NOT NULL,
            department TEXT,
            salary DECIMAL(10,2),
            hire_date DATE,
            manager_id INTEGER
        );
        
        -- Departments table
        CREATE TABLE IF NOT EXISTS departments (
            department_id INTEGER PRIMARY KEY,
            department_name TEXT NOT NULL,
            location TEXT,
            budget DECIMAL(15,2)
        );
        
        -- Projects table
        CREATE TABLE IF NOT EXISTS projects (
            project_id INTEGER PRIMARY KEY,
            project_name TEXT NOT NULL,
            department_id INTEGER,
            start_date DATE,
            end_date DATE,
            budget DECIMAL(15,2),
            status TEXT
        );
        
        -- Project assignments
        CREATE TABLE IF NOT EXISTS project_assignments (
            assignment_id INTEGER PRIMARY KEY,
            employee_id INTEGER,
            project_id INTEGER,
            role TEXT,
            hours_allocated INTEGER
        );
        
        -- Sample data
        INSERT OR REPLACE INTO departments VALUES
            (1, 'Engineering', 'San Francisco', 5000000),
            (2, 'Marketing', 'New York', 2000000),
            (3, 'Sales', 'Chicago', 3000000),
            (4, 'HR', 'San Francisco', 1000000);
        
        INSERT OR REPLACE INTO employees VALUES
            (1, 'John', 'Smith', 'Engineering', 120000, '2020-01-15', NULL),
            (2, 'Sarah', 'Johnson', 'Engineering', 110000, '2020-03-20', 1),
            (3, 'Mike', 'Williams', 'Marketing', 95000, '2021-06-01', NULL),
            (4, 'Emily', 'Brown', 'Sales', 85000, '2021-08-15', NULL),
            (5, 'David', 'Lee', 'Engineering', 130000, '2019-11-01', 1),
            (6, 'Lisa', 'Garcia', 'HR', 75000, '2022-01-10', NULL),
            (7, 'James', 'Wilson', 'Marketing', 88000, '2022-03-15', 3),
            (8, 'Anna', 'Martinez', 'Sales', 92000, '2020-09-01', 4);
        
        INSERT OR REPLACE INTO projects VALUES
            (1, 'Cloud Migration', 1, '2024-01-01', '2024-12-31', 500000, 'active'),
            (2, 'Brand Refresh', 2, '2024-02-01', '2024-06-30', 150000, 'active'),
            (3, 'Sales Portal', 3, '2024-03-01', '2024-09-30', 200000, 'planning'),
            (4, 'ML Pipeline', 1, '2024-04-01', '2024-10-31', 300000, 'active');
        
        INSERT OR REPLACE INTO project_assignments VALUES
            (1, 1, 1, 'Lead', 40),
            (2, 2, 1, 'Developer', 30),
            (3, 5, 1, 'Architect', 20),
            (4, 3, 2, 'Lead', 40),
            (5, 7, 2, 'Designer', 30),
            (6, 4, 3, 'Lead', 35),
            (7, 8, 3, 'Sales Rep', 25),
            (8, 2, 4, 'Developer', 20),
            (9, 5, 4, 'Lead', 30);
    """)
    
    conn.commit()
    conn.close()
    
    print(f"\n✓ Created sample database: {db_path}")
    print("  Tables: employees, departments, projects, project_assignments")


def show_status():
    """Show current dataset status."""
    print("\n📊 Dataset Status")
    print("=" * 50)
    
    for dataset in ["bird", "spider", "custom"]:
        dataset_path = DATA_DIR / dataset
        if dataset_path.exists():
            dbs = list(dataset_path.glob("**/*.sqlite"))
            print(f"\n{dataset.upper()}:")
            print(f"  Path: {dataset_path}")
            print(f"  Databases: {len(dbs)}")
            if dbs:
                for db in dbs[:3]:
                    size = db.stat().st_size / 1024
                    print(f"    - {db.name} ({size:.1f} KB)")
                if len(dbs) > 3:
                    print(f"    ... and {len(dbs) - 3} more")
        else:
            print(f"\n{dataset.upper()}: Not found")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        show_status()
        return
    
    command = sys.argv[1].lower()
    
    create_directories()
    
    if command == "bird":
        download_bird()
    elif command == "spider":
        download_spider()
    elif command == "sample":
        create_sample_database()
    elif command == "all":
        download_bird()
        download_spider()
        create_sample_database()
    elif command == "status":
        show_status()
    else:
        print(f"Unknown command: {command}")
        print("Available: bird, spider, sample, all, status")


if __name__ == "__main__":
    main()
