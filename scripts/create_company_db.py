#!/usr/bin/env python3
"""
Create a complex company analytics database for Text-to-SQL testing.
Generates 10 tables with 1000+ rows of realistic business data.
"""
import sqlite3
import os
import random
from datetime import datetime, timedelta

# Try to use Faker for realistic data, fall back to basic random data
try:
    from faker import Faker
    fake = Faker()
    HAS_FAKER = True
except ImportError:
    HAS_FAKER = False
    print("Faker not installed. Using basic random data. Run: pip install faker")


def random_name():
    if HAS_FAKER:
        return fake.name()
    first_names = ["John", "Jane", "Bob", "Alice", "Charlie", "Diana", "Eve", "Frank", "Grace", "Henry"]
    last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez"]
    return f"{random.choice(first_names)} {random.choice(last_names)}"

def random_company():
    if HAS_FAKER:
        return fake.company()
    prefixes = ["Acme", "Global", "Tech", "Prime", "Alpha", "Beta", "United", "First", "Pacific", "Atlantic"]
    suffixes = ["Corp", "Inc", "LLC", "Industries", "Solutions", "Services", "Group", "Enterprises"]
    return f"{random.choice(prefixes)} {random.choice(suffixes)}"

def random_email(name, unique_id=None):
    if HAS_FAKER:
        return fake.unique.email() if unique_id else fake.email()
    clean_name = name.lower().replace(" ", ".")
    domains = ["gmail.com", "yahoo.com", "outlook.com", "company.com", "work.org"]
    suffix = f".{unique_id}" if unique_id else ""
    return f"{clean_name}{suffix}@{random.choice(domains)}"

def random_date(start_year=2020, end_year=2024):
    start = datetime(start_year, 1, 1)
    end = datetime(end_year, 12, 31)
    delta = end - start
    random_days = random.randint(0, delta.days)
    return (start + timedelta(days=random_days)).strftime("%Y-%m-%d")

def random_phone():
    if HAS_FAKER:
        return fake.phone_number()
    return f"({random.randint(200,999)}) {random.randint(100,999)}-{random.randint(1000,9999)}"

def random_city():
    if HAS_FAKER:
        return fake.city()
    cities = ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix", "Philadelphia", 
              "San Antonio", "San Diego", "Dallas", "San Jose", "Austin", "Seattle", 
              "Denver", "Boston", "Portland", "Miami", "Atlanta", "Detroit"]
    return random.choice(cities)

def random_state():
    states = ["CA", "TX", "NY", "FL", "IL", "PA", "OH", "GA", "NC", "MI", 
              "WA", "AZ", "MA", "CO", "TN", "IN", "MO", "MD", "WI", "MN"]
    return random.choice(states)


def create_database(db_path: str):
    """Create the company analytics database."""
    
    # Remove existing database
    if os.path.exists(db_path):
        os.remove(db_path)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Enable foreign keys
    cursor.execute("PRAGMA foreign_keys = ON;")
    
    print("Creating tables...")
    
    # 1. Departments
    cursor.execute("""
        CREATE TABLE departments (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL UNIQUE,
            budget DECIMAL(12,2),
            location TEXT,
            created_date DATE
        )
    """)
    
    # 2. Employees
    cursor.execute("""
        CREATE TABLE employees (
            id INTEGER PRIMARY KEY,
            first_name TEXT NOT NULL,
            last_name TEXT NOT NULL,
            email TEXT UNIQUE,
            phone TEXT,
            department_id INTEGER,
            manager_id INTEGER,
            salary DECIMAL(10,2),
            hire_date DATE,
            job_title TEXT,
            FOREIGN KEY (department_id) REFERENCES departments(id),
            FOREIGN KEY (manager_id) REFERENCES employees(id)
        )
    """)
    
    # 3. Projects
    cursor.execute("""
        CREATE TABLE projects (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT,
            department_id INTEGER,
            budget DECIMAL(12,2),
            start_date DATE,
            end_date DATE,
            status TEXT CHECK(status IN ('planning', 'active', 'completed', 'on_hold', 'cancelled')),
            FOREIGN KEY (department_id) REFERENCES departments(id)
        )
    """)
    
    # 4. Project Assignments
    cursor.execute("""
        CREATE TABLE project_assignments (
            id INTEGER PRIMARY KEY,
            employee_id INTEGER NOT NULL,
            project_id INTEGER NOT NULL,
            role TEXT,
            hours_allocated INTEGER,
            start_date DATE,
            end_date DATE,
            FOREIGN KEY (employee_id) REFERENCES employees(id),
            FOREIGN KEY (project_id) REFERENCES projects(id)
        )
    """)
    
    # 5. Categories
    cursor.execute("""
        CREATE TABLE categories (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL UNIQUE,
            description TEXT,
            parent_category_id INTEGER,
            FOREIGN KEY (parent_category_id) REFERENCES categories(id)
        )
    """)
    
    # 6. Products
    cursor.execute("""
        CREATE TABLE products (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            sku TEXT UNIQUE,
            category_id INTEGER,
            price DECIMAL(10,2),
            cost DECIMAL(10,2),
            description TEXT,
            created_date DATE,
            is_active BOOLEAN DEFAULT 1,
            FOREIGN KEY (category_id) REFERENCES categories(id)
        )
    """)
    
    # 7. Locations
    cursor.execute("""
        CREATE TABLE locations (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            type TEXT CHECK(type IN ('warehouse', 'store', 'office', 'distribution_center')),
            address TEXT,
            city TEXT,
            state TEXT,
            capacity INTEGER
        )
    """)
    
    # 8. Inventory
    cursor.execute("""
        CREATE TABLE inventory (
            id INTEGER PRIMARY KEY,
            product_id INTEGER NOT NULL,
            location_id INTEGER NOT NULL,
            quantity INTEGER DEFAULT 0,
            reorder_level INTEGER DEFAULT 10,
            last_restocked DATE,
            FOREIGN KEY (product_id) REFERENCES products(id),
            FOREIGN KEY (location_id) REFERENCES locations(id),
            UNIQUE(product_id, location_id)
        )
    """)
    
    # 9. Customers
    cursor.execute("""
        CREATE TABLE customers (
            id INTEGER PRIMARY KEY,
            company_name TEXT NOT NULL,
            contact_name TEXT,
            email TEXT,
            phone TEXT,
            address TEXT,
            city TEXT,
            state TEXT,
            credit_limit DECIMAL(10,2),
            created_date DATE,
            is_active BOOLEAN DEFAULT 1
        )
    """)
    
    # 10. Sales
    cursor.execute("""
        CREATE TABLE sales (
            id INTEGER PRIMARY KEY,
            customer_id INTEGER NOT NULL,
            employee_id INTEGER NOT NULL,
            product_id INTEGER NOT NULL,
            quantity INTEGER NOT NULL,
            unit_price DECIMAL(10,2) NOT NULL,
            discount_percent DECIMAL(5,2) DEFAULT 0,
            sale_date DATE NOT NULL,
            payment_method TEXT CHECK(payment_method IN ('credit_card', 'bank_transfer', 'cash', 'check')),
            status TEXT CHECK(status IN ('pending', 'completed', 'refunded', 'cancelled')),
            FOREIGN KEY (customer_id) REFERENCES customers(id),
            FOREIGN KEY (employee_id) REFERENCES employees(id),
            FOREIGN KEY (product_id) REFERENCES products(id)
        )
    """)
    
    print("Inserting data...")
    
    # Insert Departments
    departments = [
        (1, "Engineering", 2500000, "Building A", "2018-01-15"),
        (2, "Sales", 1800000, "Building B", "2018-01-15"),
        (3, "Marketing", 1200000, "Building B", "2018-03-01"),
        (4, "Human Resources", 800000, "Building A", "2018-01-15"),
        (5, "Finance", 900000, "Building C", "2018-02-01"),
        (6, "Operations", 1500000, "Building D", "2018-04-01"),
        (7, "Customer Support", 600000, "Building B", "2019-01-01"),
        (8, "Research & Development", 3000000, "Building E", "2019-06-01"),
    ]
    cursor.executemany("INSERT INTO departments VALUES (?,?,?,?,?)", departments)
    
    # Insert Employees (50+)
    job_titles = {
        1: ["Software Engineer", "Senior Engineer", "Tech Lead", "DevOps Engineer", "QA Engineer"],
        2: ["Sales Rep", "Account Executive", "Sales Manager", "Business Development"],
        3: ["Marketing Specialist", "Content Writer", "SEO Analyst", "Brand Manager"],
        4: ["HR Specialist", "Recruiter", "HR Manager", "Benefits Coordinator"],
        5: ["Accountant", "Financial Analyst", "Controller", "Bookkeeper"],
        6: ["Operations Manager", "Logistics Coordinator", "Supply Chain Analyst"],
        7: ["Support Agent", "Support Lead", "Customer Success Manager"],
        8: ["Research Scientist", "Data Scientist", "ML Engineer", "Product Researcher"],
    }
    
    employees = []
    for i in range(1, 61):
        name = random_name()
        parts = name.split()
        first_name = parts[0]
        last_name = parts[-1] if len(parts) > 1 else "Unknown"
        dept_id = ((i - 1) % 8) + 1
        
        # First 8 are managers (no manager_id)
        manager_id = None if i <= 8 else ((i - 1) % 8) + 1
        
        salary = random.randint(50000, 180000)
        hire_date = random_date(2018, 2024)
        job_title = random.choice(job_titles[dept_id])
        
        employees.append((
            i, first_name, last_name, 
            random_email(name, i), random_phone(),
            dept_id, manager_id, salary, hire_date, job_title
        ))
    
    cursor.executemany(
        "INSERT INTO employees VALUES (?,?,?,?,?,?,?,?,?,?)", 
        employees
    )
    
    # Insert Projects (25)
    project_names = [
        "Website Redesign", "Mobile App v2", "CRM Integration", "Data Pipeline",
        "Cloud Migration", "Security Audit", "Q4 Campaign", "Brand Refresh",
        "Annual Report", "Hiring Initiative", "Cost Reduction", "Process Automation",
        "Customer Portal", "Analytics Dashboard", "API Platform", "ML Model Deployment",
        "Inventory System", "Supplier Network", "Support Chatbot", "Training Program",
        "Market Expansion", "Product Launch", "Partnership Program", "Compliance Review",
        "Infrastructure Upgrade"
    ]
    
    projects = []
    for i, name in enumerate(project_names, 1):
        dept_id = ((i - 1) % 8) + 1
        budget = random.randint(50000, 500000)
        start_date = random_date(2023, 2024)
        end_date = random_date(2024, 2025)
        status = random.choice(['planning', 'active', 'completed', 'on_hold'])
        
        projects.append((i, name, f"Description for {name}", dept_id, budget, start_date, end_date, status))
    
    cursor.executemany(
        "INSERT INTO projects VALUES (?,?,?,?,?,?,?,?)", 
        projects
    )
    
    # Insert Project Assignments (100+)
    assignments = []
    roles = ["Lead", "Contributor", "Reviewer", "Analyst", "Coordinator"]
    assignment_id = 1
    for project_id in range(1, 26):
        # Each project has 3-6 team members
        team_size = random.randint(3, 6)
        team_members = random.sample(range(1, 61), team_size)
        
        for emp_id in team_members:
            role = random.choice(roles)
            hours = random.randint(10, 40)
            start = random_date(2023, 2024)
            end = random_date(2024, 2025)
            
            assignments.append((assignment_id, emp_id, project_id, role, hours, start, end))
            assignment_id += 1
    
    cursor.executemany(
        "INSERT INTO project_assignments VALUES (?,?,?,?,?,?,?)", 
        assignments
    )
    
    # Insert Categories (15)
    categories = [
        (1, "Electronics", "Electronic devices and accessories", None),
        (2, "Computers", "Desktop and laptop computers", 1),
        (3, "Phones", "Mobile phones and tablets", 1),
        (4, "Accessories", "Electronic accessories", 1),
        (5, "Software", "Software and licenses", None),
        (6, "Office", "Office supplies and furniture", None),
        (7, "Furniture", "Desks, chairs, and storage", 6),
        (8, "Supplies", "Paper, pens, and consumables", 6),
        (9, "Services", "Professional services", None),
        (10, "Consulting", "Business consulting", 9),
        (11, "Training", "Training and education", 9),
        (12, "Networking", "Network equipment", 1),
        (13, "Storage", "Data storage solutions", 1),
        (14, "Security", "Security equipment and software", None),
        (15, "Cloud", "Cloud services and subscriptions", 5),
    ]
    cursor.executemany("INSERT INTO categories VALUES (?,?,?,?)", categories)
    
    # Insert Products (100)
    product_templates = [
        ("Laptop Pro {}", 2, 1200, 800),
        ("Desktop Workstation {}", 2, 1500, 1000),
        ("Smartphone {}", 3, 800, 500),
        ("Tablet {}", 3, 600, 400),
        ("Wireless Mouse {}", 4, 50, 20),
        ("Keyboard {}", 4, 80, 35),
        ("Monitor {}", 4, 400, 250),
        ("Headphones {}", 4, 150, 60),
        ("Office Suite {}", 5, 200, 50),
        ("Antivirus {}", 5, 80, 20),
        ("Standing Desk {}", 7, 600, 350),
        ("Office Chair {}", 7, 400, 200),
        ("Filing Cabinet {}", 7, 250, 150),
        ("Paper Pack {}", 8, 30, 15),
        ("Pen Set {}", 8, 20, 8),
        ("Router {}", 12, 200, 100),
        ("Switch {}", 12, 300, 180),
        ("External SSD {}", 13, 150, 80),
        ("NAS System {}", 13, 800, 500),
        ("Security Camera {}", 14, 200, 100),
    ]
    
    products = []
    for i in range(1, 101):
        template = product_templates[(i - 1) % len(product_templates)]
        name = template[0].format(f"v{(i // 20) + 1}")
        category_id = template[1]
        price = template[2] + random.randint(-50, 100)
        cost = template[3] + random.randint(-20, 40)
        sku = f"SKU-{i:04d}"
        created = random_date(2020, 2024)
        is_active = 1 if random.random() > 0.1 else 0
        
        products.append((i, name, sku, category_id, price, cost, f"Description for {name}", created, is_active))
    
    cursor.executemany(
        "INSERT INTO products VALUES (?,?,?,?,?,?,?,?,?)", 
        products
    )
    
    # Insert Locations (10)
    locations = [
        (1, "Main Warehouse", "warehouse", "100 Industrial Pkwy", "Chicago", "IL", 50000),
        (2, "East Coast DC", "distribution_center", "200 Logistics Way", "Newark", "NJ", 30000),
        (3, "West Coast DC", "distribution_center", "300 Pacific Blvd", "Los Angeles", "CA", 35000),
        (4, "Downtown Store", "store", "400 Main St", "New York", "NY", 5000),
        (5, "Mall Location", "store", "500 Shopping Center", "Chicago", "IL", 3000),
        (6, "Tech Campus", "office", "600 Innovation Dr", "San Jose", "CA", 2000),
        (7, "Regional Office", "office", "700 Business Park", "Austin", "TX", 1500),
        (8, "South Warehouse", "warehouse", "800 Distribution Rd", "Atlanta", "GA", 40000),
        (9, "Midwest Store", "store", "900 Retail Row", "Denver", "CO", 4000),
        (10, "Flagship Store", "store", "1000 Premium Ave", "Miami", "FL", 6000),
    ]
    cursor.executemany("INSERT INTO locations VALUES (?,?,?,?,?,?,?)", locations)
    
    # Insert Inventory (300+ entries)
    inventory = []
    inv_id = 1
    for product_id in range(1, 101):
        # Each product in 2-5 locations
        num_locations = random.randint(2, 5)
        locs = random.sample(range(1, 11), num_locations)
        
        for loc_id in locs:
            quantity = random.randint(0, 500)
            reorder = random.randint(10, 50)
            last_restock = random_date(2024, 2024)
            
            inventory.append((inv_id, product_id, loc_id, quantity, reorder, last_restock))
            inv_id += 1
    
    cursor.executemany(
        "INSERT INTO inventory VALUES (?,?,?,?,?,?)", 
        inventory
    )
    
    # Insert Customers (200)
    customers = []
    for i in range(1, 201):
        company = random_company()
        contact = random_name()
        email = random_email(contact)
        phone = random_phone()
        address = f"{random.randint(100, 9999)} {random.choice(['Main', 'Oak', 'Elm', 'First', 'Second'])} St"
        city = random_city()
        state = random_state()
        credit_limit = random.choice([5000, 10000, 25000, 50000, 100000])
        created = random_date(2018, 2024)
        is_active = 1 if random.random() > 0.05 else 0
        
        customers.append((i, company, contact, email, phone, address, city, state, credit_limit, created, is_active))
    
    cursor.executemany(
        "INSERT INTO customers VALUES (?,?,?,?,?,?,?,?,?,?,?)", 
        customers
    )
    
    # Insert Sales (500+)
    sales = []
    payment_methods = ['credit_card', 'bank_transfer', 'cash', 'check']
    statuses = ['pending', 'completed', 'completed', 'completed', 'refunded', 'cancelled']  # Weighted toward completed
    
    for i in range(1, 501):
        customer_id = random.randint(1, 200)
        employee_id = random.randint(1, 60)
        product_id = random.randint(1, 100)
        quantity = random.randint(1, 20)
        unit_price = random.uniform(20, 2000)
        discount = random.choice([0, 0, 0, 5, 10, 15, 20])
        sale_date = random_date(2023, 2024)
        payment = random.choice(payment_methods)
        status = random.choice(statuses)
        
        sales.append((i, customer_id, employee_id, product_id, quantity, round(unit_price, 2), discount, sale_date, payment, status))
    
    cursor.executemany(
        "INSERT INTO sales VALUES (?,?,?,?,?,?,?,?,?,?)", 
        sales
    )
    
    conn.commit()
    
    # Print summary
    print("\nDatabase created successfully!")
    print("-" * 40)
    
    tables = ["departments", "employees", "projects", "project_assignments", 
              "categories", "products", "locations", "inventory", "customers", "sales"]
    
    for table in tables:
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        count = cursor.fetchone()[0]
        print(f"{table:25s}: {count:5d} rows")
    
    conn.close()
    print("-" * 40)
    print(f"Database saved to: {db_path}")


if __name__ == "__main__":
    # Create database in data/custom directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    db_path = os.path.join(project_root, "data", "custom", "company_analytics.db")
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    create_database(db_path)
