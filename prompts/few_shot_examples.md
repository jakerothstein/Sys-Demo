# Few-Shot Examples for SQL Generation

These examples demonstrate how to translate natural language to SQL for the schema.

## Schema
```sql
users(id INTEGER PRIMARY KEY, name TEXT, email TEXT)
orders(id INTEGER PRIMARY KEY, user_id INTEGER, amount REAL, status TEXT)
```

## Examples

### Example 1: Count Query
**Input**: "How many users do we have?"
**Output**:
```sql
SELECT COUNT(*) FROM users;
```

### Example 2: Aggregation with Filter
**Input**: "What is the total revenue from completed orders?"
**Output**:
```sql
SELECT SUM(amount) FROM orders WHERE status = 'completed';
```

### Example 3: Join Query
**Input**: "Show me all orders for the user named Alice."
**Output**:
```sql
SELECT o.* FROM orders o
JOIN users u ON o.user_id = u.id
WHERE u.name = 'Alice';
```

### Example 4: Top N Query
**Input**: "What are the top 3 orders by amount?"
**Output**:
```sql
SELECT * FROM orders ORDER BY amount DESC LIMIT 3;
```

### Example 5: Existence Check
**Input**: "Are there any users with a gmail address?"
**Output**:
```sql
SELECT * FROM users WHERE email LIKE '%gmail.com';
```
