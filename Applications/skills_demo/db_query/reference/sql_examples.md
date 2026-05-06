# SQL Query Patterns

Use these patterns only if a future MCP server exposes a safe parameterized SQL tool such as `run_sql`. The current demo server intentionally exposes narrow tools instead of raw SQL.

## 1. Orders By Time Range

```sql
SELECT order_id, customer, total
FROM orders
WHERE created_at BETWEEN :start_date AND :end_date
  AND status = :status
ORDER BY created_at DESC
LIMIT 100;
```

## 2. Low Inventory Alert

```sql
SELECT sku, name, quantity
FROM inventory
WHERE quantity < :threshold
ORDER BY quantity ASC;
```

## 3. Customer Spend Summary

```sql
SELECT customer,
       COUNT(*) AS n_orders,
       SUM(total) AS total_spent
FROM orders
WHERE status IN ('shipped', 'delivered')
GROUP BY customer
HAVING SUM(total) > :min_total
ORDER BY total_spent DESC;
```

## Notes

- Always use parameterized queries. Do not concatenate user input into SQL strings.
- Add `LIMIT` for exploratory queries.
- Check numeric types before comparing or sorting.
