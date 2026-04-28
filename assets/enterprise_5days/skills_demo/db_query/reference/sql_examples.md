# SQL Query Patterns（参考）

当 `query_order` / `check_inventory` 等简单 tool 不够时，可以参考下面的模式手写
SQL（如果 MCP server 暴露了 `run_sql` 这种 tool）。

> **本文档由 progressive disclosure 按需加载** — Claude 只在 query 涉及『复杂条件 / 聚合 / 跨表』时才读这个文件。

## 1. 按时间范围查订单

```sql
SELECT order_id, customer, total
FROM orders
WHERE created_at BETWEEN '2026-04-01' AND '2026-04-30'
  AND status = 'shipped'
ORDER BY created_at DESC
LIMIT 100;
```

## 2. 库存预警（quantity < 阈值）

```sql
SELECT sku, name, quantity
FROM inventory
WHERE quantity < 10
ORDER BY quantity ASC;
```

## 3. 客户订单聚合

```sql
SELECT customer,
       COUNT(*) AS n_orders,
       SUM(total) AS total_spent
FROM orders
WHERE status IN ('shipped', 'delivered')
GROUP BY customer
HAVING total_spent > 1000
ORDER BY total_spent DESC;
```

## 注意事项

- 永远使用参数化查询，不要字符串拼接
- 大数据查询加 `LIMIT`，避免一次返回数百万行
- 数字比较时注意类型（VARCHAR 排序 ≠ INT 排序）
