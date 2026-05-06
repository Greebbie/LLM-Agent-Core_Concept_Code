---
name: db-query
description: Route order status, inventory, SKU, and customer notification requests to the enterprise-demo MCP server. Use for ORD-* order lookups, SKU stock checks, and notification sending.
allowed-tools: [mcp__enterprise-demo__query_order, mcp__enterprise-demo__check_inventory, mcp__enterprise-demo__send_notification]
version: "0.1"
---

# DB Query Skill

This skill demonstrates Skills + MCP integration. The skill body teaches the model how to translate natural language into the right tool call; the actual data access happens through the `enterprise-demo` MCP server.

## When To Use

- "Look up order ORD-001"
- "How much stock is left for SKU-A100?"
- "Notify alice that her order shipped"

Do not use this skill for HR policy, product documentation, API docs, or code review. Those belong to other skills or the capstone assistant.

## Workflow

1. Identify intent: order lookup, inventory check, or notification.
2. Extract identifiers:
   - `ORD-*` for `query_order(order_id)`.
   - `SKU-*` for `check_inventory(sku)`.
   - `user_id` and message for `send_notification(user_id, message)`.
3. Call the corresponding MCP tool.
4. Parse the JSON returned by the tool.
5. Present a short human-readable answer. If the tool returns `{"error": ...}`, explain the error and do not fabricate data.

## Examples

| User message | Tool call |
|---|---|
| "查 ORD-001 的状态" | `query_order(order_id="ORD-001")` |
| "SKU-A100 库存还有多少" | `check_inventory(sku="SKU-A100")` |
| "提醒 alice 订单快到了" | `send_notification(user_id="alice", message="订单即将送达")` |

## See Also

`reference/sql_examples.md` contains SQL-style patterns for a future server that exposes a safe `run_sql` tool.
