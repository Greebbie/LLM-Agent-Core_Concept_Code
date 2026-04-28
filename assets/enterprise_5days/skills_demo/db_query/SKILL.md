---
name: db-query
description: Helps the user query the company's order / inventory database via the enterprise MCP server. Translates natural language questions into the appropriate MCP tool call. Use when the user asks about orders, inventory, stock, or wants to send notifications.
allowed-tools: [mcp__enterprise-demo__query_order, mcp__enterprise-demo__check_inventory, mcp__enterprise-demo__send_notification]
version: "0.1"
---

# DB Query Skill (via MCP)

A skill that demonstrates **Skills × MCP integration**. The skill body teaches
Claude how to translate natural language to the right MCP tool call; the actual
data access goes through the `enterprise-demo` MCP server.

## When to use

- "Look up order ORD-xxx"
- "How much stock for SKU-xxx?"
- "Notify alice that her order shipped"

Do **NOT** use for:
- HR / policy questions (use `hr-policy` skill)
- Code review (use `code-review` skill)

## Workflow

1. **Identify intent**: order lookup / stock check / notification
2. **Pick MCP tool**:
   - `query_order(order_id)` for order lookup
   - `check_inventory(sku)` for stock
   - `send_notification(user_id, message)` for notify
3. **Format args**: extract `order_id` / `sku` / `user_id+message` from the user's message
4. **Call the MCP tool** (allowed-tools list above limits to these 3)
5. **Format response**: parse JSON returned by tool, present human-readable answer

## Examples

| User message | Tool call |
|---|---|
| "查 ORD-001 的状态" | `query_order(order_id="ORD-001")` |
| "SKU-A100 库存还有多少" | `check_inventory(sku="SKU-A100")` |
| "提醒 alice 她的订单要到了" | `send_notification(user_id="alice", message="订单即将送达")` |

## Error handling

- Tool returns `{"error": ...}` → return user-friendly explanation, don't fabricate data
- Tool times out → ask user to retry, log incident

## See also

- `reference/sql_examples.md` — SQL-style query patterns when the simple tools don't fit (loaded on demand)
