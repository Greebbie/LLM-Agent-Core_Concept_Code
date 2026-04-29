# Team Code Review Checklist

Apply each section to the changes under review. Loaded **on demand** by Claude — no need to read every time.

## 1. 命名（Naming）

- 函数 / 变量名表达意图，不需读实现就懂功能？
- 缩写控制在团队公认范围（如 `db`, `cfg`, 不用 `usr`, `cnt`）？
- 布尔变量用 `is_`, `has_`, `should_` 前缀？

## 2. 错误处理（Error Handling）

- 所有 IO / 网络 / 子进程都有 try/except？
- 捕获时记录上下文（log + 重新抛出 / 用户友好消息）？
- 没有空 `except: pass` 隐藏问题？
- 资源清理用 context manager（`with`）？

## 3. 测试（Tests）

- 新代码至少有 1 个 happy path 测试？
- 边界 case（空输入 / 大输入 / 异常）覆盖了？
- 测试断言**有意义**（不是 `assert True`）？
- Mock 没过度使用（容易和真实行为脱节）？

## 4. 文档（Docs）

- 公共 API 有 docstring（参数 + 返回 + 异常）？
- 复杂逻辑有内联注释**讲为什么**（不是讲做什么）？
- README / CHANGELOG 跟着代码同步更新？

## 5. 安全（Security）

- 输入验证在边界处做了？
- 没有 SQL 拼接 / shell 拼接？
- 密钥 / 密码不在代码里（用环境变量 / vault）？
- 第三方依赖检查过 CVE？

## 6. 性能（Performance）

- 循环里没重复 IO（提到外面 / 用 batch）？
- 大数据用 generator 而非 list comprehension？
- 数据库查询有 index / 没 N+1？
