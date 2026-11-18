# acp2api

将任意遵循 **ACP (Agent Client Protocol)** 标准、通过 `stdio` 启动的 Agent 服务转换为标准的 **OpenAI API** 与 **Anthropic Messages API** HTTP 网关。

采用纯 Go 标准库实现（零第三方运行时依赖，单文件静态编译），内置 **ACP 子进程池 + 后台空闲 Session 预热池 + 多轮对话增量复用缓存**。

---

## 核心特性

- **完整 `stdio` ACP 协议支持**
  - **连接与认证**：自动完成 `initialize` 能力协商、可选 `authenticate` 认证握手、JSON-RPC 2.0 批量帧（Batch）与 `$/cancelRequest` 反向请求取消。
  - **全会话生命周期**：支持 `session/new`、`session/load`（历史回放）、`session/resume`、`session/fork`（并发会话原生分叉）、`session/list`、`session/close`（淘汰会话主动释放 Agent 内存）、`session/prompt`、`session/cancel`、`session/set_config_option`、`session/set_model`、`session/set_mode`。
  - **多模态与内嵌资源**：支持 `text`、`image`、`audio`（音频输入）、`resource`（文档/PDF 内嵌上下文）与 `resource_link`，并根据 Agent 声明的 `promptCapabilities` 自动优雅降级；MCP 配置完整支持 `stdio`、`http`、`sse` 三种传输协议。
  - **实时状态同步与宿主回调**：动态同步 `config_option_update`、`current_mode_update`、`usage_update` 等通知；支持 `allow_all` / `allow_read_only` / `deny_all` 权限审批回调，以及带目录沙箱保护的 `fs/*` 与进程组级 `terminal/*` 终端管理。
- **Thinking Level 多级探测与非标降级引擎**
  - 兼容 OpenAI `reasoning_effort`、`/v1/responses` `reasoning.effort`、模型后缀（如 `model:high`、`model(xhigh)`）以及 Anthropic `thinking`（`enabled`/`adaptive`/`disabled` + `budget_tokens`）与 `output_config.effort`。
  - 针对不按标准 `category: "thought_level"` 返回配置项的 ACP 服务，内置 **6 层渐进式解析与输出兜底**：标准 `thought_level` -> 启发式 `configOptions`（模糊 ID/布尔开关/数值预算/档位枚举）-> 模型变体（`<model>-high`/`-thinking`）与 `modes` -> `override_config_id` 乐观调用 -> `_meta` 扩展透传 / `<thinking_guidance>` 提示词引导 / 静默降级 -> 输出侧自动提取内联 `<think>...</think>` 为标准思考流。
- **空闲 Session 预热池 + 智能多轮对话复用**
  - **预热池（Zero-Latency Warmup）**：启动时及每次消费后自动在后台预建指定数量（`idle_pool_size`）的空闲 Session，新请求到达时直接命中已就绪 Session，消除 `session/new` 冷启动耗时。
  - **智能多轮增量复用**：通过规范化消息前缀 SHA-256 指纹自动识别多轮对话与工具回调轮次，命中原 ACP Session 时**仅发送新增消息增量**；当发生对话分支（重新生成/修改历史消息）或并发分叉时自动回退到空闲预热池并重放上下文，零串话风险。
  - **进程池与崩溃自愈**：支持单/多常驻 ACP 子进程多路复用并发 Session，子进程意外退出时自动清理失效会话、指数退避拉起新进程并重新注满预热池。
- **双协议完整兼容**
  - **OpenAI API**：`GET /v1/models`、`POST /v1/chat/completions`（流式 SSE / 非流式、`reasoning_content` 思考流、多模态图片、`tools` Function Calling、`stream_options.include_usage`）、`POST /v1/responses`。
  - **Anthropic API**：`POST /v1/messages`（流式 SSE / 非流式、`thinking_delta` + `signature_delta` 思考块、多模态图片、`tool_use` + `input_json_delta` 工具调用）、`POST /v1/messages/count_tokens`。
  - **运行态观测**：`GET /healthz`、`GET /v1/pool/stats`。

---

## 编译与快速启动

### 1. 编译二进制

```bash
go build -o acp2api ./cmd/acp2api
```

### 2. 命令行一键启动

可以直接在 `--` 之后传入任意 stdio ACP 启动命令与参数：

```bash
# 示例 1：对接 Gemini CLI ACP
./acp2api -listen :8080 -idle-sessions 3 -- gemini --experimental-acp

# 示例 2：对接 Claude Code ACP 并开启 API Key 鉴权
./acp2api -listen :8080 -api-key "sk-my-secret" -idle-sessions 3 -- claude-code-acp

# 示例 3：对接 Codex / OpenCode ACP，启动 2 个工作进程、4 个预热空闲 Session
./acp2api -listen :8080 -processes 2 -idle-sessions 4 -- opencode acp
```

### 3. 使用配置文件启动

支持标准 JSON 与带注释的 JSONC 配置文件（参见 `config.example.json`）：

```bash
cp config.example.json config.json
./acp2api -config config.json
```

---

## 接口调用示例

### OpenAI Chat Completions (`/v1/chat/completions`)

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-my-secret" \
  -d '{
    "model": "acp-default",
    "stream": true,
    "messages": [
      {"role": "user", "content": "请用一句话介绍 Agent Client Protocol"}
    ]
  }'
```

### Anthropic Messages (`/v1/messages`)

```bash
curl http://localhost:8080/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: sk-my-secret" \
  -H "anthropic-version: 2023-06-01" \
  -d '{
    "model": "acp-default",
    "max_tokens": 1024,
    "stream": true,
    "messages": [
      {"role": "user", "content": "用 Go 写一个并发安全的环形缓冲区"}
    ]
  }'
```

### 查看空闲池与复用统计 (`/v1/pool/stats`)

```bash
curl http://localhost:8080/v1/pool/stats -H "Authorization: Bearer sk-my-secret"
```

返回示例：

```json
{
  "total_workers": 1,
  "healthy_workers": 1,
  "target_idle_sessions": 3,
  "ready_idle_sessions": 3,
  "warming_sessions": 0,
  "cached_conversations": 2,
  "active_turns": 0,
  "total_requests": 8,
  "idle_pool_hits": 3,
  "conversation_reuse_hits": 5,
  "cold_session_creations": 0,
  "worker_restarts": 0
}
```

非流式响应还会附带调试响应头：
- `X-ACP-Session-ID`: 本次请求实际使用的底层 ACP `sessionId`
- `X-ACP-Session-Reused`: `true`（命中多轮增量复用）或 `false`（取自预热空闲池）

---

## 运行测试

```bash
go test -race -v ./...
```
