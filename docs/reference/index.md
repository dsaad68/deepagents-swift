# API reference

DeepAgents ships full Swift documentation comments on every public type. The tables below index the main public API surface by module - use them as a map to find the type you need, then consult the relevant concept or adapter page for context on how they fit together.

Related concept pages: [Architecture](../concepts/architecture.md), [The agent loop](../concepts/agent-loop.md), [Middleware](../concepts/middleware.md), [Tools & policy](../concepts/tools.md), [MCP client](../concepts/mcp.md), [Memory & checkpointing](../concepts/memory.md), [Subagents](../concepts/subagents.md), [Human in the loop](../concepts/human-in-the-loop.md).

Related adapter pages: [Adapters overview](../adapters/index.md), [MLX](../adapters/mlx.md), [OpenAI & Azure](../adapters/openai.md), [Anthropic & Bedrock](../adapters/anthropic.md), [macOS tools](../adapters/mac-tools.md).

---

## Core (`DeepAgents`)

### Agent factories and runtime

| Type | Purpose |
|---|---|
| `createAgent` | Factory: minimal agent; bring your own tools and middleware |
| `createDeepAgent` | Factory: batteries-included deep agent with planning, filesystem, and subagent pillars |
| `ReactAgent` | The ReAct loop; call `run(...)` to start, `compact(...)` to compress context |
| `AgentEvent` | Streamed progress events: token, toolStarted, toolCompleted, completed, failed |
| `AgentState` | Mutable run state threaded through middleware lifecycle hooks |
| `AgentStateUpdate` | Incremental state mutations applied by tools |

### Messages and content

| Type | Purpose |
|---|---|
| `AgentMessage` | A single conversation turn; carries role, content blocks, tool calls |
| `AgentContentBlock` | Typed content: `.text`, `.reasoning`, `.image` |
| `AgentImage` | Image payload: URL, base64, MIME type, or file ID |
| `AgentToolCall` | A tool invocation requested by the model: name + arguments |
| `AgentJSON` | Typed JSON value enum used for tool arguments and results |
| `AgentLogContext` | Context metadata attached to message log entries |
| `AgentMessageLog` | Protocol: receives every appended message; for audit and debug |
| `JSONLMessageLog` | Concrete log: one JSON line per message to a timestamped file |

### Tools

| Type | Purpose |
|---|---|
| `AgentTool` | Protocol: name, description, parameters, execute |
| `ToolParameter` | Describes one tool parameter; built with `.required` / `.optional` |
| `ToolParameterType` | Typed JSON schema: string, bool, int, double, array, object, data |
| `ToolContext` | Runtime context passed to `execute`; access to agent state |
| `ToolOutput` | Return type of `execute`: result text + optional state update |
| `AgentToolPolicy` | Declarative policy: disable middleware/tools, set approval modes, configure sandbox |
| `ToolApprovalMode` | `.approve` / `.ask` / `.deny` |
| `SandboxMode` | `.off` / `.failover` / `.containerOnly` |

### Middleware

| Type | Purpose |
|---|---|
| `AgentMiddleware` | Protocol: name, tools, six lifecycle hooks |
| `ModelRequest` | Input to `wrapModelCall`: messages, system prompt, tools |
| `ModelResponse` | Output of `wrapModelCall`: assistant message |
| `ToolCallRequest` | Input to `wrapToolCall`: tool name, arguments |
| `MiddlewareCatalog` | Registry of all built-in capability middleware descriptors |
| `MiddlewareDescriptor` | Metadata for one middleware entry in the catalog |

### Structural middleware (built-in pillars)

| Type | Purpose |
|---|---|
| `TodoListMiddleware` | Planning discipline; contributes `write_todos` |
| `TodoItem` | A single todo entry in the agent's plan |
| `WriteTodosTool` | Tool that updates the agent's todo list |
| `FilesystemMiddleware` | File I/O; pluggable backend |
| `FilesystemBackend` | Protocol: read/write/list/delete |
| `StateBackend` | In-memory filesystem backend (default for `createDeepAgent`) |
| `LocalFilesystemBackend` | Real filesystem backend scoped to a root path |
| `SubAgentMiddleware` | Delegation via the `task` tool |
| `SubAgent` | Configuration for one subagent (model + tools + prompt) |
| `TaskTool` | The `task` tool that dispatches to a subagent |
| `SummarizationMiddleware` | Automatic context compaction at ~85% context window |
| `SummarizationConfig` | Tuning for the summarization trigger threshold and prompt |
| `HumanInTheLoopMiddleware` | Approval gating via `wrapToolCall` |
| `ToolApprovalRequest` | Describes a pending tool call passed to the approval handler |
| `InterruptOnConfig` | Per-tool interrupt configuration for `createDeepAgent` |
| `ToolApprovalHandler` | Type alias for the approval closure |
| `ToolDecisionType` | `.approve` / `.deny(reason:)` / `.ask` |
| `AskUserMiddleware` | Lets the model pause and prompt the human mid-run |
| `AskUserTool` | The `ask_user` tool |
| `AskUserHandler` | Type alias for the ask-user response closure |

### Capability middleware (catalog entries)

| Type | Tools contributed |
|---|---|
| `WebToolsMiddleware` | `fetch`, `curl` |
| `SearchToolsMiddleware` | `grep`, `glob`, `tree` |
| `TextToolsMiddleware` | `head`, `tail`, `diff` |
| `GitToolsMiddleware` | `git_status`, `git_diff`, `git_log`, `git_show`, `git_blame` |
| `ShellToolsMiddleware` | `shell` (gated by `ShellGuard`) |
| `ShellGuard` | Policy object controlling shell access |

### ChatModel abstraction

| Type | Purpose |
|---|---|
| `ChatModel` | Protocol: supportsVision, modelID, contextWindowTokens, makeSession |
| `ModelTurnSession` | Protocol: `nextTurn` receives full conversation, returns one assistant message |
| `MessageCodec` | Base type for backend-specific message encoding/decoding |

### Memory

| Type | Purpose |
|---|---|
| `AgentCheckpointer` | Protocol: load/save conversation by thread ID |
| `InMemoryCheckpointer` | In-process checkpointer; resets on process restart |

### MCP client

| Type | Purpose |
|---|---|
| `MultiServerMCPClient` | Connects to N stdio or HTTP MCP servers concurrently |
| `MCPMiddleware` | Contributes MCP tools to an agent |
| `MCPServerConfig` | Config for one server: kind, command/URL, auth, approvalMode |
| `MCPServerStatus` | Connection state for one server after `load()` |
| `MCPTool` | `AgentTool` wrapper around a tool discovered from an MCP server |

### Networking

| Type | Purpose |
|---|---|
| `HTTPClient` | Shared HTTP utility used internally by adapters |

---

## MLX adapter (`DeepAgentsMLX`)

| Type | Purpose |
|---|---|
| `MlxModel` | Model descriptor: ID, display name, kind, size, catalog entry |
| `MlxChatModel` | `ChatModel` backed by a loaded MLX `ModelContainer` |
| `RebuildTurnSession` | `ModelTurnSession` implementation; rebuilds prompt each round |
| `MlxModelLoader` | Loads and caches models from the local Hugging Face cache |
| `LFM2MessageCodec` | Encodes `AgentMessage` to LFM2 chat template format |
| `LFM2Decoder` | Parses LFM2 model output into `AgentMessage` |
| `LFM2ToolCalls` | Parses `<\|tool_call_start\|>...<\|tool_call_end\|>` spans |
| `LFM2ToolCallStream` | Streaming version of the tool-call parser |
| `LFM2ChatTemplate` | Renders the full conversation as an LFM2 prompt string |

---

## OpenAI adapter (`DeepAgentsOpenAI`)

| Type | Purpose |
|---|---|
| `OpenAIChatModel` | `ChatModel` for OpenAI, Azure, OpenRouter, and compatible endpoints |
| `OpenAIAuthStyle` | `.bearer` (standard) or `.apiKey` (legacy/Azure) |
| `OpenAIEndpointStyle` | `.standard` or `.azure(deployment:apiVersion:)` |
| `OpenAITurnSession` | `ModelTurnSession` for OpenAI chat completions |
| `OpenAIMessageCodec` | Encodes/decodes `AgentMessage` to the OpenAI messages format |
| `OpenAIGenerateParameters` | Sampling parameters: temperature, top-p, max tokens, etc. |
| `OpenAIModelError` | Typed error from the OpenAI API |

---

## Anthropic adapter (`DeepAgentsAnthropic`)

| Type | Purpose |
|---|---|
| `AnthropicChatModel` | `ChatModel` for the Anthropic Messages API |
| `BedrockChatModel` | `ChatModel` for AWS Bedrock (Anthropic models via SigV4) |
| `BedrockCredentials` | AWS credentials; `fromEnvironment()` reads standard env vars |
| `AnthropicTurnSession` | `ModelTurnSession` for Anthropic Messages |
| `BedrockTurnSession` | `ModelTurnSession` for Bedrock invoke-with-response-stream |
| `AnthropicMessageCodec` | Encodes/decodes `AgentMessage` to the Anthropic messages format |
| `AnthropicDecoder` | Parses Anthropic SSE stream into `AgentMessage` |
| `AnthropicGenerateParameters` | Sampling parameters for Anthropic (temperature, top-k, max tokens) |
| `AnthropicModelError` | Typed error from the Anthropic API |
| `BedrockModelError` | Typed error from AWS Bedrock |
| `SigV4Signer` | AWS SigV4 request signer used by `BedrockChatModel` |

---

## macOS tools (`DeepAgentsMacTools`)

| Type | Purpose |
|---|---|
| `MacToolsMiddleware` | Contributes macOS system tools: mdfind, open, open_app, download, say, notify |
| `ScreenshotMiddleware` | Contributes take_screenshot and take_window_screenshots via ScreenCaptureKit |
| `ClipboardMiddleware` | Contributes read_clipboard and write_clipboard |
| `AppleNotesMiddleware` | Contributes list/read/create/update note tools via osascript |
| `ScreenshotCapture` | Low-level capture API used by `ScreenshotMiddleware` |
