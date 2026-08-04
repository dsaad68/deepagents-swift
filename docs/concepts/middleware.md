# Middleware

Middleware is DeepAgents' primary extension point. Every middleware observes and can rewrite the full agent lifecycle - messages, system prompts, tool lists, and individual tool calls. The same mechanism powers DeepAgents' built-in structural pillars (planning, filesystem, subagents, summarization, human-in-the-loop) and the swappable capability toolsets (web, git, shell, macOS, ...).

---

## `AgentMiddleware` protocol

```swift
public protocol AgentMiddleware: Sendable {
    var name: String { get }
    var tools: [any AgentTool] { get }

    func beforeAgent(_ state: inout AgentState) async
    func beforeModel(_ state: inout AgentState) async
    func afterModel(_ state: inout AgentState) async
    func afterAgent(_ state: inout AgentState) async

    func wrapModelCall(
        _ request: ModelRequest,
        _ handler: (ModelRequest) async throws -> ModelResponse
    ) async throws -> ModelResponse

    func wrapToolCall(
        _ request: ToolCallRequest,
        _ handler: (ToolCallRequest) async throws -> AgentMessage
    ) async throws -> AgentMessage
}
```

### Hook execution order

```
run(...) called
│
├── beforeAgent  (all middleware, once per run, in registration order)
│
│   ┌── [Round N] ───────────────────────────────────────────────────┐
│   │  beforeModel  (all middleware, every round, registration order) │
│   │                                                                  │
│   │  wrapModelCall  ← NESTED, first-registered is OUTERMOST        │
│   │    └── session.nextTurn(...)                                     │
│   │  /wrapModelCall                                                  │
│   │                                                                  │
│   │  afterModel   (all middleware, every round, registration order) │
│   │                                                                  │
│   │  For each tool call:                                             │
│   │    wrapToolCall  ← NESTED, first-registered is OUTERMOST        │
│   │      └── tool.execute(...)                                       │
│   │    /wrapToolCall                                                 │
│   └── [End round] ────────────────────────────────────────────────┘
│
└── afterAgent   (all middleware, once per run, reverse order)
```

!!! note "Nesting vs. sequencing"
    `wrapModelCall` and `wrapToolCall` are **nested** decorators - each middleware wraps the next, exactly like HTTP middleware stacks. The middleware registered first is the outermost layer: it can inspect and modify the request before any inner middleware sees it, and it sees the final response after all inner middleware have processed it. By contrast, `beforeModel`/`afterModel` are sequential callbacks that each mutate shared `AgentState`.

### What each hook can do

| Hook | When | What you can do |
|---|---|---|
| `beforeAgent` | Once at run start | Inject system prompt additions, initialise state, log run start |
| `beforeModel` | Before each model call | Rewrite `messages`, edit `systemPrompt`, add/remove `tools` |
| `afterModel` | After each model call | Inspect the just-produced assistant message, update state |
| `afterAgent` | Once at run end | Flush logs, release resources, report metrics |
| `wrapModelCall` | Around the model call | Intercept/retry the model call; rewrite request (messages, tools, prompt) |
| `wrapToolCall` | Around each tool call | Intercept/approve/deny/retry a tool call; rewrite arguments or result |

`AgentState` holds the live conversation thread, system prompt, and tool list for the current round. Mutations in `beforeModel` are visible to the model call that follows.

### Contributing tools

The `tools` property lets middleware own the tools it provides. Tools contributed this way are merged with the explicit `tools:` array at factory time, then filtered through `disabledToolNames`. Middleware should declare all the tools it manages via this property rather than injecting them through `beforeModel` - this ensures correct deduplication and policy enforcement.

---

## Structural pillars

Structural middleware implements the core agent architecture. `createDeepAgent` wires these in automatically; `createAgent` leaves them out unless you add them yourself.

### `TodoListMiddleware`

Adds planning discipline to the agent. Appends writing guidance to the system prompt and contributes the `write_todos` tool, which lets the model maintain a structured task list that persists across rounds within a run.

### `FilesystemMiddleware`

Provides file I/O through a pluggable `FilesystemBackend`. Two backends ship:

- `StateBackend` - in-memory; good for tests and isolated runs
- `LocalFilesystemBackend` - reads and writes real files on disk

Tools: `ls`, `read_file`, `write_file`, `edit_file`, `mkdir`. Pass a backend via `createDeepAgent`'s `backend:` parameter; it defaults to `StateBackend()` when `includeFilesystem: true`.

### `SubAgentMiddleware`

Enables task delegation. Contributes the `task` tool; when the model calls `task`, the middleware routes execution to one of the registered `SubAgent` instances (or to a general-purpose sub-agent when `includeGeneralPurpose: true`). See [Subagents](subagents.md).

### `SummarizationMiddleware`

Hooks `beforeModel` and compacts the conversation when the context window reaches ~85% of capacity. Summarised segments are replaced with a single synthetic message whose `source` field is set to identify it as compaction-synthesised. Configured via `SummarizationConfig` (pass `nil` to disable). See [Summarization](summarization.md).

### `HumanInTheLoopMiddleware`

Hooks `wrapToolCall` and calls your `ToolApprovalHandler` before every tool execution. The handler can approve, deny, or ask for user confirmation. Required for tools with side effects (file writes, shell commands, macOS automation). See [Human in the loop](human-in-the-loop.md).

### `AskUserMiddleware`

Contributes the `ask_user` tool. When the model calls `ask_user`, execution suspends until your `AskUserHandler` returns a string. This lets the model request clarification mid-run without ending the run.

---

## Capability catalog

Capability middleware provides toolsets that map cleanly to a single concern. All of these live in `MiddlewareCatalog.all` and can be enabled by ID. `createDeepAgent` with `includeGeneralPurpose: true` adds web, search, text, git, and shell automatically.

| Middleware ID | Type | Tools contributed |
|---|---|---|
| `web` | `WebToolsMiddleware` | `fetch`, `curl` |
| `search` | `SearchToolsMiddleware` | `grep`, `glob`, `tree` |
| `text` | `TextToolsMiddleware` | `head`, `tail`, `diff` |
| `git` | `GitToolsMiddleware` | `git_status`, `git_diff`, `git_log`, `git_show`, `git_blame` |
| `shell` | `ShellToolsMiddleware` | `shell` (gated by `ShellGuard`) |
| `macos` | `MacToolsMiddleware` | `mdfind`, `open`, `open_app`, `download`, `say`, `notify` |
| `filesystem` | `FilesystemMiddleware` | `ls`, `read_file`, `write_file`, `edit_file`, `mkdir` |
| `clipboard` | `ClipboardMiddleware` | `read_clipboard`, `write_clipboard` |
| `screenshot` | `ScreenshotMiddleware` | `take_screenshot`, `take_window_screenshots` |
| `apple_notes` | `AppleNotesMiddleware` | `list_notes`, `read_note`, `create_note`, `update_note` |
| `container` | `ContainerShellMiddleware` | `container_shell` (sandbox mode) |

!!! warning "macOS adapter required"
    `screenshot`, `clipboard`, `apple_notes`, `macos`, and `container` are provided by the `DeepAgentsMacTools` product and require macOS entitlements. Import `DeepAgentsMacTools` separately.

---

## `createDeepAgent` composition order

When you call `createDeepAgent`, middleware is assembled in this order before being handed to `ReactAgent`:

1. `TodoListMiddleware`
2. `FilesystemMiddleware` (when `includeFilesystem: true`)
3. `SubAgentMiddleware` (when `subagents` is non-empty or `includeGeneralPurpose: true`)
4. Your `middleware` array (in the order you provide)
5. `AskUserMiddleware` (when `askUserHandler != nil`)
6. `HumanInTheLoopMiddleware` (when `approvalHandler != nil`)
7. `SummarizationMiddleware` (when `summarization != nil`)

Because `wrapToolCall` nests with the first-registered middleware outermost, `HumanInTheLoopMiddleware` always wraps the outermost layer of tool dispatch - meaning approval fires before any inner middleware can execute the call. Summarization hooks `beforeModel`, so it runs last in that phase and can compact the history produced by all prior middleware.

---

## Disabling middleware and tools

### Via `disabledToolNames`

Both factories accept `disabledToolNames: Set<String>`. Any tool whose `name` appears in this set is removed from the merged tool list at factory time. The model never sees the tool in its schema - it cannot call something that was never offered.

### Via `AgentToolPolicy`

```swift
public struct AgentToolPolicy: Codable, Sendable {
    public var disabledMiddleware: Set<String>  // middleware IDs (catalog names)
    public var disabledTools: Set<String>       // individual tool names
    public var approvals: [String: ToolApprovalMode]
    public var sandbox: SandboxMode
    public var sandboxImage: String?

    public func expand(
        catalog: [MiddlewareDescriptor] = MiddlewareCatalog.all,
        extraDefaults: [String: ToolApprovalMode] = [:]
    ) -> Expansion
}
```

`AgentToolPolicy` is a serialisable value (Codable) useful for per-user or per-session configuration - for example, persisting the user's approval preferences between sessions in Ripple. Call `expand(...)` to resolve the policy against the live catalog and get back a concrete set of tools to disable and approval modes to apply.

!!! warning "Disabling is at factory time"
    Tools removed via `disabledMiddleware` or `disabledTools` are never rendered into the model's prompt. This is architecturally different from the approval gate, which fires at dispatch time. Disabled tools cost zero tokens and cannot be called even accidentally; approval-gated tools appear in the prompt but require explicit authorisation before execution.

---

## Related pages

- [Tools & policy](tools.md) - `AgentTool` protocol, `ToolParameter`, `AgentToolPolicy` in detail
- [Subagents](subagents.md) - `SubAgentMiddleware` and the `task` tool
- [Human in the loop](human-in-the-loop.md) - `HumanInTheLoopMiddleware` and approval handlers
- [Summarization](summarization.md) - `SummarizationMiddleware` and `SummarizationConfig`
- [Write custom middleware](../guides/custom-middleware.md) - implementing `AgentMiddleware` yourself
