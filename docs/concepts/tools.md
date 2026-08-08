# Tools & policy

A tool is anything the agent can call to act on the world. DeepAgents represents tools as `AgentTool` conformers - a simple Swift protocol you can implement to add any capability, from file I/O to network requests to native macOS APIs.

---

## `AgentTool` protocol

```swift
public protocol AgentTool: Sendable {
    var name: String { get }
    var description: String { get }
    var parameters: [ToolParameter] { get }

    func execute(
        _ arguments: [String: AgentJSON],
        _ context: ToolContext
    ) async throws -> ToolOutput

    func toolSchema() -> ToolSchema

    var isParallelSafe: Bool { get }
}
```

| Member | Purpose |
|---|---|
| `name` | Stable identifier; used for dispatch, disabling, and approval rules. Normalized to `[a-z0-9_]` - see below |
| `description` | Natural-language description sent to the model in its tool schema |
| `parameters` | Typed parameter list; drives JSON schema generation and model guidance |
| `execute(_:_:)` | Called when the model invokes this tool; receives parsed arguments and context |
| `toolSchema()` | Returns the JSON schema representation sent to the backend; derived from `parameters` by default |
| `isParallelSafe` | Whether calls to this tool may run concurrently with the round's other parallel-safe calls; defaults to `false` |

!!! note "Names are normalized"
    A tool's `name` is put through `ToolName.normalized` in both directions: it is what the model
    is shown, and every name the model sends back goes through the same rule before it is matched,
    so a call finds its tool however the model spelled it. ASCII letters, digits and `_` survive
    (lowercased); everything else becomes `_`. Pick a name in that alphabet and what you write is
    what dispatch sees. See [One spelling per tool](mcp.md#one-spelling-per-tool).

!!! tip
    Descriptions matter. The model decides whether to call a tool - and how to fill its arguments - based almost entirely on `description` and the `description` fields of each parameter. Write descriptions as you would a function docstring: state the effect, not the implementation.

---

## `ToolParameter`

```swift
public enum ToolParameterType: Sendable {
    case string
    case bool
    case int
    case double
    case array(elementType: ToolParameterType)
    case object(properties: [ToolParameter])
    case data
}

public struct ToolParameter: Sendable {
    public let name: String
    public let type: ToolParameterType
    public let description: String
    public let isRequired: Bool
    public let extraProperties: [String: any Sendable]

    public static func required(
        _ name: String,
        type: ToolParameterType,
        description: String
    ) -> ToolParameter

    public static func optional(
        _ name: String,
        type: ToolParameterType,
        description: String
    ) -> ToolParameter
}
```

### Parameter types

| Type | Swift equivalent | Notes |
|---|---|---|
| `string` | `String` | Most common; use for paths, queries, free text |
| `bool` | `Bool` | Flags and toggles |
| `int` | `Int` | Integer counts, offsets, line numbers |
| `double` | `Double` | Floating-point values |
| `array(elementType:)` | `[T]` | Homogeneous list; specify the element type |
| `object(properties:)` | Structured dict | Nested parameter object; properties are themselves `[ToolParameter]` |
| `data` | `Data` | Binary payload; encoded as base64 in the JSON schema |

### Required vs optional

```swift
// Model must provide this argument
ToolParameter.required("path", type: .string, description: "Absolute path to the file")

// Model may omit this argument
ToolParameter.optional("encoding", type: .string, description: "Text encoding, defaults to UTF-8")
```

---

## `ToolContext` and `ToolOutput`

### `ToolContext`

`ToolContext` is passed to every `execute` call. It carries the agent's live state, allowing a tool to read current messages, access the filesystem backend, and emit progress events during a long-running operation.

### `ToolOutput`

```swift
// Conceptual shape - result text + optional state update
ToolOutput(text: "File written successfully.", stateUpdate: nil)
```

`ToolOutput` carries:

- A result string that is returned to the model as the tool's output message.
- An optional state update that the agent loop applies to `AgentState` after the tool returns.

Return a clear, concise result string. The model reads this to decide its next action, so vague output ("done") is less useful than specific output ("Wrote 142 bytes to /tmp/output.txt").

---

## Parallel-safe tools

A round's tool calls run one after another by default, and each call sees the state and results of the calls before it. A tool can opt out of that ordering:

```swift
struct ReadFileTool: AgentTool {
    var name: String { "read_file" }
    // …
    var isParallelSafe: Bool { true }
}
```

The loop then runs each **run of consecutive parallel-safe calls** as one concurrent batch (at most four at a time), so a round of three `read_file` calls costs one read instead of three. Results still come back in call order; the events show the batch running and each call finishing as it lands, which is why they carry a `callID` to pair on. See [The agent loop](agent-loop.md#parallel-safe-calls) for the batching rules.

Declaring `true` asserts two things about the tool:

- **It is safe to run concurrently with itself and with other tools.** Anything holding a lock, a subprocess working tree, or a device the OS serialises is a poor candidate.
- **It does not need an earlier call's result or state update.** Every call in a batch is handed the same state snapshot, taken when the batch started - siblings are invisible to each other.

Anything that writes - `write_file`, `edit_file`, `shell`, `write_clipboard`, `task` - should leave the default in place. So should a tool whose ordering relative to a writer matters, and a tool whose semantics you don't control (`MCPTool` defaults to `false` for exactly this reason).

Two further rules the runtime applies for you:

- A **gated** tool still fans out. Only the approval *request* is serialised - one card at a time, the next raised when the previous decision returns - so a gated tool costs nothing extra when the host answers the gate itself. (Read-only tools default to `ask`, so the opposite rule would make this feature inert.)
- Middleware `wrapToolCall` chains do run concurrently around a batch. A middleware whose wrapper is order-sensitive should not wrap parallel-safe tools.

Among the built-in toolsets, the read-only tools declare it: `ls`, `read_file`, `grep`, `glob`, `tree`, `head`, `tail`, `diff`, `fetch`, every `git_*` tool, `current_datetime`, `calculator`, `mdfind`, and `read_clipboard`.

---

## Generic toolsets

The following toolsets ship in DeepAgents. Each toolset is owned by a middleware that contributes its tools via the `tools` property. See [Middleware](middleware.md) for the full capability catalog table and adapter requirements.

| ID | Middleware | Tools |
|---|---|---|
| `web` | `WebToolsMiddleware` | `fetch`, `curl` |
| `search` | `SearchToolsMiddleware` | `grep`, `glob`, `tree` (see [walk limits](#search-walk-limits)) |
| `text` | `TextToolsMiddleware` | `head`, `tail`, `diff` |
| `git` | `GitToolsMiddleware` | `git_status`, `git_diff`, `git_log`, `git_show`, `git_blame` |
| `shell` | `ShellToolsMiddleware` | `shell` |
| `macos` | `MacToolsMiddleware` | `mdfind`, `open`, `open_app`, `download`, `say`, `notify` |
| `filesystem` | `FilesystemMiddleware` | `ls`, `read_file`, `write_file`, `edit_file`, `mkdir` |
| `clipboard` | `ClipboardMiddleware` | `read_clipboard`, `write_clipboard` |
| `screenshot` | `ScreenshotMiddleware` | `take_screenshot`, `take_window_screenshots` |
| `apple_notes` | `AppleNotesMiddleware` | `list_notes`, `read_note`, `create_note`, `update_note` |
| `container` | `ContainerShellMiddleware` | `container_shell` |

!!! note "`shell` and `ShellGuard`"
    The `shell` tool is gated by `ShellGuard`, which enforces command allowlisting. Raw shell access is powerful and should be paired with an approval handler in production. See [Human in the loop](human-in-the-loop.md).

---

## `AgentToolPolicy`

`AgentToolPolicy` is a `Codable`, `Sendable` struct that captures tool governance preferences in a serialisable form - suitable for persisting user settings between sessions.

```swift
public struct AgentToolPolicy: Codable, Sendable {
    public var disabledMiddleware: Set<String>          // catalog IDs to exclude entirely
    public var disabledTools: Set<String>               // individual tool names to exclude
    public var approvals: [String: ToolApprovalMode]   // per-tool or per-middleware approval rules
    public var sandbox: SandboxMode                    // .off / .failover / .containerOnly
    public var sandboxImage: String?                   // Docker image for container sandbox

    public func expand(
        catalog: [MiddlewareDescriptor] = MiddlewareCatalog.all,
        extraDefaults: [String: ToolApprovalMode] = [:]
    ) -> Expansion
}
```

Call `expand(...)` to resolve a policy against the live `MiddlewareCatalog` and get back the concrete sets of tools to disable and the per-tool approval modes to pass to `HumanInTheLoopMiddleware`.

### `ToolApprovalMode`

```swift
public enum ToolApprovalMode: String, Codable {
    case approve   // always allow without asking
    case ask       // show approval UI before executing
    case deny      // always block
}
```

### `SandboxMode`

```swift
public enum SandboxMode {
    case off            // no sandboxing; shell runs on the host
    case failover       // try container first; fall back to host if unavailable
    case containerOnly  // all shell commands routed through container_shell
}
```

### Disabling happens at factory time

When you pass `disabledToolNames:` to `createAgent` / `createDeepAgent`, or expand a policy with disabled tools, the excluded tools are removed from the merged tool list **before** the agent is constructed. The model never sees these tools in its schema. This is architecturally different from the approval gate:

| Mechanism | When it fires | Model sees the tool? |
|---|---|---|
| `disabledTools` / `disabledToolNames` | Factory construction | No - never rendered |
| `ToolApprovalMode.deny` | Dispatch time (wrapToolCall) | Yes - but call is blocked |
| `ToolApprovalMode.ask` | Dispatch time (wrapToolCall) | Yes - pending user confirmation |

Prefer disabling at factory time for tools that should never be available in a given context (e.g. no filesystem writes in a read-only agent). Use `ask` / `deny` approval modes for tools that should be available but audited or constrained at runtime.

---

## Search walk limits

`grep`, `glob` and `tree` share one file walk, with two limits worth knowing about.

**Build output is skipped.** `build`, `DerivedData`, `node_modules`, `Pods`, `Carthage` and
`__pycache__` are never descended into. They are generated, routinely far larger than the source
they came from, and never what a search is looking for - in the Mispher repo, `Ripple/build` is
23,288 files, 96% of everything under the root. `tree` names such a folder rather than descending
it, so the layout stays honest.

The list is deliberately short and excludes names that plausibly hold real sources (`target`,
`dist`, `site`, `vendor`). A wrongly skipped directory is a search that confidently misses
something real - the same defect as the one below, from the other direction.

**A truncated walk says so.** The walk stops after `FileWalk.maxFiles` (20,000). When it does, the
tools report that they searched the first N files and stopped, rather than "No matches":

```
Searched the first 20000 files under "." and found no match for /pattern/, then stopped -
the folder holds more than that, so the pattern may still be present in what was not
searched. Narrow it with `path` or `include`.
```

This matters more than the cap itself. Reporting a truncated search as an absence gives the model a
false fact it cannot detect, and it will act on it - in one measured case, spending 18 rounds trying
to reconcile "no matches" with a question that presumed the symbol existed. A partial result that
admits it is partial costs a follow-up call; a wrong one costs the whole turn.

---

## Related pages

- [Middleware](middleware.md) - how middleware contributes tools, the capability catalog, and hook order
- [Write a custom tool](../guides/custom-tool.md) - step-by-step guide to implementing `AgentTool`
- [Human in the loop](human-in-the-loop.md) - `ToolApprovalMode`, approval handlers, and the `ask` flow
