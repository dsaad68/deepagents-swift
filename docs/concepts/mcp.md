# MCP client

DeepAgents ships a built-in [Model Context Protocol](https://modelcontextprotocol.io) client that connects to any number of stdio or HTTP MCP servers, exposes their tools as native `AgentTool` instances, and isolates per-server failures so one bad server does not break the rest.

## `MultiServerMCPClient`

```swift
public actor MultiServerMCPClient {
    public init(
        configs: [MCPServerConfig],
        openURL: @escaping (URL) -> Void = { _ in }
    )
    public func tools() async -> [any AgentTool]
    public func load() async -> (tools: [any AgentTool], statuses: [MCPServerStatus])
    public func disconnectAll() async
}
```

| Method | Meaning |
|---|---|
| `init(configs:openURL:)` | Create the client with a list of server configs. `openURL` is called when an HTTP/OAuth flow needs to open a browser URL. |
| `tools()` | Connect to all enabled servers (if not already connected) and return the merged tool list. Failures are logged and skipped. |
| `load()` | Like `tools()`, but also returns a `[MCPServerStatus]` array so callers can inspect which servers connected successfully and which failed. |
| `disconnectAll()` | Gracefully disconnect all active sessions. Call this when your agent is done to avoid leaking subprocess or network connections. |

## `MCPServerConfig`

```swift
public struct MCPServerConfig: Codable, Sendable, Identifiable {
    public enum Kind: String { case stdio, http }
    public enum Auth: String { case none, oauth }

    public var id: UUID
    public var name: String
    public var kind: Kind
    public var isEnabled: Bool

    // stdio fields
    public var command: String
    public var args: [String]
    public var env: [String: String]

    // http fields
    public var url: String
    public var headers: [String: String]
    public var auth: Auth

    // shared
    public var approvalMode: ToolApprovalMode
}
```

### `Kind.stdio` - local subprocess

The client launches `command` with `args` and communicates over stdin/stdout using the MCP stdio transport. `env` is merged into the subprocess environment.

```swift
MCPServerConfig(
    id: UUID(),
    name: "filesystem",
    kind: .stdio,
    isEnabled: true,
    command: "npx",
    args: ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
    env: [:],
    url: "", headers: [:], auth: .none,
    approvalMode: .ask
)
```

### `Kind.http` - remote server

The client connects to `url` over HTTP using the MCP Streamable HTTP transport (or SSE, depending on the server). `headers` are sent with every request; `auth: .oauth` triggers an OAuth 2.0 flow that calls `openURL` when a browser redirect is needed.

```swift
MCPServerConfig(
    id: UUID(),
    name: "remote-search",
    kind: .http,
    isEnabled: true,
    command: "", args: [], env: [:],
    url: "https://my-mcp-server.example.com/mcp",
    headers: ["Authorization": "Bearer \(token)"],
    auth: .none,
    approvalMode: .approve
)
```

### `approvalMode`

Each server config carries an `approvalMode: ToolApprovalMode` that is used by `AgentToolPolicy` when the host manages a policy-driven approval flow. See [Tools & policy](tools.md) for details on `ToolApprovalMode`.

## Tool namespacing

Tools from an MCP server are namespaced by prepending the server name and a double underscore:

```
{server_name}__{tool_name}
```

For example, a server named `filesystem` with a tool named `read_file` becomes `filesystem__read_file` in the agent's tool set. This prevents name collisions across servers and makes it clear in the conversation history which server handled each call.

Both halves are put through `ToolName.normalized`, so the published name is always `[a-z0-9_]`. See [One spelling per tool](#one-spelling-per-tool) below - it applies to every tool, not just MCP ones.

Two servers whose names normalize to the same prefix (`parallel-search` and `parallel_search` both give `parallel_search__`) would otherwise contribute colliding tool names; the loader keeps both reachable by appending a numeric suffix to the later one (`server__tool_2`).

## One spelling per tool

`ToolName.normalized` is the single rule for what a tool is called, and it runs in **both** directions:

- **On the way out**, a tool's name is published to the model through it.
- **On the way in**, every name the model emits is put through it as the message enters the agent loop, and the call is renamed to the matching tool's own spelling.

```swift
ToolName.normalized("parallel-search__web_search")  // "parallel_search__web_search"
ToolName.normalized("Read File")                    // "read_file"
```

The rule: ASCII letters, digits and `_` survive (lowercased); everything else becomes `_`. Models are trained on `[A-Za-z0-9_]` function names and normalize other punctuation away when emitting a call, so a name outside that set cannot round-trip.

Because both ends share one function, there is exactly one spelling of a tool inside the loop - which is what lets dispatch stay an exact match, and what guarantees the approval gate, the deny list, the duplicate-round guard and the message log all key on the same name. A call that matches no tool keeps the model's spelling, so the unknown-tool error quotes what was actually emitted.

!!! note "Why it matters that this is one rule and not two"
    Enforcing the convention only on the way out is what made a hyphenated server unreachable: it published `parallel_search__web_search`, the model emitted `parallel-search__web_search`, and dispatch answered "unknown tool". Papering over that with a second, more forgiving match *at dispatch* then let a call reach the middleware chain under a name the approval gate did not recognise - running an `.ask` (or `.deny`) tool with no prompt at all.

## Attributing a tool to its server

To ask which server contributed a tool, use `toolsFromServer(_:in:)` - never the dispatch-name prefix:

```swift
let searchTools = toolsFromServer("parallel-search", in: agent.tools)
```

Attribution reads the server off the tool itself, through the `ServerScopedTool` protocol:

```swift
public protocol ServerScopedTool: AgentTool {
    /// The contributing server, exactly as configured (unsanitized).
    var serverName: String { get }
    /// The tool's own name on that server, without the namespace prefix.
    var toolName: String { get }
}
```

`MCPTool` conforms, exposing `serverName` and `toolName` publicly. Both `mcpApprovalDefaults(servers:tools:)` and `mcpToolsForDisplay(server:in:)` attribute this way, and a host adding its own server-backed tools should conform too.

`MCPTool.dispatchPrefix(forServer:)` still exists for *building* or displaying a name, but must not be used to attribute one: the prefix is normalized, so distinct server names collapse onto a single prefix and the match is ambiguous. Matching on it meant a tool could take a neighbouring server's approval mode - a `.deny` server's tool running as `.approve`.

## Calls are checked against the server's schema

A server's `inputSchema` is injected into the model's tool list verbatim, so the model sees exactly
what the server declares. Before a call is forwarded, DeepAgents checks one thing: that every
property the schema lists in `required` is present. A call that leaves one out is refused locally
with a message naming it:

```
Missing required argument `urls` (array). You passed `url`, which this tool does not take.
It takes an array, so pass a list even for a single value. Required: `urls`.
```

Left to the server, the same mistake comes back in whatever dialect that server speaks. A Pydantic
dump naming `v1_extract_toolArguments` tells a model nothing it can act on, and a small model
repeats the call unchanged - observed with a 2.6B sending `url` for a `urls` array, twice in a row.

**Only `required` is checked.** Types, `anyOf`, unknown extra arguments and nested requirements all
pass through untouched, and a schema that cannot be parsed falls **open** rather than shut.
Refusing a call the server would have accepted costs a capability with no way to override it, which
is worse than forwarding one the server will reject.

## Per-server failure isolation

If a server fails to connect or crashes mid-run, that server's tools become unavailable but all other servers continue operating normally. The failure is logged; the agent does not see an error unless it actually tries to call one of the unavailable tools.

This means you can list experimental or optional servers in your config without risking the whole agent run if one is offline.

## Lifecycle

The recommended pattern:

```swift
import DeepAgents

// 1. Build the client
let mcpClient = MultiServerMCPClient(configs: [
    MCPServerConfig(
        id: UUID(), name: "search", kind: .stdio, isEnabled: true,
        command: "npx", args: ["-y", "@modelcontextprotocol/server-brave-search"],
        env: ["BRAVE_API_KEY": ProcessInfo.processInfo.environment["BRAVE_API_KEY"] ?? ""],
        url: "", headers: [:], auth: .none, approvalMode: .ask
    )
])

// 2. Connect and get tools (with status)
let (mcpTools, statuses) = await mcpClient.load()
for status in statuses {
    print("\(status.name): \(status.isConnected ? "ok" : "failed")")
}

// 3. Wrap tools in MCPMiddleware and pass to the agent
let agent = createDeepAgent(
    model: model,
    middleware: [MCPMiddleware(tools: mcpTools)]
)

// 4. Run
let ok = await agent.run([.human("Search for Swift 6 concurrency improvements")]) { event in
    // handle events
}

// 5. Disconnect
await mcpClient.disconnectAll()
```

!!! tip "Use `load()` over `tools()` in production"
    `load()` returns connection statuses alongside the tools, letting you log or surface server failures before the agent run starts. `tools()` is a convenience shortcut when you don't need the status array.

## `MCPMiddleware`

```swift
public struct MCPMiddleware: AgentMiddleware {
    public init(tools: [any AgentTool])
    public var tools: [any AgentTool]
}
```

`MCPMiddleware` is a thin wrapper that registers the MCP tools into the agent's tool set. It carries no hooks (no `beforeModel`, no `wrapToolCall`); it exists purely to deliver tools through the standard middleware pipeline. Tool execution calls the underlying MCP session transparently.

!!! note "MCPMiddleware vs. catalog middleware"
    Built-in toolsets like `WebToolsMiddleware` and `FilesystemMiddleware` are in the [middleware catalog](middleware.md) and can be referenced by string ID in `AgentToolPolicy`. `MCPMiddleware` is not in the catalog - it is an ad-hoc middleware you construct at runtime with whatever tools `MultiServerMCPClient` returned.

## Complete example with `createDeepAgent`

```swift
import DeepAgents
import DeepAgentsAnthropic

let mcpClient = MultiServerMCPClient(configs: [
    MCPServerConfig(
        id: UUID(), name: "notes", kind: .stdio, isEnabled: true,
        command: "/usr/local/bin/my-notes-server", args: [],
        env: [:], url: "", headers: [:], auth: .none, approvalMode: .ask
    )
])

let mcpTools = await mcpClient.tools()

let model = AnthropicChatModel(
    baseURL: URL(string: "https://api.anthropic.com")!,
    model: "claude-opus-4-8",
    apiKey: ProcessInfo.processInfo.environment["ANTHROPIC_API_KEY"]
)

let agent = createDeepAgent(
    model: model,
    middleware: [MCPMiddleware(tools: mcpTools)],
    includeGeneralPurpose: true
)

_ = await agent.run([.human("List my recent notes")]) { _ in }

await mcpClient.disconnectAll()
```

The `notes__list_notes`, `notes__read_note`, etc. tools are now available to the agent exactly like built-in tools.

See [Connect MCP servers](../guides/mcp.md) for a step-by-step setup guide including OAuth configuration and policy-driven approval modes.
