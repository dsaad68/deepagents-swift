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
