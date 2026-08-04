# Connect MCP servers

DeepAgents ships a built-in MCP client that connects to any number of stdio or HTTP MCP servers concurrently, wraps their tools as `AgentTool` conformers, and feeds them through the same middleware pipeline as every other tool in the agent.

---

## Overview

The connection flow has four steps:

1. Build one `MCPServerConfig` per server.
2. Create a `MultiServerMCPClient` with those configs.
3. Fetch the tools and wrap them in `MCPMiddleware`.
4. Pass the middleware to `createAgent` or `createDeepAgent`.
5. Call `disconnectAll()` when the agent is done.

---

## `MCPServerConfig`

`MCPServerConfig` is a `Codable, Sendable` struct. Two transport kinds are supported:

=== "stdio"

    ```swift
    var config = MCPServerConfig()
    config.name = "filesystem-mcp"
    config.kind = .stdio
    config.command = "/usr/local/bin/mcp-server-filesystem"
    config.args = ["--root", "/tmp/workspace"]
    config.env = ["LOG_LEVEL": "warn"]
    config.isEnabled = true
    config.approvalMode = .ask     // .approve / .ask / .deny
    ```

    The client launches the command as a subprocess and communicates over stdin/stdout. `env` is merged into the subprocess environment.

=== "HTTP"

    ```swift
    var config = MCPServerConfig()
    config.name = "remote-tools"
    config.kind = .http
    config.url = "https://tools.example.com/mcp"
    config.headers = ["Authorization": "Bearer \(apiKey)"]
    config.auth = .none            // .oauth for OAuth 2 flow
    config.isEnabled = true
    config.approvalMode = .approve
    ```

    HTTP transport sends requests to the given URL. For OAuth-protected servers set `auth = .oauth` and supply an `openURL` callback to `MultiServerMCPClient` so the framework can open the authorization URL.

    The OAuth authorizer is attached **lazily**: it only runs the browser flow when the server actually returns a `401`. A server you have signed into before (its token cached in the Keychain) reconnects silently on later runs, even without an `auth = .oauth` declaration.

---

## `MultiServerMCPClient`

```swift
public actor MultiServerMCPClient {
    public init(configs: [MCPServerConfig], openURL: @escaping (URL) -> Void = { _ in })
    public func tools() async -> [any AgentTool]
    public func load() async -> (tools: [any AgentTool], statuses: [MCPServerStatus])
    public func disconnectAll() async
}
```

- `tools()` connects to all enabled servers and returns the union of their tools. Per-server failures are isolated: a server that fails to start or authenticate is skipped and logged; other servers continue normally.
- `load()` does the same but also returns a `[MCPServerStatus]` array so you can surface connection errors in your UI.
- `disconnectAll()` closes all open connections. Call this when the agent's lifetime ends (e.g. in a `defer` block or `deinit`).

---

## `MCPMiddleware`

```swift
public struct MCPMiddleware: AgentMiddleware {
    public init(tools: [any AgentTool])
    public var tools: [any AgentTool]
}
```

`MCPMiddleware` is a thin wrapper that contributes the fetched MCP tools to the agent. It has no lifecycle hooks of its own - all routing logic lives in the `MultiServerMCPClient`.

---

## End-to-end example

```swift
import DeepAgents
import DeepAgentsAnthropic

// 1. Configure servers
var fsConfig = MCPServerConfig()
fsConfig.name = "filesystem-mcp"
fsConfig.kind = .stdio
fsConfig.command = "/usr/local/bin/mcp-server-filesystem"
fsConfig.args = ["--root", FileManager.default.temporaryDirectory.path]
fsConfig.isEnabled = true
fsConfig.approvalMode = .ask

var webConfig = MCPServerConfig()
webConfig.name = "brave-search"
webConfig.kind = .http
webConfig.url = "https://api.search.example.com/mcp"
webConfig.headers = ["Authorization": "Bearer \(braveAPIKey)"]
webConfig.isEnabled = true
webConfig.approvalMode = .approve

// 2. Create the client
let mcpClient = MultiServerMCPClient(configs: [fsConfig, webConfig])

// 3. Fetch tools and build the middleware
let mcpTools = await mcpClient.tools()
let mcpMiddleware = MCPMiddleware(tools: mcpTools)

// 4. Create the agent
let model = AnthropicChatModel(
    baseURL: URL(string: "https://api.anthropic.com")!,
    model: "claude-opus-4-8",
    apiKey: ProcessInfo.processInfo.environment["ANTHROPIC_API_KEY"],
    supportsVision: false
)

let agent = createDeepAgent(
    model: model,
    middleware: [mcpMiddleware],
    systemPrompt: "You are a research assistant."
)

// 5. Run the agent
defer { Task { await mcpClient.disconnectAll() } }

let ok = await agent.run([.human("Summarize the latest news on Swift concurrency.")]) { event in
    // handle events
}
```

---

## Tool namespacing

MCP tools are namespaced with their server name using double underscores: `server__tool_name`. For example, a `read_file` tool from a server named `filesystem-mcp` becomes `filesystem-mcp__read_file` in the agent's tool list. This prevents collisions when multiple servers expose tools with the same name.

---

## Approval mode per server

Each `MCPServerConfig` has an `approvalMode: ToolApprovalMode` property:

| Mode | Behavior |
|---|---|
| `.approve` | All tools from this server are auto-approved (no prompt) |
| `.ask` | Every call pauses for human approval |
| `.deny` | Tools from this server are never executed |

`approvalMode` is applied on top of any `approvalHandler` you pass to `createDeepAgent`. If you pass an `approvalHandler`, it gates all tools; the per-server `approvalMode` provides an additional coarse-grained filter. See [Gate tools with approvals](approvals.md) for the full approval model.

!!! warning
    `.ask` requires an `approvalHandler` to be passed to `createDeepAgent`, otherwise the agent has no mechanism to pause and surface the approval request.

---

## Checking connection status

Use `load()` instead of `tools()` when you need to surface server errors:

```swift
let (tools, statuses) = await mcpClient.load()
for status in statuses {
    if case .failed(let error) = status.state {
        print("Server \(status.name) failed: \(error)")
    }
}
```

---

## Related pages

- [MCP client](../concepts/mcp.md) - Architecture of `MultiServerMCPClient`, tool lifecycle, and isolation model
