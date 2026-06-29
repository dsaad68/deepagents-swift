# Gate tools with approvals

DeepAgents provides two complementary mechanisms for requiring human approval before a tool is executed: a global approval handler on the agent, and per-tool interrupt configuration. Together they let you build fine-grained human-in-the-loop workflows.

---

## Why approvals matter

Autonomous agents can call tools with real-world side effects: writing files, running shell commands, sending messages. An approval gate lets your application pause the agent at any tool call and ask a human whether to proceed, modify the arguments, or block the call entirely - without stopping the agent loop.

---

## The global approval handler

Pass an `approvalHandler: ToolApprovalHandler` to `createDeepAgent` to gate every tool call in the run:

```swift
public typealias ToolApprovalHandler = @Sendable (ToolApprovalRequest) async -> ToolDecisionType
```

`ToolApprovalHandler` is a closure. It receives a `ToolApprovalRequest` describing the pending call and must return a `ToolDecisionType` synchronously (from within the async context).

When `approvalHandler` is non-nil, `createDeepAgent` inserts `HumanInTheLoopMiddleware` as the outermost `wrapToolCall` layer. The middleware calls your handler before forwarding to the actual tool.

---

## `ToolApprovalRequest`

`ToolApprovalRequest` describes the pending tool call your handler must evaluate:

| Property | Type | Description |
|---|---|---|
| `toolName` | `String` | The name of the tool being called |
| `arguments` | `[String: AgentJSON]` | The arguments the model passed |
| `describedArguments` | `String` | Human-readable summary of the arguments |

---

## `ToolDecisionType`

Your handler returns one of three decisions:

| Case | Behavior |
|---|---|
| `.approve` | Execute the tool as-is |
| `.deny(reason:)` | Block the call; the reason string is returned to the model as a tool-result error message |
| `.ask` | Block the call and surface a request to the user (used by interactive UIs; behavior depends on your handler) |

---

## Per-tool interrupt configuration

For more granular control, pass `interruptOn: [String: InterruptOnConfig]` to `createDeepAgent`. This lets you mark specific tools as always-interrupt regardless of the global handler:

```swift
let agent = createDeepAgent(
    model: model,
    interruptOn: [
        "shell": InterruptOnConfig(alwaysAsk: true),
        "write_file": InterruptOnConfig(alwaysAsk: true),
    ],
    approvalHandler: myHandler
)
```

`InterruptOnConfig` marks a tool so the approval handler is always invoked for it, even if the handler would otherwise auto-approve.

---

## `AgentToolPolicy.approvals`

`AgentToolPolicy` offers a more declarative approach to per-tool approval modes, useful when you load policy from a config file:

```swift
public struct AgentToolPolicy: Codable, Sendable {
    public var approvals: [String: ToolApprovalMode]
    // ...
}

public enum ToolApprovalMode: String, Codable {
    case approve   // auto-approve
    case ask       // always ask
    case deny      // always deny
}
```

Build a policy and expand it via `policy.expand(catalog:)` to resolve middleware IDs to individual tool names. The expanded result can be used alongside or instead of `interruptOn`.

---

## Console approval handler example

```swift
import DeepAgents

// A synchronous console handler suitable for CLI tools (e.g. Ripple)
let consoleApprovalHandler: ToolApprovalHandler = { request in
    print("\n[Approval required]")
    print("Tool:      \(request.toolName)")
    print("Arguments: \(request.describedArguments)")
    print("Allow? [y/n/d(eny with reason)]: ", terminator: "")

    guard let input = readLine()?.trimmingCharacters(in: .whitespaces).lowercased() else {
        return .deny(reason: "No input received; call blocked by default.")
    }

    switch input {
    case "y", "yes":
        return .approve
    case "n", "no":
        return .deny(reason: "User declined.")
    default:
        // Anything else is treated as a deny with the raw input as the reason
        let reason = input.isEmpty ? "User declined." : "User declined: \(input)"
        return .deny(reason: reason)
    }
}

let agent = createDeepAgent(
    model: model,
    approvalHandler: consoleApprovalHandler
)
```

!!! tip
    In a SwiftUI application, replace `readLine()` with a continuation that suspends until the user taps "Approve" or "Deny" in a sheet or alert. The `ToolApprovalHandler` is `async`, so you can call `await withCheckedContinuation { ... }` inside it.

---

## Interaction with MCP server approval modes

MCP servers each carry an `approvalMode` on their `MCPServerConfig` (`.approve`, `.ask`, `.deny`). This is a coarse server-level filter. The global `approvalHandler` provides the fine-grained per-call gate. Both operate independently: a server-level `.ask` does not call your handler unless `approvalHandler` is also set; a server-level `.deny` blocks tool execution regardless of the handler's decision.

---

## What happens when a call is denied

When your handler returns `.deny(reason:)`, the framework generates a tool-result message containing the denial reason and appends it to the conversation. The model sees the denial as a tool response and may choose to retry with different arguments, ask the user for clarification, or abandon the task. The agent loop continues normally.

---

## Related pages

- [Human in the loop](../concepts/human-in-the-loop.md) - `HumanInTheLoopMiddleware`, `AskUserMiddleware`, and the full approval architecture
- [Tools & policy](../concepts/tools.md) - `AgentToolPolicy`, `ToolApprovalMode`, and `SandboxMode`
