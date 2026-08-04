# Write custom middleware

Middleware is the primary extension point in DeepAgents. It lets you contribute tools, rewrite prompts, intercept model calls, and observe or transform every tool dispatch - all from one protocol conformance.

This guide explains how to implement `AgentMiddleware` and when to use each hook.

---

## The `AgentMiddleware` protocol

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

`name` should be a unique snake-case string (e.g. `"my_logging"`). It is used for disabling middleware via `AgentToolPolicy.disabledMiddleware`.

All methods have default no-op implementations, so you only override the hooks you care about.

---

## Hook order and nesting

Understanding execution order is critical when you register multiple middleware:

**Lifecycle hooks** run in registration order:

```
beforeAgent  [m1 → m2 → m3]  (once, before the run)
  per round:
    beforeModel  [m1 → m2 → m3]
    ... model call ...
    afterModel   [m1 → m2 → m3]
afterAgent   [m1 → m2 → m3]  (once, after the run)
```

**Wrapping hooks** nest - the first-registered middleware is the outermost wrapper:

```
wrapModelCall: m1 wraps (m2 wraps (m3 wraps (actual model call)))
wrapToolCall:  m1 wraps (m2 wraps (m3 wraps (actual tool execute)))
```

This means m1's `wrapModelCall` runs first on the way in and last on the way out - the same semantics as classic middleware stacks. Use this to add timing, logging, or retry logic at the outermost layer, and request-rewriting at inner layers.

---

## Contributing tools

Return tools from the `tools` property. The factory merges them with directly-passed tools and renders them all into the model's prompt:

```swift
struct MyDataMiddleware: AgentMiddleware {
    var name: String { "my_data" }
    var tools: [any AgentTool] { [QueryDatabaseTool(), ListTablesTool()] }
    // ... hooks
}
```

This is the preferred way to group related tools with their lifecycle logic.

---

## Example 1: logging middleware

Use `wrapToolCall` to log every tool invocation with timing:

```swift
import DeepAgents
import Foundation

struct LoggingMiddleware: AgentMiddleware {

    var name: String { "logging" }
    var tools: [any AgentTool] { [] }

    func wrapToolCall(
        _ request: ToolCallRequest,
        _ handler: (ToolCallRequest) async throws -> AgentMessage
    ) async throws -> AgentMessage {
        let start = Date()
        print("[\(request.toolName)] called with: \(request.arguments)")
        do {
            let result = try await handler(request)
            let elapsed = Date().timeIntervalSince(start)
            print("[\(request.toolName)] completed in \(String(format: "%.2f", elapsed))s")
            return result
        } catch {
            print("[\(request.toolName)] threw: \(error)")
            throw error
        }
    }
}
```

Register it as the first middleware so it wraps all inner layers:

```swift
let agent = createAgent(
    model: model,
    tools: myTools,
    middleware: [LoggingMiddleware(), OtherMiddleware()]
)
```

---

## Example 2: system-prompt-augmenting middleware

Use `wrapModelCall` to inject dynamic context into the system prompt every round. This is useful for injecting the current date, user preferences, or retrieved context:

```swift
import DeepAgents
import Foundation

struct ContextInjectionMiddleware: AgentMiddleware {

    var name: String { "context_injection" }
    var tools: [any AgentTool] { [] }

    func wrapModelCall(
        _ request: ModelRequest,
        _ handler: (ModelRequest) async throws -> ModelResponse
    ) async throws -> ModelResponse {
        // Augment the system prompt with dynamic context
        var modified = request
        let dynamicContext = "Current date (UTC): \(ISO8601DateFormatter().string(from: Date()))."
        if let existing = modified.systemPrompt {
            modified.systemPrompt = existing + "\n\n" + dynamicContext
        } else {
            modified.systemPrompt = dynamicContext
        }
        return try await handler(modified)
    }
}
```

!!! note
    `ModelRequest` carries `messages`, `systemPrompt`, and `tools`. All three are mutable in `wrapModelCall`. You can filter messages, reorder tools, or replace the system prompt entirely - the change applies only to this one model call; it does not persist in the agent's conversation history.

---

## Example 3: using lifecycle hooks

Use `beforeAgent`/`afterAgent` for setup and teardown that happens once per run, and `beforeModel`/`afterModel` for per-round work like token counting:

```swift
struct MetricsMiddleware: AgentMiddleware {

    var name: String { "metrics" }
    var tools: [any AgentTool] { [] }

    func beforeAgent(_ state: inout AgentState) async {
        // Called once before the run starts
        print("Run starting for thread: \(state.threadId ?? "anonymous")")
    }

    func afterAgent(_ state: inout AgentState) async {
        // Called once after the run ends (success or failure)
        print("Run complete. Total rounds: \(state.round)")
    }

    func beforeModel(_ state: inout AgentState) async {
        // Called before every model round
        print("Round \(state.round) starting. Messages in thread: \(state.messages.count)")
    }
}
```

---

## Ordering considerations

- **Logging / observability** - register first (outermost wrap).
- **Prompt augmentation** - register after logging so the augmented prompt is what gets logged.
- **Approval gating** (`HumanInTheLoopMiddleware`) - DeepAgents places it last in the `createDeepAgent` stack for a reason: it should see the final tool call after all inner rewrites. Follow the same convention for custom gates.
- **Tool-contributing middleware** - order within the list only matters for wrap nesting; tool availability is unaffected by registration position.

---

## Related pages

- [Middleware](../concepts/middleware.md) - Built-in middleware catalog, hook order, and `AgentToolPolicy`
- [Write a custom tool](custom-tool.md) - Implement `AgentTool` to pair with your middleware
