# The agent loop

DeepAgents implements a **ReAct** (Reason + Act) loop: the model reasons about a task, calls tools, observes results, and continues until the task is complete or the iteration cap is hit. `ReactAgent` is the concrete type that owns this loop.

---

## `createAgent` vs `createDeepAgent`

Both factories return a `ReactAgent`. Choose based on how much structure you need.

### `createAgent` - minimal, bring your own

```swift
public func createAgent(
    model: any ChatModel,
    tools: [any AgentTool] = [],
    systemPrompt: String? = nil,
    middleware: [any AgentMiddleware] = [],
    memory: (any AgentCheckpointer)? = nil,
    maxIterations: Int = 24,
    disabledToolNames: Set<String> = [],
    messageLog: (any AgentMessageLog)? = nil
) -> ReactAgent
```

Use `createAgent` when you want full manual control: you supply every tool and every middleware yourself. The factory merges the `tools` array with tools contributed by each middleware, then filters out any names in `disabledToolNames`.

### `createDeepAgent` - batteries-included

```swift
public func createDeepAgent(
    model: any ChatModel,
    tools: [any AgentTool] = [],
    systemPrompt: String? = nil,
    subagents: [SubAgent] = [],
    middleware: [any AgentMiddleware] = [],
    memory: (any AgentCheckpointer)? = nil,
    backend: (any FilesystemBackend)? = nil,
    interruptOn: [String: InterruptOnConfig] = [:],
    approvalHandler: ToolApprovalHandler? = nil,
    askUserHandler: AskUserHandler? = nil,
    includeFilesystem: Bool = true,
    includeGeneralPurpose: Bool = true,
    maxIterations: Int = 24,
    disabledToolNames: Set<String> = [],
    messageLog: (any AgentMessageLog)? = nil,
    summarization: SummarizationConfig? = .default
) -> ReactAgent
```

Use `createDeepAgent` when you want the full structural stack out of the box. It composes middleware in a fixed, intentional order:

1. `TodoListMiddleware` - planning discipline; contributes `write_todos`
2. `FilesystemMiddleware` - file I/O via the supplied `backend` (defaults to `StateBackend` when `includeFilesystem` is true)
3. `SubAgentMiddleware` - delegation via the `task` tool; wires up `subagents`
4. Your additional `middleware` (inserted here, in the order you provide)
5. `AskUserMiddleware` - lets the model pause and ask the user a question (only when `askUserHandler != nil`)
6. `HumanInTheLoopMiddleware` - approval gating on every tool call (only when `approvalHandler != nil`)
7. `SummarizationMiddleware` - automatic context compaction (when `summarization != nil`)

`includeGeneralPurpose: true` adds the web, search, text, git, and shell capability middleware. See [Middleware](middleware.md) for the full capability catalog.

---

## One ReAct round - step by step

A single run of `agent.run(...)` may span many rounds. Each round is:

```
┌─────────────────────────────────────────────┐
│  1. beforeModel middleware hooks fire        │
│  2. session.nextTurn(messages, tools, ...)   │  ← full conversation rebuilt from scratch
│     streaming tokens → onEvent(.token(...)) │
│  3. afterModel middleware hooks fire         │
│  4. assistant turn appended to thread        │
│  5. If no tool calls → done (final answer)  │
│  6. duplicate-round guard check              │
│  7. For each tool call (in parallel):        │
│       wrapToolCall nest → execute            │
│       onEvent(.toolStarted / .toolCompleted)│
│  8. Tool result messages appended            │
│  9. Repeat from step 1                       │
└─────────────────────────────────────────────┘
```

### The model turn

`ReactAgent` creates one `ModelTurnSession` per `run(...)` call. Each round, it calls:

```swift
session.nextTurn(
    messages: [AgentMessage],   // full thread rebuilt every round
    systemPrompt: String?,
    tools: [any AgentTool],
    onChunk: ...
)
```

The session is stateless from the agent's perspective: the full conversation history is passed in every time. This matters because any middleware that rewrites messages in `beforeModel` or `wrapModelCall` sees a complete, consistent view each round.

### Tool dispatch

When the model returns an assistant message containing tool calls, the loop:

1. Passes each call through the `wrapToolCall` middleware nest (first-registered middleware is outermost).
2. Calls `tool.execute(arguments, context)` on the appropriate `AgentTool`.
3. Appends a `.tool` role message for each result.
4. Emits `AgentEvent.toolStarted` before and `AgentEvent.toolCompleted` after each call.

### The duplicate-round guard

Before dispatching tool calls for a round, the loop compares the tool call set against the previous round. If the model emits the exact same calls again (identical names and arguments), the round is skipped to prevent infinite loops in cases where a tool keeps returning the same output.

### `maxIterations` and the forced final answer

If the loop reaches `maxIterations` without the model producing a no-tool turn, the agent forces one final call to the model with tools stripped from the prompt. This guarantees the agent always produces a human-readable answer rather than silently stopping.

---

## `ReactAgent` - the surface you call

```swift
public struct ReactAgent: Sendable {
    public func run(
        _ input: [AgentMessage],
        threadId: String? = nil,
        onEvent: @Sendable @escaping (AgentEvent) -> Void
    ) async -> Bool

    public var contextWindowTokens: Int?

    @discardableResult
    public func compact(threadId: String?) async -> CompactionOutcome?
}
```

- `run(...)` returns `true` on success, `false` on failure. Failures are also delivered via `onEvent(.failed(error))`.
- `threadId` keys the conversation in the `AgentCheckpointer` (memory). Passing the same `threadId` across calls gives the agent short-term memory of prior turns.
- `compact(threadId:)` triggers manual context compaction for a thread. Normally `SummarizationMiddleware` handles this automatically, but you can call it explicitly - for example, before archiving a conversation.

---

## The `AgentEvent` stream

The `onEvent` closure receives a stream of typed events as the run progresses. The principal cases are:

| Event | When it fires |
|---|---|
| `.token(text, ...)` | A streamed token from the model arrives |
| `.toolStarted(name, input)` | A tool call is about to be dispatched |
| `.toolCompleted(name, result, ...)` | A tool call has returned |
| `.completed` | The run finished successfully |
| `.failed(error)` | The run failed; error is attached |

!!! note
    The `onEvent` closure is `@Sendable` and may be called from a non-main actor context. If you're updating UI, dispatch to the main actor inside the closure.

Use the event stream to drive progress indicators, log tool usage, stream assistant tokens to a chat UI, or observe costs and timing.

---

## Message logging

Pass a `messageLog:` conformer to either factory to record every message the agent appends. `JSONLMessageLog` writes one JSON line per message to a timestamped file - useful for debugging and audit trails. See [Messages & content](messages.md) for the `AgentMessageLog` protocol.

---

## Related pages

- [Messages & content](messages.md) - `AgentMessage`, roles, content blocks, and `AgentJSON`
- [Middleware](middleware.md) - hook order, built-in middleware, capability catalog
- [Summarization](summarization.md) - automatic context compaction
