# Quickstart

This guide walks you through building and running a deep agent from scratch. You will:

1. Construct a `ChatModel` backed by either Anthropic cloud or on-device MLX.
2. Create a deep agent with `createDeepAgent`.
3. Run the agent and handle streamed events.

## 1. Build a model

The `ChatModel` protocol is the only seam between the agent and the backend. Pick the adapter that matches your deployment target.

=== "Anthropic (cloud)"

    ```swift
    import DeepAgents
    import DeepAgentsAnthropic

    let model = AnthropicChatModel(
        baseURL: URL(string: "https://api.anthropic.com")!,
        model: "claude-opus-4-8",
        apiKey: ProcessInfo.processInfo.environment["ANTHROPIC_API_KEY"],
        supportsVision: false
    )
    ```

    `AnthropicChatModel` targets the Anthropic Messages API at `POST {baseURL}/v1/messages`. Set `supportsVision: true` if you plan to send images via `.human(_:imageURLs:)`.

    For AWS Bedrock, use `BedrockChatModel` with a `BedrockAuth` value - `BedrockAuth.resolve()` picks a bearer token (`AWS_BEARER_TOKEN_BEDROCK`) or SigV4 env credentials, or pass `.sigV4(...)` / `.bearerToken(...)` explicitly (bearer auth also needs a verbatim `baseURL`). The model id can be a cross-region inference-profile id (e.g. `us.anthropic.claude-...`).

=== "On-device MLX"

    On-device inference requires Apple Silicon and the `DeepAgentsMLX` product.

    ```swift
    import DeepAgents
    import DeepAgentsMLX

    // MlxModelLoader builds an MlxChatModel from a Hugging Face model id,
    // reading from your local Hugging Face cache. Returns nil if unavailable.
    let loader = MlxModelLoader()
    guard let model = await loader.loadChatModel("LiquidAI/LFM2.5-1.2B-Instruct-MLX-bf16") else {
        fatalError("model not in the local Hugging Face cache - run `hf download` first")
    }
    ```

    `MlxModel.catalog` lists the LFM2.5 family by Hugging Face id: 350M, 1.2B Instruct, 1.2B Thinking, 8B-A1B, and vision variants (450M VL, 1.6B VL). The loader applies each model's recommended sampling parameters automatically.

    !!! warning "350M below tool floor"
        The 350M model is below the reliable tool-call floor. Use 1.2B Instruct or larger for agents that need to call tools.

## 2. Create a deep agent

`createDeepAgent` wires the three structural "pillars" automatically and returns a `ReactAgent`:

```swift
let agent = createDeepAgent(
    model: model,
    systemPrompt: "You are a helpful coding assistant.",
    backend: StateBackend(),          // in-memory filesystem
    includeFilesystem: true,          // planning + filesystem pillars on
    includeGeneralPurpose: true,      // general-purpose toolset on
    maxIterations: 24
)
```

The three built-in pillars, wired in order, are:

| Pillar | Middleware | What it adds |
|---|---|---|
| Planning | `TodoListMiddleware` | `write_todos` tool + guidance appended to the system prompt |
| Filesystem | `FilesystemMiddleware` | `ls`, `read_file`, `write_file`, `edit_file`, `mkdir` (backed by `backend:`) |
| Subagents | `SubAgentMiddleware` | `task` tool for delegating subtasks to named subagents |

Set `includeFilesystem: false` or `includeGeneralPurpose: false` to remove pillars you do not need. You can also pass additional `tools:` (any `AgentTool` values), `subagents:`, and `middleware:` to extend the agent further.

!!! tip "Use StateBackend for sandboxed or test runs"
    `StateBackend()` keeps all filesystem operations in memory. Use `LocalFilesystemBackend(root:)` to operate on real files. Passing `backend: nil` with `includeFilesystem: false` disables the filesystem pillar entirely.

## 3. Run the agent

Call `agent.run(_:threadId:onEvent:)` with the initial message array and an event handler:

```swift
let ok = await agent.run(
    [.human("Summarize the files in the current directory.")],
    threadId: "session-42"
) { event in
    switch event {
    case .token(let text, _):
        print(text, terminator: "")
    case .toolStarted(let name, let input):
        print("\n[tool] \(name)(\(input))")
    case .toolCompleted(let name, let result, _):
        print("[done] \(name): \(result)")
    case .completed:
        print("\n[agent done]")
    case .failed(let error):
        print("\n[error] \(error)")
    default:
        break
    }
}
```

`run` returns `true` on success and `false` when an unrecoverable error occurs (the error is also delivered through `.failed`). The call is `async` and suspends until the agent reaches a final answer or `maxIterations` is exhausted. When `maxIterations` is hit, the framework forces a final no-tools answer rather than erroring out.

### The threadId

`threadId` is an optional string key for persistent memory. If you pass a `memory:` parameter to `createDeepAgent` (an `AgentCheckpointer`), the agent loads prior conversation history for that key before each run and saves it after. Passing the same `threadId` across calls gives the agent continuity across multiple interactions with the same user or session. Pass `nil` for stateless one-shot runs.

### What happens per round

Each round the agent:

1. Calls `session.nextTurn(messages:systemPrompt:tools:onChunk:)` with the full conversation history, yielding tokens via `onChunk` (surfaced as `.token` events).
2. Parses tool calls from the response.
3. If tool calls are present, dispatches each one (surfaced as `.toolStarted` / `.toolCompleted`), appends results to history, and loops.
4. If no tool calls are present, the response is the final answer and the run ends with `.completed`.

Middleware hooks fire around each model call (`wrapModelCall`) and each tool call (`wrapToolCall`), enabling logging, retry, approval gating, and prompt rewriting. See [Middleware](../concepts/middleware.md) for details.

## Next steps

- [Architecture](../concepts/architecture.md) - The inference-agnostic design and how the ReAct loop works.
- [The agent loop](../concepts/agent-loop.md) - Round-by-round mechanics, duplicate detection, and iteration limits.
- [Middleware](../concepts/middleware.md) - Composing capabilities and writing your own.
- [Tools & policy](../concepts/tools.md) - The `AgentTool` protocol, tool parameters, and approval modes.
- [Adapters](../adapters/index.md) - Full reference for all three backend adapters.
