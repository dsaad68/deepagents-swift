# Build a custom ChatModel

DeepAgents is inference-agnostic by design. The `ChatModel` protocol is the only seam between an agent and its backend. Conforming to it is all you need to connect any inference engine - a remote API, a local runtime, or a mock for testing.

This guide walks through a full custom backend implementation.

---

## The two protocols

Two protocols form the backend contract:

**`ChatModel`** is a stateless factory. It describes the model's capabilities and creates per-run sessions on demand:

```swift
public protocol ChatModel: Sendable {
    var supportsVision: Bool { get }
    var modelID: String? { get }
    var contextWindowTokens: Int? { get }
    func makeSession() -> any ModelTurnSession
}
```

**`ModelTurnSession`** performs a single assistant turn. It receives the full conversation and returns one assistant message:

```swift
public protocol ModelTurnSession: AnyObject {
    func nextTurn(
        messages: [AgentMessage],
        systemPrompt: String?,
        tools: [any AgentTool],
        onChunk: @escaping @Sendable (AgentStreamChunk) -> Void
    ) async throws -> AgentMessage
}
```

`ReactAgent` calls `makeSession()` once per `run(...)` invocation, then calls `nextTurn(...)` once per ReAct round, passing the **entire conversation** each time.

---

## The stateless rebuild contract

This is the most important design constraint to understand: `ModelTurnSession` is stateless from the framework's perspective. The agent rebuilds the full prompt from the `messages` array on every call - it does not maintain a live KV cache or cumulative token stream across rounds.

Why this matters:

- Middleware can freely rewrite or filter `messages` in `beforeModel` and `wrapModelCall` without corrupting state.
- `wrapModelCall` can safely retry: there is no partial state to roll back.
- Summarization can replace earlier turns with a compressed summary and the session picks up the new history cleanly.

Your session implementation should treat `messages` as the authoritative source of truth each call. Do not cache tokens or hidden state between `nextTurn` calls.

---

## Writing a message codec

You need to convert between `AgentMessage` (DeepAgents' universal format) and your backend's wire format. The existing adapters each implement a `MessageCodec`-like component (`LFM2MessageCodec`, `OpenAIMessageCodec`, `AnthropicMessageCodec`). You should do the same.

The key mapping tasks are:

1. **Roles** - `AgentMessage.Role` has four cases: `.system`, `.human`, `.ai`, `.tool`. Map each to your backend's equivalent.
2. **Content blocks** - `AgentContentBlock` carries `.text(String)`, `.reasoning(String)`, and `.image(AgentImage)`. Backends that don't support reasoning or images should skip those blocks gracefully.
3. **Tool calls** - `AgentToolCall` carries a `name` and `arguments: [String: AgentJSON]`. Serialize `AgentJSON` to your backend's JSON format. On the response side, parse the backend's tool-call output back into `AgentToolCall` values.
4. **Tool results** - messages with `.tool` role carry the output of a prior tool call; they reference the originating call via `toolCallID`.

`AgentJSON` is a typed enum covering all JSON value kinds:

```swift
public enum AgentJSON: Sendable, Codable {
    case null
    case bool(Bool)
    case int(Int)
    case double(Double)
    case string(String)
    case array([AgentJSON])
    case object([String: AgentJSON])
}
```

Encode it recursively to whatever representation your wire format expects (e.g. `Any`, a `Codable` struct, raw JSON bytes).

---

## Minimal skeleton

```swift
import DeepAgents
import Foundation

// 1. The session - one per run, stateless between rounds
final class MyModelSession: ModelTurnSession {

    private let endpoint: URL
    private let apiKey: String
    private let modelName: String

    init(endpoint: URL, apiKey: String, modelName: String) {
        self.endpoint = endpoint
        self.apiKey = apiKey
        self.modelName = modelName
    }

    func nextTurn(
        messages: [AgentMessage],
        systemPrompt: String?,
        tools: [any AgentTool],
        onChunk: @escaping @Sendable (AgentStreamChunk) -> Void
    ) async throws -> AgentMessage {
        // 1. Convert messages -> your wire format
        let wireMessages = encode(messages: messages, systemPrompt: systemPrompt)

        // 2. Convert tools -> your wire format
        let wireTools = tools.map { $0.toolSchema() }

        // 3. Call your backend (streaming or non-streaming)
        let response = try await callBackend(
            messages: wireMessages,
            tools: wireTools
        )

        // 4. Parse the response back into an AgentMessage
        return decode(response: response)
    }

    // MARK: - Codec helpers (implement for your format)

    private func encode(messages: [AgentMessage], systemPrompt: String?) -> [[String: Any]] {
        // Map AgentMessage role/content/toolCalls to your wire format
        fatalError("implement me")
    }

    private func decode(response: Data) -> AgentMessage {
        // Parse assistant text and/or tool calls from the response
        fatalError("implement me")
    }

    private func callBackend(messages: [[String: Any]], tools: [ToolSchema]) async throws -> Data {
        fatalError("implement me")
    }
}

// 2. The model - a stateless factory
public struct MyBackendChatModel: ChatModel {

    public var supportsVision: Bool { false }
    public var modelID: String? { modelName }
    public var contextWindowTokens: Int? { 32_768 }

    private let endpoint: URL
    private let apiKey: String
    private let modelName: String

    public init(endpoint: URL, apiKey: String, modelName: String) {
        self.endpoint = endpoint
        self.apiKey = apiKey
        self.modelName = modelName
    }

    public func makeSession() -> any ModelTurnSession {
        MyModelSession(endpoint: endpoint, apiKey: apiKey, modelName: modelName)
    }
}
```

Pass it to any factory just like a built-in adapter:

```swift
let model = MyBackendChatModel(
    endpoint: URL(string: "https://my-backend.example/v1/chat")!,
    apiKey: "sk-...",
    modelName: "my-model-7b"
)

let agent = createAgent(model: model, tools: myTools)
```

---

## Vision support

If your backend accepts image inputs, set `supportsVision: true` and handle `AgentContentBlock.image(AgentImage)` in your codec. `AgentImage` carries an optional `url`, `base64` string, `mimeType`, and `fileID` - use whichever your backend accepts.

When `supportsVision` is `false`, the framework will still pass image blocks in the messages array (they come from the user's input). Your codec should simply skip or strip them.

---

## Streaming tokens

If your backend supports streaming, call `onChunk` for each arriving token. `AgentStreamChunk` is what the framework uses to propagate `.token` events upstream to the `onEvent` callback on `ReactAgent.run`. If your backend does not stream, you can ignore `onChunk` entirely and return the completed `AgentMessage` directly - the agent loop works either way.

---

## Testing with a mock

Because `ChatModel` and `ModelTurnSession` are pure protocols, you can write a deterministic mock for unit tests:

```swift
struct EchoModel: ChatModel {
    var supportsVision: Bool { false }
    var modelID: String? { "echo" }
    var contextWindowTokens: Int? { nil }

    func makeSession() -> any ModelTurnSession { EchoSession() }
}

final class EchoSession: ModelTurnSession {
    func nextTurn(
        messages: [AgentMessage],
        systemPrompt: String?,
        tools: [any AgentTool],
        onChunk: @escaping @Sendable (AgentStreamChunk) -> Void
    ) async throws -> AgentMessage {
        .ai("Echo: \(messages.last?.text ?? "")")
    }
}
```

---

## Related pages

- [Architecture](../concepts/architecture.md) - How `ChatModel` fits into the broader agent design
- [Adapters](../adapters/index.md) - The three built-in adapters and when to use each
