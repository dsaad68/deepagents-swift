# Messages & content

All information flowing through DeepAgents - user input, model output, tool calls, and tool results - is represented as `AgentMessage`. Understanding the message model is essential for writing middleware, custom tools, and message logging.

---

## `AgentMessage`

```swift
public struct AgentMessage: Sendable, Identifiable {
    public enum Role: String { case system, human, ai, tool }

    public let id: UUID
    public var role: Role
    public var content: [AgentContentBlock]
    public var toolCalls: [AgentToolCall]
    public var malformedToolCallBlocks: [String]
    public var toolCallID: UUID?
    public var source: String?
    public var text: String
    public var reasoning: String?
    public var images: [AgentImage]
}
```

### Roles

| Role | Who produces it | Purpose |
|---|---|---|
| `system` | Your code / middleware | Instruction preamble; not part of the visible conversation |
| `human` | Your code | User turn; may include text and images |
| `ai` | The model | Assistant turn; may include text, reasoning, and tool calls |
| `tool` | The agent loop | Result of a tool execution; paired to an `ai` message by `toolCallID` |

### Static factories

Prefer the static factories over direct initialisation - they set role, content, and convenience properties consistently:

```swift
// System message
AgentMessage.system("You are a helpful assistant.")

// Human turn with optional inline images
AgentMessage.human("What's in this screenshot?", imageURLs: [screenshotURL])

// Assistant turn (typically produced by the model, not your code)
AgentMessage.ai("Here is my answer.", toolCalls: [], reasoning: nil)

// Tool result, paired to a specific tool call
AgentMessage.tool("File contents: ...", toolCallID: callID)
```

### Convenience properties

- `text` - a flat string view of all `.text` content blocks concatenated; useful for display and logging.
- `reasoning` - a flat string view of all `.reasoning` blocks; populated on models that expose chain-of-thought (e.g. LFM2.5 Thinking, OpenAI o-series with `reasoning: true`).
- `images` - a flat array of all `AgentImage` values across content blocks.

---

## `AgentContentBlock`

```swift
public enum AgentContentBlock: Sendable {
    case text(String)
    case reasoning(String)
    case image(AgentImage)
}
```

Content is first-class structured data rather than a flat string. This matters for three reasons:

1. **Multimodal fidelity** - images are embedded at precise positions within the turn, not tacked on separately.
2. **Reasoning transparency** - chain-of-thought from thinking models is preserved as a distinct block type rather than being stripped or concatenated into the visible text.
3. **Adapter correctness** - each backend adapter (MLX, OpenAI, Anthropic) serialises content blocks individually to match its wire format. A flat string would lose the structure needed for correct encoding.

### `AgentImage`

```swift
public struct AgentImage: Sendable {
    public var url: URL?
    public var base64: String?
    public var mimeType: String?
    public var fileID: String?
}
```

An image may be represented as a URL, base64-encoded data, or a provider file ID. The adapter layer converts between these and whatever the backend expects. When `supportsVision` is `false` on the `ChatModel`, image blocks in human messages are silently stripped before the prompt is sent.

---

## `AgentToolCall`

```swift
public struct AgentToolCall: Sendable, Identifiable, Codable {
    public let id: UUID
    public let name: String
    public let arguments: [String: AgentJSON]
    public var describedArguments: String
}
```

Tool calls appear in the `toolCalls` array of an `ai` role message. Each call has:

- `id` - matched against `toolCallID` on the corresponding `.tool` result message.
- `name` - the tool's `name` property; used to dispatch to the correct `AgentTool`.
- `arguments` - a dictionary of `AgentJSON` values parsed from the model's output.
- `describedArguments` - a human-readable summary of the arguments; useful for approval UI and logging without needing to decode `AgentJSON` manually.

### `AgentJSON`

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

`AgentJSON` is a typed JSON value tree. Tool implementations receive `[String: AgentJSON]` in their `execute` method. This enum is fully recursive - arrays and objects may contain any mix of types, including nested objects and arrays.

!!! tip
    Pattern-match `AgentJSON` to extract typed values:
    ```swift
    if case .string(let path) = arguments["path"] {
        // use path
    }
    ```

---

## `malformedToolCallBlocks` and `source`

### `malformedToolCallBlocks`

When a model emits a tool call whose JSON cannot be parsed, the raw block is preserved in `malformedToolCallBlocks` rather than being silently dropped. This gives middleware and logging a chance to inspect and handle parse failures. The DeepAgentsMLX adapter's custom LFM2 parser is specifically designed to minimise these by handling nested arrays and objects that the upstream parser truncates.

### `source`

The `source` field is set to a non-nil string when a message was synthesised by the compaction pipeline rather than produced by a model or tool during a live run. `SummarizationMiddleware` uses this to mark summary messages injected during context compaction, allowing downstream code (middleware, loggers) to distinguish live messages from compaction-synthesised ones.

---

## Message logging

```swift
public protocol AgentMessageLog: Sendable {
    func append(_ message: AgentMessage, threadId: String?, context: AgentLogContext) async
}
```

Pass a conformer via the `messageLog:` parameter to either factory. The built-in `JSONLMessageLog` writes one JSON-encoded `AgentMessage` per line to a timestamped file - suitable for post-run debugging, cost analysis, and audit trails.

```swift
let log = JSONLMessageLog()   // creates a file in the default log directory

let agent = createAgent(
    model: model,
    tools: myTools,
    messageLog: log
)
```

`AgentLogContext` carries metadata about the run (thread ID, iteration number, etc.) alongside each message.

---

## Related pages

- [Tools & policy](tools.md) - `AgentTool`, `ToolContext`, `ToolOutput`, and `AgentJSON` in execute
- [The agent loop](agent-loop.md) - how messages are assembled each round and how the loop dispatches tool calls
