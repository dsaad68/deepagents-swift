# MLX (on-device)

`DeepAgentsMLX` brings on-device inference to DeepAgents via Apple's
[MLX](https://github.com/ml-explore/mlx-swift) framework. All computation runs on the
Neural Engine and GPU of the host Mac - no network call, no API key, no token metering.

!!! warning "Platform requirements"
    Apple Silicon (arm64) is required. `DeepAgentsMLX` will not build or run on Intel Macs.
    macOS 26+ (Tahoe), Swift 6.1+, and Xcode 26+ are required. Xcode must be the builder because
    it emits MLX's Metal shader library (`.metallib`) as part of the build; `swift build` alone
    does not.

## Model catalog

`MlxModel` is a value type that describes a downloadable model. A static catalog ships with the
package:

```swift
public struct MlxModel: Identifiable, Sendable {
    public enum Kind { case language, vision }
    public let id: String
    public let displayName: String
    public let detail: String
    public let kind: Kind
    public let approxGB: Double
    public var isVision: Bool
    public var agentParameters: GenerateParameters
    public var contextWindowTokens: Int
    public static let catalog: [MlxModel]
    public static var languageCatalog: [MlxModel]
}
```

### LFM2.5 family

All LFM2.5 models run with a 32k-token context window. Recommended sampling parameters are baked
into each `MlxModel.agentParameters` entry so you can pass them straight through.

| Model | Kind | approx. size | Notes |
|---|---|---|---|
| LFM2.5 350M | language | ~0.7 GB | Below the reliable tool-call floor; useful for classification/summarization tasks without tools |
| LFM2.5 1.2B Instruct | language | ~2.4 GB | General-purpose; reliable tool calling |
| LFM2.5 1.2B Thinking | language | ~2.4 GB | Reasoning mode; uses `top_p 0.1` sampling; higher `maxTokens` (8192) to fit `<think>` blocks |
| LFM2.5 8B-A1B | language | ~5 GB | Mixture-of-experts; strongest tool-calling in the family |
| LFM2.5 VL 450M | vision | ~0.9 GB | Vision-language; `supportsVision: true` |
| LFM2.5 VL 1.6B | vision | ~3.2 GB | Vision-language; higher accuracy than 450M |

!!! tip "Choosing a model"
    For most agentic tasks start with **LFM2.5 1.2B Instruct** - small enough to keep on a
    16 GB Mac without noticeable memory pressure, reliable enough for multi-step tool use. Step up
    to **8B-A1B** when you need stronger reasoning or complex nested tool calls. Use a **VL**
    variant when your agent receives screenshots or images in `AgentMessage.images`.

## MlxChatModel

`MlxChatModel` wraps a loaded `ModelContainer` into a `ChatModel`:

```swift
public struct MlxChatModel: ChatModel {
    public init(
        container: ModelContainer,
        supportsVision: Bool,
        ...,
        generateParameters: GenerateParameters
    )
    public func makeSession() -> any ModelTurnSession   // returns RebuildTurnSession
}
```

`makeSession()` returns a `RebuildTurnSession`, which rebuilds the full prompt from the message
history on every call. This is intentional - there is no live KV-cache session boundary to manage,
and it lets middleware rewrite the history freely between rounds.

### RebuildTurnSession

```swift
public final class RebuildTurnSession: ModelTurnSession { ... }
```

Implements `ModelTurnSession`. On each `nextTurn` call it serialises the full `[AgentMessage]`
conversation via `LFM2MessageCodec` and `LFM2ChatTemplate`, runs the MLX generate loop, and
streams chunks back via `onChunk`. Tool calls are extracted by `LFM2ToolCalls`/`LFM2ToolCallStream`
after generation completes.

## MlxModelLoader

`MlxModelLoader` reads models from the local Hugging Face cache at `~/.cache/huggingface/hub`.
Pre-fetch a model with `hf download <id>`; the loader serves it from the cache. The high-level
entry point turns a Hugging Face model id straight into a ready `MlxChatModel`:

```swift
// Typical usage - high-level loader
let loader = MlxModelLoader()
guard let model = await loader.loadChatModel("LiquidAI/LFM2.5-1.2B-Instruct-MLX-bf16") else {
    fatalError("model not in the local Hugging Face cache - run `hf download` first")
}
```

`loadChatModel(_:)` returns `nil` when the model is not available locally, and applies the
catalog's recommended sampling parameters for that id. For finer control, the static
`MlxModelLoader.loadContainer(...)` returns a `ModelContainer` you can wrap with the
`MlxChatModel(container:supportsVision:...:generateParameters:)` initializer shown above.

Pass `model` directly to `createAgent` or `createDeepAgent`.

## LFM2-specific tool-call parsing

LFM2.5 models emit tool calls inside a custom span:

```text
<|tool_call_start|>
{"name": "...", "arguments": {...}}
<|tool_call_end|>
```

`DeepAgentsMLX` suppresses mlx-swift-lm's built-in Pythonic tool-call parser, which truncates
list and dict argument values at the first comma. The framework's own parser (`LFM2ToolCalls`,
`LFM2ToolCallStream`) consumes the full `<|tool_call_start|>...<|tool_call_end|>` span and
correctly preserves nested arrays and objects in argument values.

Reasoning content (`<think>...</think>` blocks emitted by the Thinking variant) is split out by
`LFM2ThinkStream` and surfaced as `AgentContentBlock.reasoning` - it does not appear in the tool
arguments or the text content block.

## Example

```swift
import DeepAgents
import DeepAgentsMLX

// 1. Load a model from the local Hugging Face cache by its id
let loader = MlxModelLoader()
guard let model = await loader.loadChatModel("LiquidAI/LFM2.5-1.2B-Instruct-MLX-bf16") else {
    fatalError("model not in the local Hugging Face cache - run `hf download` first")
}

// 2. Create an agent
let agent = createDeepAgent(
    model: model,
    systemPrompt: "You are a helpful assistant.",
    includeFilesystem: false,
    includeGeneralPurpose: true,
    maxIterations: 12
)

// 3. Run
let ok = await agent.run([.human("Summarise the top-level files in this repo.")]) { event in
    // handle AgentEvent
}
```

## Related

- [Adapters overview](index.md)
- [Architecture](../concepts/architecture.md)
