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
history on every call. This is intentional - it lets middleware rewrite the history freely between
rounds, and the chat template re-renders the exchange faithfully (dropping historical `<think>`
blocks). Prompt *processing* is still incremental: the session reuses the KV/SSM state of the
unchanged token prefix (see below), so rebuilding does not mean re-prefilling.

### RebuildTurnSession

```swift
public final class RebuildTurnSession: ModelTurnSession { ... }
```

Implements `ModelTurnSession`. On each `nextTurn` call it serialises the full `[AgentMessage]`
conversation via the model's codec, runs the MLX generate loop, and streams chunks back via
`onChunk`. Tool calls are extracted per codec family (see below).

### Prefix caching

The system prompt and tool schemas dominate the prompt (often ~10k tokens) and are identical
across a run's rounds - and across runs on the same model + configuration. `RebuildTurnSession`
reuses their computed KV/SSM state through a `PrefixCacheSlot` so each round prefills only the
tokens that changed:

```swift
public final class PrefixCacheSlot { public init() }

public struct MlxChatModel: ChatModel {
    public init(..., prefixCache: PrefixCacheSlot? = nil)
}
```

- With `prefixCache: nil`, reuse is **within-run**: rounds 2+ of a ReAct loop prefill only the new
  tool results instead of the whole prompt.
- Passing a shared per-model `PrefixCacheSlot` extends reuse **across queries**: a new conversation
  on a resident model reuses the system+tools prefix. `MlxModelLoader` does this automatically and
  drops the slot on `unload(_:)` (the state is tied to the resident container).

Fully-trimmable caches are rewound with `KVCache.trim`. The LFM2.5 and qwen3_5 families are
attention + Mamba/conv hybrids whose recurrent layer state cannot be rewound - and Gemma 4's
sliding-window layers use a `RotatingKVCache` that claims trim success even after the window has
rotated the tokens away - so the session keeps `copy()`-snapshots at two boundaries instead: a
*tip* just before the end of the latest prompt (next round resumes there) and a *base* at the
first observed cross-conversation divergence - in practice the system+tools boundary (a new query
resumes there). Reuse is keyed on the longest
common **token** prefix, so a middleware rewrite, a dropped `<think>` block, or a different user
turn simply falls after the reused prefix and is re-prefilled; image turns bypass caching entirely
(image embeddings are not represented in the text tokens). In practice this takes a multi-round
turn's time-to-first-token from ~30 s to under a second on a 9B model with a ~10k-token prompt.

### Disk persistence

The base snapshot also survives across processes. After a turn, the session persists it under
`~/.cache/deepagents/prefix-kv/` (via mlx-swift-lm's prompt-cache serialization, which round-trips
the hybrid models' recurrent `MambaCache` too); a fresh process resumes from it on its very first
turn and prefills only the conversation suffix - so a one-shot CLI run or an app relaunch skips the
multi-second prompt prefill entirely.

Base snapshots are **content-addressed**: keyed by model id + a hash of the exact tokens they
cover, not by configuration. On a cold start the store picks the *longest* stored base whose
tokens are a strict prefix of the incoming prompt - whichever configuration wrote it. Configs that
share a prompt prefix therefore share snapshots: if an MCP server is down or a tool set changed,
the run still resumes from the shared part of the prompt and re-prefills only what differs, then
*deepens* the snapshot to its own stable boundary so its next run resumes past the shared part
too. Every base is validated before use: the metadata must match the model id and its *downloaded
revision* (so re-downloaded weights can't replay stale KV), and the snapshot's tokens must be a
strict prefix of the incoming prompt (the same guarantee as the in-memory snapshots - a template
change such as a new date simply misses and re-warms).

Alongside the bases, a small token *trace* of each configuration's last prompt is persisted after
every turn. Diffing the incoming prompt against the best trace on record is how a fresh process
learns where the stable prefix ends - to establish a base on a config's first sighting, or to
deepen a shared one. Hosts can switch the store off at runtime via
`PrefixKVStore.isEnabledOverride` (ripple exposes this as the **Prefill cache** toggle in
`/config`, persisted as `prefixKVCache` in `settings.json`); the `DEEPAGENTS_PREFIX_KV=0` env kill
switch and `DEEPAGENTS_PREFIX_KV_DIR` relocation also apply. The store keeps the 6 most recently
used snapshots (plus 16 traces) and drops corrupt or stale files silently (they are an
optimization, never a dependency).

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
Two observable properties help a UI narrate loads: `lastLoadError` records why the most recent
load returned nil, and `loadingModelID` is the id currently cold-loading from disk (nil
otherwise) - so a lazy reload can be labeled as a model load instead of reading as slow prompt
processing.

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

Reasoning content (`<think>...</think>` blocks emitted by reasoning models) is split out by the
shared `ThinkStream` and surfaced as `AgentContentBlock.reasoning` - it does not appear in the tool
arguments or the text content block.

## qwen3_5 family: Ornith and Qwen3.6

`DeepAgentsMLX` drives more than one wire format. Each catalog model declares a `codecFamily`
(`MlxModel.codecFamily`), and `RebuildTurnSession` selects the matching codec per turn - a new wire
format is a new codec, not a change to the agent loop.

The `qwen35` codec drives [Ornith-1.0-9B](https://huggingface.co/deepreinforce-ai/Ornith-1.0-9B)
(`model_type: qwen3_5`) and the [Qwen3.6](https://huggingface.co/Qwen/Qwen3.6-27B) models
(`qwen3_5` / `qwen3_5_moe`). Unlike LFM2, their `<tool_call>` blocks hold Qwen XML
(`<function=name><parameter=k>value</parameter></function>`), which mlx-swift-lm's own
`XMLFunctionParser` parses correctly - so the adapter does **not** suppress the built-in parser for
this family; it consumes the resulting `Generation.toolCall` events and only splits `<think>`
reasoning in-app (`Qwen35Decoder`).

Ornith is a unified vision-language model used as a reasoning planner: cataloged `kind: .language`
(it lists in the planner picker and drives the tool loop) with `acceptsImages: true`. The
`acceptsImages` capability - distinct from the `kind`-based `isVision` - makes `loadsAsVision`
true, so it loads through the VLM factory and can also back the vision subagent. One repo, both
roles.

The Qwen3.6 OptiQ conversions are **text-only planners** by contrast: those repos ship no
processor configs (images could not be prepared), and their sidecar weight files
(`mtp.safetensors`, `optiq_vision.safetensors`) are only tolerated by the LLM path's weight
sanitize - the VLM path fails with `Unhandled keys ["mtp"]`. They therefore load through the text
factory and appear only in the planner picker.

| Model | approx. size | Notes |
|---|---|---|
| Ornith 1.0 9B (4-bit / 8-bit) | ~5.2 / ~9.6 GB | temp 0.6 · top-p 0.95 · top-k 20; vision-capable |
| Qwen3.6 27B (OptiQ 4-bit) | ~20 GB | temp 1.0 · top-p 0.95 · top-k 20; text-only |
| Qwen3.6 35B A3B (OptiQ 4-bit, MoE) | ~24.7 GB | as 27B + presence penalty 1.5 (per card) |

The qwen3_5 family reports a 40k on-device context window (trained for 262k; memory is the real
limit). The Qwen3.6 OptiQ quants carry per-layer bit overrides, handled by mlx-swift-lm's
`perLayerQuantization`.

## gemma4 family: Gemma 4 E4B

The `gemma4` codec drives [Gemma 4 E4B](https://huggingface.co/google/gemma-4-E4B-it)
(`model_type: gemma4`, 4.5B effective / 8B raw parameters, hybrid sliding-window + global
attention). Like the qwen family, tool calls are parsed by mlx-swift-lm itself: the model emits
`<|tool_call>call:name{...}<tool_call|>` spans that the library's `GemmaFunctionParser` strips
and surfaces as `Generation.toolCall` events, so the adapter keeps the built-in parser active and
only splits the reasoning channel in-app (`Gemma4Decoder`).

Reasoning is opt-in via the chat template: the adapter passes `enable_thinking` so the template
opens the system turn with `<|think|>`, and the model then wraps its reasoning in a
`<|channel>thought ... <channel|>` block that the decoder routes to
`AgentContentBlock.reasoning`. Unlike Ornith, the template prefills no opener - the stream starts
in the answer and the model emits the full marker pair itself.

Gemma 4's chat template also reads two shapes the other families ignore: tool responses are
matched to their calls by `tool_call_id`, and recent assistant turns re-render their `reasoning` -
so the gemma encode includes both (`LFM2MessageCodec.renderMessages` extras that stay off for
the other families).

| Model | approx. size | Notes |
|---|---|---|
| Gemma 4 E4B (8-bit) | ~9 GB | temp 1.0 · top-p 0.95 · top-k 64; text-only for now (see below) |
| Gemma 4 E4B (OptiQ 4-bit) | ~7.5 GB | mixed-precision per-layer bits; text-only |

Both rows load through the **text (LLM) factory** and are planner-only for now. The OptiQ
conversion is text-only by construction: it ships no processor configs and carries an
`optiq_vision.safetensors` sidecar outside the weight index, so it goes behind the
sidecar-filtered snapshot view (the same treatment as the Qwen3.6 OptiQ quants). The 8-bit
conversion *does* ship full processor configs, but mlx-swift-lm 3.31.4's MLXVLM Gemma4 path
cannot load E-series checkpoints - its backbone builds K/V projections for every layer while its
sanitize drops them for the `num_kv_shared_layers` tail, so weight verification always fails
(upstream issue #338, fix pending in #384). Once that fix ships, the 8-bit row flips to
`acceptsImages: true` for Ornith-style dual-role duty. Both report the 40k on-device context
window (trained for 128k).

A follow-up will wire `mlx-community/gemma-4-E4B-it-assistant-bf16` - Google's 78.8M
multi-token-prediction drafter for Gemma 4 - as a speculative-decoding accelerator (mlx-swift-lm
already ships the Gemma 4 MTP machinery); it is not a standalone chat model and is not cataloged.

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
