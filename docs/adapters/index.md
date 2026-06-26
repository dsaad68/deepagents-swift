# Adapters (overview)

DeepAgents ships an inference-agnostic core. Every backend is a value type that conforms to
`ChatModel` - a stateless factory for run-scoped `ModelTurnSession` objects. The core library
(`DeepAgents`) carries zero backend dependencies; backends arrive as separate SwiftPM library
products you add only when you need them.

```swift
public protocol ChatModel: Sendable {
    var supportsVision: Bool { get }
    var modelID: String? { get }
    var contextWindowTokens: Int? { get }
    func makeSession() -> any ModelTurnSession
}
```

Because `ChatModel` is the only contract the agent loop cares about, you can swap backends - or
supply a mock - without touching any other part of your setup. See
[Architecture](../concepts/architecture.md) for how `makeSession()` fits into the ReAct loop.

## Available products

| Product | Import | What it provides | When to use |
|---|---|---|---|
| `DeepAgentsMLX` | `import DeepAgentsMLX` | On-device inference via MLX; LFM2.5 model catalog; `MlxChatModel`, `MlxModelLoader`, custom tool-call parser | Apple Silicon Mac; privacy-sensitive workloads; offline use; low-latency development |
| `DeepAgentsOpenAI` | `import DeepAgentsOpenAI` | OpenAI chat-completions; also covers OpenRouter, self-hosted OpenAI-compatible endpoints, and Azure OpenAI | Any OpenAI-compatible API; GPT-4o, o3, OpenRouter routing, Azure deployments |
| `DeepAgentsAnthropic` | `import DeepAgentsAnthropic` | Anthropic Messages API (direct) + AWS Bedrock with SigV4 signing | Claude models via api.anthropic.com or via Bedrock in AWS-hosted workloads |
| `DeepAgentsMacTools` | `import DeepAgentsMacTools` | macOS-native `AgentMiddleware` set: screenshots, clipboard, Apple Notes, mac CLI tools | Any agent that needs to interact with the macOS desktop environment; NOT a `ChatModel` |

!!! note "MacTools is middleware, not a model"
    `DeepAgentsMacTools` does not provide a `ChatModel`. It contributes tools via
    `AgentMiddleware` conformers. Pair it with any of the three model adapters. See the
    [macOS tools](mac-tools.md) page for details.

## How to pick

**On-device / offline** -- use `DeepAgentsMLX`. Requires Apple Silicon and macOS 26+. The LFM2.5
family (350 M to 8 B-A1B) covers most agentic tasks; VL variants handle vision. Models are pulled
once from Hugging Face and cached locally.

**Cloud - OpenAI or compatible** -- use `DeepAgentsOpenAI`. One struct covers OpenAI, OpenRouter
(multi-provider routing), any self-hosted vLLM or Ollama endpoint, and Azure OpenAI via
`OpenAIEndpointStyle.azure`.

**Cloud - Anthropic / Bedrock** -- use `DeepAgentsAnthropic`. `AnthropicChatModel` talks directly
to `api.anthropic.com`; `BedrockChatModel` talks to AWS Bedrock with SigV4 request signing and
cross-region inference profile IDs.

**Desktop automation** -- add `DeepAgentsMacTools` regardless of which model adapter you choose.
Pass the middleware instances in the `middleware:` array of `createAgent` or `createDeepAgent`.

## Adding a product to your package

```swift
// Package.swift
.package(url: "https://github.com/dsaad68/deepagents-swift", from: "0.2.3")

// target dependencies - add only what you need:
.product(name: "DeepAgents", package: "DeepAgents"),
.product(name: "DeepAgentsMLX", package: "DeepAgents"),
.product(name: "DeepAgentsOpenAI", package: "DeepAgents"),
.product(name: "DeepAgentsAnthropic", package: "DeepAgents"),
.product(name: "DeepAgentsMacTools", package: "DeepAgents"),
```

## Adapter pages

- [MLX (on-device)](mlx.md)
- [OpenAI & Azure](openai.md)
- [Anthropic & Bedrock](anthropic.md)
- [macOS tools](mac-tools.md)
