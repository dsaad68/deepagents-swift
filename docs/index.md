<div class="hero" markdown>

# DeepAgents

<p class="tagline">An inference-agnostic agent framework for Swift.</p>

[Get started](getting-started/installation.md){ .md-button .md-button--primary }
[View on GitHub](https://github.com/dsaad68/deepagents-swift){ .md-button }

</div>

<div class="grid cards" markdown>

- :material-swap-horizontal: __Inference-agnostic ChatModel__

    A single `ChatModel` protocol is the only seam between your agent and the backend. Swap between on-device MLX, Anthropic, OpenAI, Azure, Bedrock, or your own model without touching agent logic.

- :material-layers-triple: __Composable middleware__

    Every capability - tools, prompt rewriting, approval gating, summarization - is a middleware. Hook order is well-defined, nesting is deterministic, and you can add your own with one protocol conformance.

- :material-robot: __Deep-agent pillars__

    `createDeepAgent` wires three structural pillars in a fixed order: planning (TodoList), filesystem, and subagent delegation. You get a capable planning agent by default; each pillar is independently removable.

- :material-hammer-wrench: __Built-in toolsets__

    Twelve ready-made middleware toolsets ship in the catalog: filesystem, web fetch, grep/glob, git, shell (gated), clipboard, Apple Notes, macOS utilities, and more. Add them to any agent with a single string id.

- :material-connection: __MCP client__

    `MultiServerMCPClient` connects to any number of stdio or HTTP MCP servers concurrently. Per-server failures are isolated. Tools are namespaced `server__tool` and flow through the same middleware pipeline as every other tool.

- :material-chip: __Backend adapters__

    Three adapters ship: **DeepAgentsMLX** for on-device LFM2.5 inference via Apple MLX; **DeepAgentsOpenAI** for OpenAI, OpenRouter, Azure, and any OpenAI-compatible endpoint; **DeepAgentsAnthropic** for the Anthropic Messages API and AWS Bedrock with SigV4. The core framework carries zero backend dependencies.

</div>

## What is DeepAgents?

DeepAgents is a Swift 6 framework for building autonomous AI agents. It implements a ReAct (Reason + Act) loop over any language model backend, letting you compose capabilities from a library of built-in middleware toolsets or build your own.

It is heavily inspired by [LangChain's Deep Agents](https://github.com/langchain-ai/deepagents) framework, reimagined for Swift with first-class support for on-device inference.

The core design principle is strict backend independence. The `ChatModel` protocol is the only thing an agent knows about the model. This means you can build and test your agent against a lightweight in-memory stub, then point it at an on-device MLX model, an Anthropic endpoint, or an OpenAI-compatible server without changing a single line of agent code.

All public types are `Sendable` and designed to run off the main actor. The core framework target imports only Foundation and the MCP SDK - no MLX, no AppKit, no UI dependencies - making it safe to include in CI and server environments where those frameworks are unavailable.

## The 5 library products

| Product | Purpose | Notable dependencies |
|---|---|---|
| `DeepAgents` | ReAct loop, middleware, MCP client, toolsets (core) | Foundation, swift-sdk (MCP) |
| `DeepAgentsMLX` | On-device inference via Apple MLX; LFM2 template + parser | MLX, Tokenizers |
| `DeepAgentsOpenAI` | OpenAI-compatible chat-completions (OpenAI, Azure, OpenRouter, self-hosted) | URLSession |
| `DeepAgentsAnthropic` | Anthropic Messages API + AWS Bedrock (SigV4) | CryptoKit |
| `DeepAgentsMacTools` | macOS tools: screenshot, clipboard, Apple Notes, mac CLI, container shell | AppKit, ScreenCaptureKit |

## Quick example

The snippet below creates a deep agent backed by Anthropic and runs a single turn. The agent has planning, filesystem (in-memory), and general-purpose toolsets enabled by default.

```swift
import DeepAgents
import DeepAgentsAnthropic

let model = AnthropicChatModel(
    baseURL: URL(string: "https://api.anthropic.com")!,
    model: "claude-opus-4-8",
    apiKey: ProcessInfo.processInfo.environment["ANTHROPIC_API_KEY"],
    supportsVision: false
)

let agent = createDeepAgent(
    model: model,
    systemPrompt: "You are a helpful assistant.",
    backend: StateBackend(),          // in-memory filesystem
    includeFilesystem: true,
    includeGeneralPurpose: true,
    maxIterations: 24
)

let ok = await agent.run([.human("What is 2 + 2?")], threadId: "user-123") { event in
    // bind UI / logging to streamed events
}
```

## Install

Add the package to your `Package.swift`:

```swift
.package(url: "https://github.com/dsaad68/deepagents-swift", from: "0.1.0")
```

Then add the products you need to your target's `dependencies`. See [Installation](getting-started/installation.md) for the full setup guide.

## Next steps

- [Installation](getting-started/installation.md) - Add DeepAgents to your project and understand which products to include.
- [Quickstart](getting-started/quickstart.md) - Build and run your first agent in minutes.
- [Architecture](concepts/architecture.md) - Understand the inference-agnostic design, the ReAct loop, and how middleware fits together.
