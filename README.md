<p align="center">
  <img src="docs/assets/images/deepagents-logo.png" width="180" alt="DeepAgents">
</p>

<h1 align="center">DeepAgents</h1>

<p align="center">
  <a href="https://deepagents-swift.verybad.engineer"><img src="https://img.shields.io/badge/docs-deepagents--swift.verybad.engineer-0ea5e9?style=flat-square" alt="Docs"></a>
  <img src="https://img.shields.io/badge/Swift-6.1%2B-F05138?style=flat-square&logo=swift&logoColor=white" alt="Swift 6.1+">
  <img src="https://img.shields.io/badge/macOS-26%2B-000000?style=flat-square&logo=apple&logoColor=white" alt="macOS 26+">
  <img src="https://img.shields.io/badge/SPM-compatible-34d399?style=flat-square" alt="Swift Package Manager">
  <img src="https://img.shields.io/badge/License-MIT-22c55e?style=flat-square" alt="MIT">
</p>

<p align="center">
  An inference-agnostic agent framework for Swift: a ReAct loop, a composable middleware
  system, an MCP client, and generic toolsets, with adapters for on-device MLX, OpenAI,
  Anthropic, and macOS.
</p>

---

## Products

| Product | What it is | Dependencies |
|---------|------------|-------------|
| `DeepAgents` | ReAct loop, middleware, MCP client, toolsets | Foundation + MCP only |
| `DeepAgentsMLX` | On-device inference via MLX (LFM2 parser/template) | MLX, Tokenizers |
| `DeepAgentsOpenAI` | OpenAI-compatible chat-completions (also Azure) | URLSession only |
| `DeepAgentsAnthropic` | Anthropic Messages + AWS Bedrock (SigV4) | CryptoKit only |
| `DeepAgentsMacTools` | macOS tools: screenshot, clipboard, Apple Notes, CLI | AppKit, ScreenCaptureKit |

The core `DeepAgents` target carries no MLX and no AppKit dependency. A CI guard enforces this
so the framework can be retargeted to any backend by writing a single `ChatModel` conformance.

## Requirements

- macOS 26+
- Swift 6.1+ (Swift 6 language mode)

## Install

```swift
.package(url: "https://github.com/dsaad68/deepagents-swift.git", from: "0.2.3")
```

Then depend on the products you need:

```swift
.product(name: "DeepAgents", package: "deepagents-swift"),
.product(name: "DeepAgentsAnthropic", package: "deepagents-swift")
```

## Docs

Full documentation at [deepagents-swift.verybad.engineer](https://deepagents-swift.verybad.engineer).

## License

MIT. See `LICENSE`.
