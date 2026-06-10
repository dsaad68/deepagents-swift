## Linting & Formatting

After every code change, run SwiftFormat then SwiftLint from the repo root and resolve anything they flag before considering the task done:

```sh
swiftformat .          # apply formatting (or `swiftformat --lint .` to check only)
swiftlint lint --strict
```

Configs live at `.swiftformat` and `.swiftlint.yml`, and are reconciled so the two tools don't fight (e.g. no trailing commas, braces on the same line). The SwiftFormat pass is deliberately conservative — it preserves the codebase's concise one-liner style and leaves signatures/concurrency annotations alone. For SwiftLint, use `swiftlint --fix` for the mechanical rules, then fix the rest by hand. The tree should stay clean under `--strict` (warnings treated as errors).


## Package And Library Research

Use the DeepWiki MCP to look up package and library details directly from their source repositories.
If the available documentation is unclear or incomplete, ask DeepWiki follow-up questions before making assumptions.

- https://github.com/ml-explore/mlx-swift
- https://github.com/ml-explore/mlx-swift-examples
- https://github.com/ml-explore/mlx-swift-lm
- https://github.com/langchain-ai/deepagents
- https://github.com/langchain-ai/langchain
- https://github.com/langchain-ai/langgraph


## Documentation

Start with these documents:

- https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxlmcommon
- https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxlmcommon/using
- https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxllm
- https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxvlm


# Other
- Avoid em dashes (—) in UI strings; prefer " - " or "--" instead.