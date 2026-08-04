# Subagents

A **subagent** is a child `ReactAgent` the planner can delegate an isolated subtask to. Its work runs in a fresh conversation with no access to the parent's history; only its final answer comes back as the `task` tool result. This keeps the planner's context clean and lets expensive multi-step subtasks run without polluting the parent's token budget.

## How delegation works

When `createDeepAgent` is called, a `SubAgentMiddleware` is wired into the pipeline. It contributes a single `task` tool to the main agent's tool set and injects a system-prompt note listing the available subagents by name and description.

At runtime the planner calls:

```
task(description: "...", subagent_type: "general-purpose")
```

`TaskTool.execute` then:

1. Looks up the named subagent in its registry.
2. Calls `createAgent(model:tools:systemPrompt:middleware:maxIterations:)` with the subagent's settings.
3. Runs the new `ReactAgent` on a single-message input - just the delegated task description. No parent history is forwarded.
4. Returns the subagent's committed final answer as the tool result.

The parent sees only the answer. The subagent's intermediate tool calls, reasoning, and partial outputs never enter the parent's conversation.

## The `SubAgent` struct

```swift
public struct SubAgent: Sendable {
    public init(
        name: String,
        description: String,
        systemPrompt: String,
        tools: [any AgentTool]? = nil,
        model: (any ChatModel)? = nil,
        middleware: [any AgentMiddleware] = [],
        maxIterations: Int = 24
    )
}
```

| Field | Required | Meaning |
|---|---|---|
| `name` | yes | Identifier the planner passes as `subagent_type` |
| `description` | yes | Surfaced to the planner (in the tool schema and prompt) so it knows when to delegate here |
| `systemPrompt` | yes | The subagent's own system prompt - not inherited from the parent |
| `tools` | no | `nil` inherits the deep agent's base tools; `[]` gives the subagent no tools |
| `model` | no | `nil` inherits the deep agent's model; supply an explicit model to use a different backend |
| `middleware` | no | Extra middleware for this subagent (logging, rate limiting, etc.) |
| `maxIterations` | no | Hard cap on ReAct rounds; defaults to 24 |

!!! note "Subagents cannot spawn subagents"
    The `task` tool is contributed by `SubAgentMiddleware` and is intentionally NOT part of the base tool set that subagents inherit. A delegated subagent therefore has no way to spawn further subagents, which keeps the delegation graph shallow and predictable.

## The general-purpose subagent

When `includeGeneralPurpose: true` (the default) is passed to `createDeepAgent`, a built-in general-purpose subagent is prepended to the registry:

```
name:        "general-purpose"
description: "General-purpose agent for arbitrary multi-step subtasks. Use it to silo
              isolated work (research, drafting, multi-step lookups) off your main
              context. It has the same tools you do."
systemPrompt: "You are a focused subagent handling one delegated task. Do exactly what
               was asked, using your tools as needed, and finish with a single clear,
               self-contained answer the calling agent can use directly..."
tools:        nil  (inherits parent's base tools)
model:        nil  (inherits parent's model)
```

This covers the common delegation pattern without requiring any configuration. Disable it with `includeGeneralPurpose: false` if you want to control the registry entirely yourself.

## Passing custom subagents

Supply an array of `SubAgent` values to `createDeepAgent`:

```swift
import DeepAgents

let researcher = SubAgent(
    name: "researcher",
    description: "Searches the web and synthesizes a concise factual answer.",
    systemPrompt: "You are a research assistant. Use the fetch and grep tools to gather "
        + "information, then write a short, well-cited summary.",
    tools: nil,   // inherit base tools
    maxIterations: 12
)

let codeReviewer = SubAgent(
    name: "code-reviewer",
    description: "Reviews a file for correctness and style issues.",
    systemPrompt: "You are a Swift code reviewer. Read the file provided, then list "
        + "specific issues with file paths and line numbers.",
    tools: nil,
    maxIterations: 8
)

let agent = createDeepAgent(
    model: model,
    subagents: [researcher, codeReviewer],
    includeGeneralPurpose: true   // still prepended; total: [general-purpose, researcher, codeReviewer]
)
```

The planner's system prompt receives a list like:

```
## Delegating with `task`
For isolated, multi-step subtasks, call `task` with a thorough `description` and the
`subagent_type` of the right subagent below. The subagent runs on its own and returns a
single final result; it can't ask you follow-ups, so give it everything it needs.
Available subagents:
- `general-purpose`: General-purpose agent for arbitrary multi-step subtasks. ...
- `researcher`: Searches the web and synthesizes a concise factual answer.
- `code-reviewer`: Reviews a file for correctness and style issues.
```

## Inheritance rules

| Setting | `nil` means | `[]` means |
|---|---|---|
| `tools` | Inherit the deep agent's base tool set | Subagent has NO tools |
| `model` | Inherit the deep agent's model | N/A (use `nil` to inherit) |

"Base tools" are the tools that `createDeepAgent` built before `SubAgentMiddleware` was added - they include any explicit `tools:` you passed to the factory plus any catalog middleware tools, but NOT the `task` tool itself.

## Shared filesystem and approval gating

`SubAgentMiddleware` threads two things into every subagent's middleware stack automatically:

- **The shared `FilesystemBackend`** - if `createDeepAgent` was given a `backend:`, each subagent gets a `FilesystemMiddleware` for the same backend, so subagents can read and write the same working files as the parent and each other.
- **The `HumanInTheLoopMiddleware`** - if `createDeepAgent` was given an `approvalHandler:`, it is passed into every subagent too. Delegation therefore never bypasses the user's approval gate.

## `SubAgentMiddleware` and the middleware pipeline

`SubAgentMiddleware` is a structural pillar wired by `createDeepAgent` in a fixed position in the pipeline: after planning (`TodoListMiddleware`) and filesystem (`FilesystemMiddleware`), before any caller-supplied middleware. You cannot position it elsewhere; if you need subagents with `createAgent` instead, wire `SubAgentMiddleware` manually.

See [Middleware](middleware.md) for the full pipeline order and hook semantics.

## Context isolation: why it matters

Each subagent run starts with only the delegated task string (and optionally a forwarded screenshot). It never sees:

- The parent's conversation history.
- Other subagents' tool calls or results.
- The parent's system prompt.

This means expensive tool results - large file reads, long web fetches, multi-step git diffs - accumulate in the subagent's context without growing the planner's token count. When the subtask finishes, only its final answer (typically a compact paragraph) is appended to the parent's history.
