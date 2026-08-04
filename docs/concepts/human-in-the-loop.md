# Human in the loop

DeepAgents provides two complementary mechanisms for keeping a human in the agent's decision loop:

- **`HumanInTheLoopMiddleware`** - gates named tools behind explicit user approval before they run.
- **`AskUserMiddleware`** - gives the model an `ask_user` tool to pose clarifying questions mid-run and wait for answers before proceeding.

Both are wired by `createDeepAgent` when you supply the relevant handler, and both suspend the in-process agent run on an `async` callback until the user responds.

---

## `HumanInTheLoopMiddleware` - approval gating

### Overview

`HumanInTheLoopMiddleware` intercepts tool calls via `wrapToolCall`. When a tool is listed in `interruptOn`, the middleware:

1. Builds a `ToolApprovalRequest` and calls your `approvalHandler`.
2. Suspends until the handler returns a `ToolApprovalDecision`.
3. Proceeds, edits the call's arguments, rejects the call (feeding an error back to the model), or substitutes a synthetic result - depending on the decision.

Tools NOT listed in `interruptOn` pass through without interruption.

The middleware also appends a note to the system prompt on every model call that lists the gated tool names and instructs the model to adjust its approach if a call is rejected rather than retrying with the same arguments.

### `InterruptOnConfig`

```swift
public struct InterruptOnConfig: Sendable, Equatable {
    public init(
        allowedDecisions: [ToolDecisionType] = [.approve, .reject],
        description: String? = nil
    )
}
```

| Field | Meaning |
|---|---|
| `allowedDecisions` | Which `ToolDecisionType` values the handler may return for this tool |
| `description` | Fixed description shown to the user; if `nil`, one is generated from the tool name and arguments |

Pass a `[String: InterruptOnConfig]` dictionary to gate specific tools:

```swift
let interruptOn: [String: InterruptOnConfig] = [
    "write_file": InterruptOnConfig(),               // approve or reject
    "shell":      InterruptOnConfig(
        allowedDecisions: [.approve, .edit, .reject],
        description: "Run a shell command - review the command before approving."
    )
]
```

### `ToolDecisionType`

```swift
public enum ToolDecisionType: String, Sendable, CaseIterable {
    case approve, edit, reject, respond
}
```

| Decision | Effect |
|---|---|
| `approve` | Run the tool call as the model issued it |
| `edit` | Run the tool call with the handler's replacement arguments |
| `reject` | Do not run the call; feed a rejection error back to the model |
| `respond` | Do not run the call; feed the handler's string back as the tool result |

### `ToolApprovalRequest`

The struct surfaced to your handler:

```swift
public struct ToolApprovalRequest: Sendable, Identifiable {
    public let id: UUID                          // == AgentToolCall.id
    public let toolName: String
    public let arguments: [String: AgentJSON]
    public let description: String               // human-readable summary for the UI
    public let allowedDecisions: [ToolDecisionType]
    public var argumentRows: [ArgumentRow]       // key-sorted display rows
}
```

### `ToolApprovalHandler`

```swift
public typealias ToolApprovalHandler = @Sendable (ToolApprovalRequest) async -> ToolApprovalDecision
```

The handler is called from inside the agent's async task. A typical host publishes the request to drive a UI sheet, suspends on a `CheckedContinuation`, and resumes it from the approve/reject buttons:

```swift
let approvalHandler: ToolApprovalHandler = { request in
    await withCheckedContinuation { continuation in
        DispatchQueue.main.async {
            self.pendingApproval = PendingApproval(request: request, continuation: continuation)
        }
    }
}
```

### `ToolApprovalDecision`

```swift
public enum ToolApprovalDecision: Sendable {
    case approve
    case edit(arguments: [String: AgentJSON])
    case reject(message: String?)
    case respond(message: String)
}
```

### Wiring via `createDeepAgent`

When `approvalHandler:` is supplied, `createDeepAgent` automatically gates ALL tools by passing the handler into `HumanInTheLoopMiddleware`, which it wires as the outermost middleware (last in the chain). It also threads the same middleware into every subagent so that delegation never bypasses the gate.

```swift
let agent = createDeepAgent(
    model: model,
    interruptOn: [
        "write_file": InterruptOnConfig(),
        "shell": InterruptOnConfig(allowedDecisions: [.approve, .edit, .reject])
    ],
    approvalHandler: { request in
        // present request.description and request.argumentRows to the user
        // return .approve, .reject(message:), .edit(arguments:), or .respond(message:)
        await myApprovalUI.ask(request)
    }
)
```

!!! tip "Per-tool policy"
    `interruptOn` is a dictionary - you can configure different `allowedDecisions` and `description` strings for each tool. Tools omitted from the dictionary run without interruption even when an `approvalHandler` is set.

See the [Gate tools with approvals](../guides/approvals.md) guide for a complete working example.

---

## `AskUserMiddleware` - model-initiated questions

### Overview

`AskUserMiddleware` gives the model an `ask_user` tool and injects guidance on when to use it. When the model calls `ask_user`, the tool:

1. Parses and validates the questions array.
2. Calls your `AskUserHandler` with an `AskUserRequest`.
3. Suspends until the handler returns an `AskUserResponse`.
4. Formats the answers as a `Q: ... / A: ...` string and returns it as the tool result.

The model then reads the answers on the next round and continues accordingly. The mechanism mirrors `HumanInTheLoopMiddleware`: an in-process `async` suspension while the host presents the UI, rather than a LangGraph checkpoint-and-resume.

`AskUserMiddleware` is registered on the main agent only - not on delegated subagents. A subagent asking the user mid-subtask would be confusing and breaks the isolation model.

### Question types

| Type | Meaning |
|---|---|
| `text` | Free-form text answer |
| `multiple_choice` | User picks exactly one option from `choices`; an "Other" free-text escape is always offered |
| `multi_select` | User picks one or more options; the result is the chosen values joined by `", "` |

### `AskUserRequest` and `AskUserResponse`

```swift
public struct AskUserRequest: Sendable, Identifiable {
    public let id: UUID
    public let questions: [AskUserQuestion]   // one or more questions
}

public enum AskUserResponse: Sendable {
    case answered([String])   // one string per question, in order
    case cancelled            // user dismissed without answering
    case error(String)        // host could not present the interaction
}
```

Each `AskUserQuestion` carries:

```swift
public struct AskUserQuestion: Sendable, Identifiable {
    public let question: String
    public let type: AskUserQuestionType      // .text / .multipleChoice / .multiSelect
    public let choices: [AskUserChoice]       // options for pick questions
    public let required: Bool                 // default true
}
```

### `AskUserHandler`

```swift
public typealias AskUserHandler = @Sendable (AskUserRequest) async -> AskUserResponse
```

The pattern mirrors `ToolApprovalHandler`: publish the request, suspend on a continuation, resume it when the user submits:

```swift
let askUserHandler: AskUserHandler = { request in
    await withCheckedContinuation { continuation in
        DispatchQueue.main.async {
            self.pendingQuestion = PendingQuestion(request: request, continuation: continuation)
        }
    }
}
```

### Wiring via `createDeepAgent`

```swift
let agent = createDeepAgent(
    model: model,
    askUserHandler: { request in
        await myQuestionUI.present(request)
    }
)
```

When `askUserHandler:` is supplied, `createDeepAgent` wires `AskUserMiddleware` just before `HumanInTheLoopMiddleware` (if present). No `interruptOn` entry is needed for `ask_user` - the tool is always allowed.

### The `ask_user` guidance injected into the system prompt

The middleware appends this guidance to the system prompt on every model call:

```
## `ask_user`

You have access to the `ask_user` tool to ask the user questions when you need
clarification or input. Use this tool sparingly - only when you genuinely need
information from the user that you cannot determine from context.

When using `ask_user`:
- Be concise and specific with your questions
- Use multiple choice when there are clear options to choose from
- Group related questions into a single ask_user call rather than making multiple calls
- Never ask questions you can answer yourself from the available context
```

---

## Interaction between the two mechanisms

The two middlewares are independent but compose cleanly. A run can have both:

- `ask_user` fires when the model needs clarification before it can even choose which tool to call.
- `HumanInTheLoopMiddleware` fires when the model has already decided on a tool call and the user needs to approve or edit it.

Both suspend the same agent run. Because the loop is in-process, only one approval or question prompt can be active at a time per run.

See [Tools & policy](tools.md) for how tool schemas are rendered and filtered before reaching the model.
