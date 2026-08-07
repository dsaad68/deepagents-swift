# Write a custom tool

A tool is the primary way an agent takes action in the world. Any `AgentTool` conformer - your own or a built-in - goes through exactly the same dispatch path: the model names it in a tool call, the framework extracts the arguments, calls `execute`, and appends the result as a `.tool` message.

This guide shows how to build one from scratch.

---

## The `AgentTool` protocol

```swift
public protocol AgentTool: Sendable {
    var name: String { get }
    var description: String { get }
    var parameters: [ToolParameter] { get }
    func execute(_ arguments: [String: AgentJSON], _ context: ToolContext) async throws -> ToolOutput
    func toolSchema() -> ToolSchema
    var isParallelSafe: Bool { get }
}
```

Four things to implement:

| Member | Purpose |
|---|---|
| `name` | Snake-case identifier the model uses to call the tool (e.g. `"calculator"`) |
| `description` | Natural language description placed in the model's prompt - make it precise |
| `parameters` | Array of `ToolParameter` describing the JSON schema the model must pass |
| `execute` | The async implementation; receives parsed args and a context handle |

`toolSchema()` is derived automatically for most cases - the default implementation serializes `name`, `description`, and `parameters` into the JSON Schema format the framework passes to the model. You only need to override it for unusual schema shapes.

`isParallelSafe` defaults to `false`, which keeps the tool in the round's serial order. Override it to `true` only for a tool that neither writes anything nor needs to see an earlier call's result - see [Parallel-safe tools](#parallel-safe-tools) below.

---

## Declaring parameters

`ToolParameter` is built with two factory methods:

```swift
public static func required(
    _ name: String,
    type: ToolParameterType,
    description: String
) -> ToolParameter

public static func optional(
    _ name: String,
    type: ToolParameterType,
    description: String
) -> ToolParameter
```

`ToolParameterType` covers the full range of JSON-representable types:

```swift
public enum ToolParameterType: Sendable {
    case string
    case bool
    case int
    case double
    case array(elementType: ToolParameterType)
    case object(properties: [ToolParameter])
    case data
}
```

Example parameter declarations:

```swift
let parameters: [ToolParameter] = [
    .required("expression", type: .string, description: "The arithmetic expression to evaluate, e.g. \"3 * (4 + 2)\""),
    .optional("precision", type: .int, description: "Decimal places to round the result to (default 2)"),
]
```

---

## Reading arguments from `AgentJSON`

The `arguments` dictionary is `[String: AgentJSON]`. Extract values with a switch or a convenience helper:

```swift
guard case .string(let expr) = arguments["expression"] else {
    throw ToolError.missingArgument("expression")
}

let precision: Int
if case .int(let p) = arguments["precision"] {
    precision = p
} else {
    precision = 2
}
```

`AgentJSON` is exhaustive:

| Case | Swift type |
|---|---|
| `.null` | absence / JSON null |
| `.bool(Bool)` | boolean |
| `.int(Int)` | integer |
| `.double(Double)` | floating point |
| `.string(String)` | text |
| `.array([AgentJSON])` | ordered list |
| `.object([String: AgentJSON])` | nested object |

Always guard defensively - the model might pass the wrong type or omit an optional key.

---

## Returning a `ToolOutput`

`ToolOutput` carries a result string the agent appends as a `.tool` message, plus an optional state update for the agent's shared state store. For most tools, returning plain text is sufficient:

```swift
return ToolOutput(text: "Result: 42.00")
```

The text becomes the content of the tool-result message. Keep it informative - the model reads it in the next round to decide what to do next.

### Make the result say what it is

The single most common cause of a small model looping is a result that doesn't read like an answer. The `tool` role already marks the turn as a result, but it says nothing about *what* the result is, and a bare fragment gets read as noise:

| Bad | Why it loops | Good |
|---|---|---|
| `## main` | Reads like a markdown heading, not a status | `On branch main. The working tree is clean - no files added, changed, or deleted.` |
| `99` | An unlabeled number; the model has to remember which call it answers | `(12 * 8) + 3 = 99` |
| `(no output)` | Did it work, or do nothing? | `The command finished successfully and printed no output.` |
| `` (empty) | Indistinguishable from a broken tool | `No matches for /foo/ under "src".` |

This is not hypothetical. A 2.6B planner asked for `git_status` on a clean tree re-called it in five consecutive rounds, because 22 characters of branch line never read as "here is the status". `read_clipboard` carries a comment recording the same failure from an earlier round of this.

The rules of thumb:

- **Name the thing in the result**, especially when the value is a short scalar. Echo the input where it disambiguates ("`<expression>` = `<value>`").
- **An empty result is a finding, so state it** - "no staged changes", "no matches", "the file is empty" - never an empty string or a bare `(none)`.
- **Don't wrap large text in JSON.** Escaping turns every newline into `\n` and inflates a file read by ~6%, which makes it harder for a small model to read, not easier. Reserve structure for small values, the way `read_clipboard` returns `{"clipboard_text": …}` with a `note` when it's empty. Errors are the framework's own `{"error": …}` shape, applied for you.

---

## Worked example: expression calculator

```swift
import DeepAgents
import Foundation

struct CalculatorTool: AgentTool {

    var name: String { "calculator" }

    var description: String {
        "Evaluates a basic arithmetic expression and returns the numeric result. " +
        "Supports +, -, *, / and parentheses. Example: \"3 * (4 + 2)\"."
    }

    var parameters: [ToolParameter] {
        [
            .required(
                "expression",
                type: .string,
                description: "The arithmetic expression to evaluate."
            ),
            .optional(
                "precision",
                type: .int,
                description: "Number of decimal places in the result. Defaults to 2."
            ),
        ]
    }

    func execute(
        _ arguments: [String: AgentJSON],
        _ context: ToolContext
    ) async throws -> ToolOutput {
        guard case .string(let expression) = arguments["expression"] else {
            return ToolOutput(text: "Error: missing required argument 'expression'.")
        }

        let precision: Int
        if case .int(let p) = arguments["precision"] {
            precision = max(0, min(p, 10))
        } else {
            precision = 2
        }

        // NSExpression handles basic arithmetic safely
        let expr = NSExpression(format: expression)
        guard let result = expr.expressionValue(with: nil, context: nil) as? Double else {
            return ToolOutput(text: "Error: could not evaluate expression '\(expression)'.")
        }

        let formatted = String(format: "%.\(precision)f", result)
        return ToolOutput(text: "\(expression) = \(formatted)")
    }
}
```

---

## Registering the tool

Pass it to either factory in the `tools:` array:

```swift
let agent = createAgent(
    model: model,
    tools: [CalculatorTool()],
    systemPrompt: "You are a helpful math assistant."
)
```

Or pass it via a middleware's `tools` property (see the guide below).

---

## Tools with side effects and async I/O

`execute` is `async throws`, so network requests, file I/O, and other async work are fine. Throw on unrecoverable errors - the framework catches thrown errors, formats them as tool-result messages, and lets the model decide how to proceed.

```swift
func execute(_ arguments: [String: AgentJSON], _ context: ToolContext) async throws -> ToolOutput {
    guard case .string(let url) = arguments["url"] else {
        return ToolOutput(text: "Error: missing 'url' argument.")
    }
    let (data, _) = try await URLSession.shared.data(from: URL(string: url)!)
    let body = String(data: data, encoding: .utf8) ?? "(binary)"
    return ToolOutput(text: body)
}
```

---

## Parallel-safe tools

If your tool only reads - a lookup, a search, a GET - declare it parallel-safe so several calls to it in one round run at once:

```swift
struct StockQuoteTool: AgentTool {
    var name: String { "stock_quote" }
    var description: String { "Look up the current price of a ticker symbol." }
    var parameters: [ToolParameter] {
        [.required("symbol", type: .string, description: "Ticker symbol, e.g. AAPL")]
    }

    /// A quote lookup writes nothing and needs nothing from the round's other calls.
    var isParallelSafe: Bool { true }

    func execute(_ arguments: [String: AgentJSON], _ context: ToolContext) async throws -> ToolOutput {
        // …
    }
}
```

Leave the default in place if any of these is true:

- The tool writes - to disk, to state, to a remote service, to the clipboard.
- The tool needs to see what an earlier call in the same round did. Calls in a concurrent batch all receive the same `context.state`, taken before the batch ran.
- Two simultaneous invocations would contend - a lock, a working directory, a single-user device or app.

You do not need to handle the approval case: a gated tool still fans out, and `HumanInTheLoopMiddleware` queues the approval requests so the user is only ever shown one at a time.

---

## Related pages

- [Tools & policy](../concepts/tools.md) - How tools are rendered, dispatched, and gated by policy
- [Write custom middleware](custom-middleware.md) - Bundle tools with lifecycle hooks in one middleware conformer
