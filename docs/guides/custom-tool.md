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

## Related pages

- [Tools & policy](../concepts/tools.md) - How tools are rendered, dispatched, and gated by policy
- [Write custom middleware](custom-middleware.md) - Bundle tools with lifecycle hooks in one middleware conformer
