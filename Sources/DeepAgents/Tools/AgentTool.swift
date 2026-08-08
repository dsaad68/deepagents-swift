import Foundation

/// Context handed to a tool at execution time — Mispher's lightweight mirror of
/// LangChain's `ToolRuntime`. Carries a read-only snapshot of the conversation so a
/// tool can inspect state if it needs to. `Sendable` so tools run off the main actor.
public struct ToolContext: Sendable {
    /// The agent state visible to the tool (read-only).
    let state: AgentState
    /// Sink for streaming sub-events while the tool runs — e.g. the `task` tool forwards a
    /// subagent's tokens as `.toolProgress` so the UI shows them live. Defaults to a no-op, so
    /// tools that don't stream (the vast majority) can ignore it.
    let onEvent: @Sendable (AgentEvent) -> Void

    init(
        state: AgentState = .init(),
        onEvent: @escaping @Sendable (AgentEvent) -> Void = { _ in }
    ) {
        self.state = state
        self.onEvent = onEvent
    }
}

/// What a tool returns: the textual result the model sees next turn, plus an optional
/// state update (LangChain's `Command`) the runtime applies to the agent state.
public struct ToolOutput: Sendable {
    var content: String
    var stateUpdate: AgentStateUpdate?
    /// Whether this result reports a failure. A tool that returns an error message rather than
    /// throwing still failed, and a host has no way to know that from the text - so `read_file`
    /// answering `Error: no file at "…"` used to be drawn with a success tick, telling the user
    /// the call worked. The message reaches the model unchanged either way; this only decides how
    /// the run reports it (`.toolFailed` rather than `.toolCompleted`).
    var isFailure = false

    public init(_ content: String, stateUpdate: AgentStateUpdate? = nil) {
        self.content = content
        self.stateUpdate = stateUpdate
    }

    /// A recoverable failure. The text is returned to the model as the tool result exactly as
    /// given, so it can correct itself; unlike throwing, nothing wraps or re-describes it.
    public static func failure(_ content: String, stateUpdate: AgentStateUpdate? = nil) -> ToolOutput {
        var output = ToolOutput(content, stateUpdate: stateUpdate)
        output.isFailure = true
        return output
    }
}

/// A callable tool the agent can invoke — Mispher's mirror of a LangChain `BaseTool`.
/// A conformer declares a JSON-schema interface (`parameters`) and an async `execute`.
/// The generated schema is injected into the chat template by `mlx-swift-lm`.
public protocol AgentTool: Sendable {
    var name: String { get }
    var description: String { get }
    var parameters: [ToolParameter] { get }
    func execute(_ arguments: [String: AgentJSON], _ context: ToolContext) async throws -> ToolOutput

    /// Whether a call to this tool may run *at the same time* as the other parallel-safe calls
    /// the model emitted in the same round. Defaults to `false`, which is the framework's
    /// standing guarantee: a round's tools run one after another and each one sees the state and
    /// the tool results the calls before it produced.
    ///
    /// Declare `true` only when the tool both reads and writes nothing another call in the same
    /// round could care about - `read_file`, `grep`, `git_log`, `calculator`. Two things follow
    /// from the declaration, and a conformer is asserting both: the tool's own work is safe to
    /// run concurrently with itself, and it does not need to see an earlier call's result or
    /// state update (siblings in a batch all see the state as of the batch's start).
    ///
    /// The declaration is about the tool, not its wrapping: a gated tool never fans out however
    /// it answers here, because an approval card is shown one at a time. Middleware
    /// `wrapToolCall` chains do run concurrently around a parallel batch, so a middleware whose
    /// wrapper is order-sensitive should not wrap parallel-safe tools.
    var isParallelSafe: Bool { get }

    /// The `mlx-swift-lm` tool schema injected into the chat template. A protocol
    /// requirement (not just an extension method) so a conformer that already holds a
    /// server-provided JSON Schema — e.g. ``MCPTool`` — can override it to inject that
    /// schema verbatim through the `any AgentTool` existential. Most tools rely on the
    /// default below, which builds the schema from `parameters`.
    func toolSchema() -> ToolSchema
}

extension AgentTool {
    public var parameters: [ToolParameter] { [] }

    /// Serial by default - see the requirement. A tool opts into fan-out explicitly, so a tool
    /// written before parallel dispatch existed (or an MCP server's, whose semantics we can't
    /// know) keeps the old guarantee.
    public var isParallelSafe: Bool { false }

    /// Build the `mlx-swift-lm` tool schema (`ToolSchema`) injected into the chat
    /// template — same shape as `MLXLMCommon.Tool`'s generated schema.
    public func toolSchema() -> ToolSchema {
        var properties: [String: any Sendable] = [:]
        var required: [String] = []
        for parameter in parameters {
            properties[parameter.name] = parameter.schema
            if parameter.isRequired { required.append(parameter.name) }
        }
        return [
            "type": "function",
            "function": [
                "name": name,
                "description": description,
                "parameters": [
                    "type": "object",
                    "properties": properties,
                    "required": required
                ] as [String: any Sendable]
            ] as [String: any Sendable]
        ]
    }
}
