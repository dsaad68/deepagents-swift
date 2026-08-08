import Foundation

/// A model invocation request flowing through the middleware chain — LangChain's
/// `ModelRequest`. Middleware produce a modified copy via `override(...)` before the
/// model runs (e.g. to append to the system prompt or filter the tool set).
public struct ModelRequest: Sendable {
    public var messages: [AgentMessage]
    public var systemPrompt: String?
    public var tools: [any AgentTool]

    /// Return a copy with selected fields replaced. Pass `.some(nil)` for
    /// `systemPrompt` to clear it; omit an argument to leave a field unchanged.
    public func override(
        messages: [AgentMessage]? = nil,
        systemPrompt: String?? = nil,
        tools: [any AgentTool]? = nil
    ) -> ModelRequest {
        ModelRequest(
            messages: messages ?? self.messages,
            systemPrompt: systemPrompt ?? self.systemPrompt,
            tools: tools ?? self.tools
        )
    }
}

/// The model's response for one agent turn.
public struct ModelResponse: Sendable {
    var message: AgentMessage
}

/// A single tool invocation flowing through the `wrapToolCall` chain.
public struct ToolCallRequest: Sendable {
    var call: AgentToolCall
    var state: AgentState
}

/// Agent middleware — Mispher's port of LangChain's `AgentMiddleware`. Override only
/// the hooks you need; all default to no-ops. `tools` contributes extra tools to the
/// agent. The `wrap*` hooks nest, with the first-registered middleware outermost.
public protocol AgentMiddleware: Sendable {
    /// A short identifier, for diagnostics.
    var name: String { get }
    /// Tools this middleware contributes to the agent.
    var tools: [any AgentTool] { get }

    /// Runs once before the loop begins.
    func beforeAgent(_ state: inout AgentState) async
    /// Runs before the model is called.
    func beforeModel(_ state: inout AgentState) async
    /// Runs after the model is called.
    func afterModel(_ state: inout AgentState) async
    /// Runs once after the loop completes.
    func afterAgent(_ state: inout AgentState) async

    /// Wrap the model call — call `handler` to proceed (optionally with a modified
    /// request), or short-circuit by returning a response without calling it.
    func wrapModelCall(
        _ request: ModelRequest,
        _ handler: (ModelRequest) async throws -> ModelResponse
    ) async throws -> ModelResponse

    /// Wrap a single tool call — for retries, monitoring, or result rewriting.
    func wrapToolCall(
        _ request: ToolCallRequest,
        _ handler: (ToolCallRequest) async throws -> AgentMessage
    ) async throws -> AgentMessage
}

extension AgentMiddleware {
    public var name: String { String(describing: Self.self) }
    public var tools: [any AgentTool] { [] }

    /// Whether any of this middleware's tools will actually be rendered into this round's prompt.
    ///
    /// **Gate prompt guidance on this.** A capability middleware normally appends a section naming its
    /// tools and telling the model how to use them, but ``ToolSearchMiddleware`` withholds the schemas
    /// of auxiliary tools - and prose insisting the model can create an Apple Note, when no
    /// `create_note` schema was rendered, is worse than silence: it asserts a capability and leaves no
    /// way to reach it, so the model either claims it did something it could not, or reaches for an
    /// unrelated tool. (The guidance for an auxiliary tool arrives with its signature, in the
    /// `search_tools` result.)
    ///
    /// Middleware that contribute no tools keep their guidance unconditionally.
    public func contributesRenderedTools(to request: ModelRequest) -> Bool {
        let mine = tools
        guard !mine.isEmpty else { return true }
        let rendered = Set(request.tools.map(\.name))
        return mine.contains { rendered.contains($0.name) }
    }

    public func beforeAgent(_ state: inout AgentState) async {}
    public func beforeModel(_ state: inout AgentState) async {}
    public func afterModel(_ state: inout AgentState) async {}
    public func afterAgent(_ state: inout AgentState) async {}

    public func wrapModelCall(
        _ request: ModelRequest,
        _ handler: (ModelRequest) async throws -> ModelResponse
    ) async throws -> ModelResponse {
        try await handler(request)
    }

    public func wrapToolCall(
        _ request: ToolCallRequest,
        _ handler: (ToolCallRequest) async throws -> AgentMessage
    ) async throws -> AgentMessage {
        try await handler(request)
    }
}
