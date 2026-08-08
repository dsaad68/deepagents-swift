import Foundation

/// A run-scoped model session — Mispher's single-shot model node. Mirrors the role the
/// `model` node plays in LangChain's agent graph: given the conversation so far, produce
/// **one** assistant turn (visible text and/or tool calls) and stop. It does **not**
/// dispatch tools — the agent (`ReactAgent`) owns the ReAct loop and dispatches tools
/// between turns.
///
/// A session is created once per `ReactAgent.run` (via `ChatModel.makeSession`) and then
/// `nextTurn` is called once per ReAct round, with the **full** conversation each round.
/// The model node is stateless: the prompt is a pure function of `messages`, rebuilt and
/// generated from a fresh cache every round (`RebuildTurnSession`). That keeps the engine
/// honest — what the model sees is exactly what we pass — and lets middleware rewrite
/// history (trim/summarise) or a `wrapModelCall` retry re-invoke the handler, neither of
/// which a live KV cache could honor.
public protocol ModelTurnSession: AnyObject {
    /// Generate exactly one assistant turn from the whole conversation so far.
    /// - Parameters:
    ///   - messages: the full conversation to condition on (prior turns + this run's input
    ///     + any tool results from earlier rounds). The system prompt is supplied
    ///     separately; do not include it here.
    ///   - systemPrompt: the (possibly middleware-composed) system prompt for this round.
    ///   - tools: the tools available this round (already middleware-filtered).
    ///   - onChunk: receives streamed pieces of this round - visible answer `text` and, on a
    ///     reasoning model, `reasoning` (chain-of-thought) on its own channel.
    /// - Returns: an `.ai(text, toolCalls:)` message. Stops at the tool calls of one pass.
    func nextTurn(
        messages: [AgentMessage],
        systemPrompt: String?,
        tools: [any AgentTool],
        onChunk: @escaping @Sendable (AgentStreamChunk) -> Void
    ) async throws -> AgentMessage

    /// As above, but the turn begins in the **answer** channel rather than the reasoning one.
    ///
    /// Reasoning models routinely have their chat template open a `<think>` block in the generation
    /// prompt, so generation starts inside reasoning and only a closing `</think>` moves output into
    /// the visible answer. A model that never emits that tag produces a turn whose text is empty no
    /// matter what it wrote - the answer had nowhere to go. Asking such a model to "answer in plain
    /// text" cannot work, because the channel it would answer on is not open.
    ///
    /// `startingOutsideReasoning` closes the block up front, so the very first token is answer text.
    /// Use it where an answer is required and reasoning is not - ``ReactAgent``'s forced final turn.
    /// The default implementation ignores it: a session whose model has no reasoning channel (or
    /// whose template opens none) already starts in the answer.
    func nextTurn(
        messages: [AgentMessage],
        systemPrompt: String?,
        tools: [any AgentTool],
        startingOutsideReasoning: Bool,
        onChunk: @escaping @Sendable (AgentStreamChunk) -> Void
    ) async throws -> AgentMessage
}

public extension ModelTurnSession {
    func nextTurn(
        messages: [AgentMessage],
        systemPrompt: String?,
        tools: [any AgentTool],
        startingOutsideReasoning _: Bool,
        onChunk: @escaping @Sendable (AgentStreamChunk) -> Void
    ) async throws -> AgentMessage {
        try await nextTurn(
            messages: messages, systemPrompt: systemPrompt, tools: tools, onChunk: onChunk
        )
    }
}
