import DeepAgents
import Foundation
import MLXLMCommon

// Ornith (qwen3_5) is a reasoning model: each assistant turn opens with a `<think>…</think>` block,
// then the answer, and it emits tool calls as Qwen XML
// (`<tool_call><function=name><parameter=k>value</parameter></function></tool_call>`). Unlike LFM2 -
// whose Pythonic `<|tool_call_start|>` calls we must parse ourselves to dodge mlx-swift-lm's
// comma-truncating `PythonicToolCallParser` - the XML format is handled correctly by the library's
// `XMLFunctionParser` (`ToolCallFormat.xmlFunction`, auto-inferred for `qwen3_5`). So here we do NOT
// suppress the built-in parser: ``RebuildTurnSession`` leaves the inferred `.xmlFunction` active, the
// `<tool_call>` spans are stripped from the visible chunks and surfaced as `Generation.toolCall`
// events, and this codec's decoder only splits `<think>` reasoning from the answer and collects the
// already-parsed calls.

/// The qwen3_5 / Ornith message codec. Encoding reuses ``LFM2MessageCodec/renderMessages`` verbatim:
/// Ornith's shipped chat template renders an assistant turn's `tool_calls` from the same
/// `{"function": {"name", "arguments"}}` shape, tool results from `tool`-role turns, and image turns
/// from the same structured `[{"type":"text"}, {"type":"image"}]` content - so only decoding differs.
public struct Qwen35MessageCodec: MessageCodec {
    public init() {}

    public func encode(
        _ history: [AgentMessage], systemPrompt: String?, tools: [any AgentTool], supportsVision: Bool
    ) -> LFM2Request {
        let (messages, imageURLs) = LFM2MessageCodec.renderMessages(
            systemPrompt: systemPrompt, messages: history, supportsVision: supportsVision
        )
        return LFM2Request(messages: messages, imageURLs: imageURLs, toolSpecs: tools.map { $0.toolSchema() })
    }

    public func makeDecoder() -> any TurnDecoder<String> { Qwen35Decoder() }
}

/// Reassembles the qwen3_5 token stream into one canonical assistant turn. The `<tool_call>` spans are
/// already removed from the chunk stream by the library's `XMLFunctionParser` and delivered separately
/// via ``ingestToolCall(_:)``, so this decoder only splits each visible chunk into the answer and
/// `<think>…</think>` reasoning (``ThinkStream``) so reasoning streams on its own channel.
public final class Qwen35Decoder: TurnDecoder, ToolCallIngesting {
    public typealias RawChunk = String

    // Ornith's chat template prefills the opening `<think>` into the generation prompt, so the
    // model's stream starts inside reasoning and only emits the closing `</think>`.
    private var thinkSplitter = ThinkStream(startInThink: true)
    private var answer = ""
    private var reasoning = ""
    private var calls: [AgentToolCall] = []

    public init() {}

    public func ingest(_ chunk: String) -> [AgentStreamChunk] {
        route(thinkSplitter.consume(chunk))
    }

    /// Collect a tool call the library already parsed from the model's `<tool_call>` XML, mapping its
    /// `[String: JSONValue]` arguments onto the canonical ``AgentJSON``. Called by
    /// ``RebuildTurnSession`` for each `Generation.toolCall` event during the stream.
    public func ingestToolCall(_ call: ToolCall) {
        let arguments = call.function.arguments.mapValues(AgentJSON.init)
        calls.append(AgentToolCall(name: call.function.name, arguments: arguments))
    }

    public func finish() -> (stream: [AgentStreamChunk], message: AgentMessage) {
        let stream = route(thinkSplitter.finish())
        let trimmedReasoning = reasoning.trimmingCharacters(in: .whitespacesAndNewlines)
        return (
            stream,
            .ai(
                answer, toolCalls: calls, malformedToolCallBlocks: [],
                reasoning: trimmedReasoning.isEmpty ? nil : trimmedReasoning
            )
        )
    }

    /// Accumulate a split's answer/reasoning and return the non-empty pieces to stream.
    private func route(_ split: (answer: String, reasoning: String)) -> [AgentStreamChunk] {
        var pieces: [AgentStreamChunk] = []
        if !split.answer.isEmpty {
            answer += split.answer
            pieces.append(.text(split.answer))
        }
        if !split.reasoning.isEmpty {
            reasoning += split.reasoning
            pieces.append(.reasoning(split.reasoning))
        }
        return pieces
    }
}
