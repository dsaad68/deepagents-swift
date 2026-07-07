import DeepAgents
import Foundation

/// The LFM2 (Liquid) message codec: converts the canonical ``AgentMessage`` history into the
/// chat-template dictionaries the LFM2 Jinja template renders (encode), and reassembles the raw
/// generated token stream - stripping the `<|tool_call_start|>…<|tool_call_end|>` spans and parsing
/// their Pythonic syntax - back into a canonical assistant turn (decode). All LFM2-specific quirks
/// live here, so ``RebuildTurnSession`` only drives the MLX transport.
public struct LFM2MessageCodec: MessageCodec {
    public init() {}

    public func encode(
        _ history: [AgentMessage], systemPrompt: String?, tools: [any AgentTool], supportsVision: Bool
    ) -> LFM2Request {
        let (messages, imageURLs) = Self.renderMessages(
            systemPrompt: systemPrompt, messages: history, supportsVision: supportsVision
        )
        return LFM2Request(messages: messages, imageURLs: imageURLs, toolSpecs: tools.map { $0.toolSchema() })
    }

    public func makeDecoder() -> any TurnDecoder<String> { LFM2Decoder() }

    /// Render the conversation into the `[Message]` dictionaries the chat template expects
    /// (`Message == [String: any Sendable]`), plus the ordered image URLs to attach as
    /// `UserInput` media. The system prompt is the single leading `system` message (it is
    /// supplied separately, not part of `messages`); assistant turns carry their tool calls
    /// as `{"function": {"name", "arguments"}}` (the shape `render_tool_calls` in the LFM2
    /// template reads); tool results are plain `tool`-role turns. For VLMs, a human turn
    /// with images uses the structured `[{"type":"text"}, {"type":"image"}, …]` content the
    /// template/processor interleave with the attached images.
    ///
    /// `includeReasoning` adds each assistant turn's `"reasoning"` and `includeToolCallIDs` adds
    /// per-call `"id"` / tool-turn `"tool_call_id"` - shapes the Gemma 4 template reads (it
    /// matches tool responses to calls by id and re-renders recent reasoning). Both default off
    /// so the LFM2/qwen renders stay byte-identical.
    static func renderMessages(
        systemPrompt: String?,
        messages: [AgentMessage],
        supportsVision: Bool,
        includeReasoning: Bool = false,
        includeToolCallIDs: Bool = false
    ) -> (messages: [[String: any Sendable]], imageURLs: [URL]) {
        var dicts: [[String: any Sendable]] = []
        var imageURLs: [URL] = []

        if let systemPrompt {
            dicts.append(["role": "system", "content": systemPrompt])
        }

        for message in messages {
            switch message.role {
            case .system:
                // Not expected (the system prompt arrives separately), but render it
                // rather than drop it if one ever appears in the message list.
                dicts.append(["role": "system", "content": message.text])
            case .human:
                if supportsVision, !message.imageURLs.isEmpty {
                    var content: [[String: any Sendable]] = [
                        ["type": "text", "text": message.text]
                    ]
                    for url in message.imageURLs {
                        content.append(["type": "image"])
                        imageURLs.append(url)
                    }
                    dicts.append(["role": "user", "content": content])
                } else {
                    dicts.append(["role": "user", "content": message.text])
                }
            case .ai:
                var dict: [String: any Sendable] = [
                    "role": "assistant", "content": message.text
                ]
                if includeReasoning, let reasoning = message.reasoning, !reasoning.isEmpty {
                    dict["reasoning"] = reasoning
                }
                if !message.toolCalls.isEmpty {
                    dict["tool_calls"] = message.toolCalls.map { call -> [String: any Sendable] in
                        var rendered: [String: any Sendable] = [
                            "function": [
                                "name": call.name,
                                "arguments": call.arguments.mapValues { $0.jinjaSendable }
                            ] as [String: any Sendable]
                        ]
                        if includeToolCallIDs { rendered["id"] = call.id.uuidString }
                        return rendered
                    }
                }
                dicts.append(dict)
            case .tool:
                var dict: [String: any Sendable] = ["role": "tool", "content": message.text]
                if includeToolCallIDs, let callID = message.toolCallID {
                    dict["tool_call_id"] = callID.uuidString
                }
                dicts.append(dict)
            }
        }
        return (dicts, imageURLs)
    }
}

/// The encoded LFM2 request: the chat-template message dicts, the ordered image URLs to attach as
/// `UserInput` media, and the tool JSON schemas. ``RebuildTurnSession`` turns this into a
/// `UserInput` inside the model container.
public struct LFM2Request: Sendable {
    public let messages: [[String: any Sendable]]
    public let imageURLs: [URL]
    public let toolSpecs: [ToolSchema]
}

/// Reassembles the LFM2 token stream into one canonical assistant turn. Each chunk first has its
/// `<|tool_call_start|>…<|tool_call_end|>` spans stripped (``LFM2ToolCallStream``), then the visible
/// remainder is split into the answer and `<think>…</think>` reasoning (``ThinkStream``) so
/// reasoning streams on its own channel. `finish` parses the collected tool-call spans with
/// ``LFM2ToolCalls`` (a span that parses to nothing is a fumbled call and surfaces as a malformed
/// block for the loop to retry).
public final class LFM2Decoder: TurnDecoder {
    public typealias RawChunk = String

    private var toolSplitter = LFM2ToolCallStream()
    private var thinkSplitter = ThinkStream()
    private var answer = ""
    private var reasoning = ""

    public init() {}

    public func ingest(_ chunk: String) -> [AgentStreamChunk] {
        let visible = toolSplitter.consume(chunk)
        guard !visible.isEmpty else { return [] }
        return route(thinkSplitter.consume(visible))
    }

    public func finish() -> (stream: [AgentStreamChunk], message: AgentMessage) {
        var stream: [AgentStreamChunk] = []
        let tail = toolSplitter.finish()
        if !tail.isEmpty { stream += route(thinkSplitter.consume(tail)) }
        stream += route(thinkSplitter.finish())

        var calls: [AgentToolCall] = []
        var malformed: [String] = []
        for block in toolSplitter.toolCallBlocks {
            let parsed = LFM2ToolCalls.parse(block)
            if parsed.isEmpty {
                let trimmed = block.trimmingCharacters(in: .whitespacesAndNewlines)
                if !trimmed.isEmpty { malformed.append(trimmed) }
            }
            calls += parsed
        }
        let trimmedReasoning = reasoning.trimmingCharacters(in: .whitespacesAndNewlines)
        return (
            stream,
            .ai(
                answer, toolCalls: calls, malformedToolCallBlocks: malformed,
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

extension AgentJSON {
    /// A native, `Sendable` value for a chat-template message dictionary, so swift-jinja can
    /// render it. Mirrors `anyValue` but stays `Sendable` for the `[String: any Sendable]` message
    /// dicts. (Tool-call arguments are never `null` in practice; map it to an empty string so the
    /// template still renders something.)
    fileprivate var jinjaSendable: any Sendable {
        switch self {
        case .null: return ""
        case .bool(let value): return value
        case .int(let value): return value
        case .double(let value): return value
        case .string(let value): return value
        case .array(let value): return value.map { $0.jinjaSendable }
        case .object(let value): return value.mapValues { $0.jinjaSendable }
        }
    }
}
