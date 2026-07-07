import DeepAgents
import Foundation
import MLXLMCommon

// Gemma 4 (gemma4) is a reasoning VLM: with thinking enabled (the chat template's
// `enable_thinking` flag opens the first system turn with `<|think|>`), each assistant turn may
// begin with a `<|channel>thought\n…<channel|>` block before the answer, and tool calls are
// emitted as `<|tool_call>call:name{…}<tool_call|>`. Like qwen3_5 - and unlike LFM2 - the
// library's parser handles the tool format correctly (`ToolCallFormat.gemma4`, auto-inferred
// from the model_type), so ``RebuildTurnSession`` leaves it active: the `<|tool_call>` spans are
// stripped from the visible chunks and surfaced as `Generation.toolCall` events, and this codec's
// decoder only splits the thought channel from the answer and collects the already-parsed calls.

/// The Gemma 4 message codec. Encoding reuses ``LFM2MessageCodec/renderMessages`` with the two
/// extras the Gemma 4 template reads: per-call ids + `tool_call_id` on tool turns (the template
/// matches tool responses to calls by id, falling back to the name `'unknown'`), and each
/// assistant turn's `reasoning` (re-rendered for recent tool-calling turns). Only decoding
/// otherwise differs from the qwen path: the thought channel uses Gemma's marker pair, and the
/// stream starts in the answer (the generation prompt ends at `<|turn>model` with no prefilled
/// thought opener - the opposite of Ornith's prefilled `<think>`).
public struct Gemma4MessageCodec: MessageCodec {
    public init() {}

    public func encode(
        _ history: [AgentMessage], systemPrompt: String?, tools: [any AgentTool], supportsVision: Bool
    ) -> LFM2Request {
        let (messages, imageURLs) = LFM2MessageCodec.renderMessages(
            systemPrompt: systemPrompt, messages: history, supportsVision: supportsVision,
            includeReasoning: true, includeToolCallIDs: true
        )
        return LFM2Request(
            messages: messages, imageURLs: imageURLs,
            toolSpecs: tools.map { Self.gemmaSafeSchema($0.toolSchema()) }
        )
    }

    public func makeDecoder() -> any TurnDecoder<String> { Gemma4Decoder() }

    /// The Gemma 4 chat template requires every schema node it renders to carry a *string*
    /// `type` (`value['type'] | upper`, unguarded - Python jinja stringifies a missing key to ""
    /// and renders an empty type; swift-jinja raises "upper filter requires string" and kills the
    /// turn). Built-in tools always emit one, but MCP servers hand over arbitrary JSON Schema -
    /// union `type` arrays, `anyOf` branches with no top-level `type` at all (e.g. DeepWiki's
    /// `repoName`). Normalize every node to a concrete string `type` - which also renders a more
    /// useful declaration than Python's silent empty string.
    static func gemmaSafeSchema(_ schema: ToolSchema) -> ToolSchema {
        var schema = schema
        if var function = schema["function"] as? [String: any Sendable] {
            if let parameters = function["parameters"] as? [String: any Sendable] {
                function["parameters"] = normalizedNode(parameters)
            }
            schema["function"] = function
        }
        return schema
    }

    /// Recursively give a JSON-schema node a string `type` (deriving one from a union list, an
    /// `anyOf`/`oneOf` branch, or the node's own shape) and descend into `properties`/`items`.
    private static func normalizedNode(_ node: [String: any Sendable]) -> [String: any Sendable] {
        var node = node
        if !(node["type"] is String) {
            if let union = node["type"] as? [Any] {
                // A union type list: pick the first concrete member (never "null").
                let members = union.compactMap { $0 as? String }.filter { $0 != "null" }
                node["type"] = members.first ?? "string"
            } else if let branches = (node["anyOf"] ?? node["oneOf"]) as? [Any] {
                // No type at all: adopt the first typed branch's shape (the template ignores the
                // anyOf/oneOf keys themselves, so this is the only part of the union it can see).
                let typed = branches
                    .compactMap { $0 as? [String: any Sendable] }
                    .map { normalizedNode($0) }
                    .first { $0["type"] is String }
                node["type"] = typed?["type"] ?? "string"
                for key in ["items", "properties", "enum", "required"] where node[key] == nil {
                    node[key] = typed?[key]
                }
            } else if node["properties"] is [String: any Sendable] {
                node["type"] = "object"
            } else {
                node["type"] = node["items"] != nil ? "array" : "string"
            }
        }
        if var properties = node["properties"] as? [String: any Sendable] {
            for (key, value) in properties {
                if let property = value as? [String: any Sendable] {
                    properties[key] = normalizedNode(property)
                }
            }
            node["properties"] = properties
        }
        if let items = node["items"] as? [String: any Sendable] {
            node["items"] = normalizedNode(items)
        }
        return node
    }
}

/// Reassembles the gemma4 token stream into one canonical assistant turn. The `<|tool_call>` spans
/// are already removed from the chunk stream by the library's `GemmaFunctionParser` and delivered
/// separately via ``ingestToolCall(_:)``, so this decoder only splits each visible chunk into the
/// answer and `<|channel>thought\n…<channel|>` reasoning (``ThinkStream`` with Gemma's markers).
public final class Gemma4Decoder: TurnDecoder, ToolCallIngesting {
    public typealias RawChunk = String

    /// Gemma 4's thought-channel markers, per the shipped chat template. The start tag includes
    /// the trailing newline the template shows, so the reasoning text starts clean.
    static let thoughtStartTag = "<|channel>thought\n"
    static let thoughtEndTag = "<channel|>"

    private var thinkSplitter = ThinkStream(startTag: thoughtStartTag, endTag: thoughtEndTag)
    private var answer = ""
    private var reasoning = ""
    private var calls: [AgentToolCall] = []

    public init() {}

    public func ingest(_ chunk: String) -> [AgentStreamChunk] {
        route(thinkSplitter.consume(chunk))
    }

    /// Collect a tool call the library already parsed from the model's `<|tool_call>` span,
    /// mapping its `[String: JSONValue]` arguments onto the canonical ``AgentJSON``. Called by
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
