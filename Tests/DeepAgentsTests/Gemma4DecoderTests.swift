@testable import DeepAgents
@testable import DeepAgentsMacTools
@testable import DeepAgentsMLX
import Foundation
import MLXLMCommon
import Testing

/// The gemma4 decoder mirrors the qwen3_5 one: the library's `GemmaFunctionParser` strips the
/// `<|tool_call>` spans and hands each call back as a `Generation.toolCall` event (collected via
/// `ingestToolCall`), so the decoder only splits the `<|channel>thought\n…<channel|>` reasoning
/// off the visible answer. Unlike Ornith, the generation prompt prefills no thought opener - the
/// stream starts in the answer and the model emits the full marker pair itself.
struct Gemma4DecoderTests {
    private func tool(_ name: String, _ arguments: [String: JSONValue]) -> ToolCall {
        ToolCall(function: .init(name: name, arguments: arguments))
    }

    @Test func splitsThoughtChannelFromAnswerAndCollectsToolCall() {
        let decoder = Gemma4Decoder()
        _ = decoder.ingest("<|channel>thought\nplan the call\n<channel|>I'll call echo.")
        decoder.ingestToolCall(tool("echo", ["text": .string("x")]))
        let message = decoder.finish().message

        #expect(message.reasoning == "plan the call")
        #expect(message.text == "I'll call echo.")
        #expect(!message.text.contains("channel"))
        #expect(message.toolCalls.first?.name == "echo")
        #expect(message.toolCalls.first?.arguments["text"] == .string("x"))
    }

    @Test func thoughtMarkersSplitAcrossChunksNeverLeak() {
        let decoder = Gemma4Decoder()
        for chunk in ["<|chan", "nel>thought\npl", "an\n<chan", "nel|>ans", "wer"] {
            _ = decoder.ingest(chunk)
        }
        let message = decoder.finish().message

        #expect(message.reasoning == "plan")
        #expect(message.text == "answer")
        #expect(!message.text.contains("channel"))
    }

    @Test func turnWithoutThoughtChannelIsAllAnswer() {
        // Thinking is template-triggered; a turn that skips the thought channel must pass through.
        let decoder = Gemma4Decoder()
        _ = decoder.ingest("just the answer")
        let message = decoder.finish().message

        #expect(message.reasoning == nil)
        #expect(message.text == "just the answer")
    }

    @Test func unterminatedThoughtChannelIsAllReasoning() {
        // The model spent its whole budget thinking and never closed the channel / answered.
        let decoder = Gemma4Decoder()
        _ = decoder.ingest("<|channel>thought\nstill reasoning")
        let message = decoder.finish().message

        #expect(message.reasoning == "still reasoning")
        #expect(message.text.isEmpty)
    }

    @Test func nestedJSONValueArgumentsBridgeToAgentJSON() {
        let decoder = Gemma4Decoder()
        decoder.ingestToolCall(tool("write_todos", [
            "todos": .array([
                .object(["title": .string("a"), "done": .bool(false)]),
                .object(["title": .string("b"), "done": .bool(true)])
            ])
        ]))
        let arguments = decoder.finish().message.toolCalls.first?.arguments

        #expect(arguments?["todos"] == .array([
            .object(["title": .string("a"), "done": .bool(false)]),
            .object(["title": .string("b"), "done": .bool(true)])
        ]))
    }
}

/// Gemma 4's chat template reads shapes the other families don't: tool responses are matched to
/// calls by `tool_call_id` == the call's `id` (an unmatched response renders as name `'unknown'`),
/// and recent assistant turns re-render their `reasoning`. The gemma encode adds those fields; the
/// LFM2/qwen renders must stay byte-identical without them.
struct Gemma4EncodeTests {
    /// A short history exercising every enriched field: an assistant tool-call turn with
    /// reasoning, then its tool result.
    private let call = AgentToolCall(name: "echo", arguments: ["text": .string("x")])
    private var history: [AgentMessage] {
        [
            .human("go"),
            .ai("calling", toolCalls: [call], reasoning: "the plan"),
            .tool("echoed", toolCallID: call.id)
        ]
    }

    @Test func gemmaEncodeCarriesReasoningAndToolCallIDs() throws {
        let request = Gemma4MessageCodec().encode(
            history, systemPrompt: "SYS", tools: [], supportsVision: false
        )
        let assistant = try #require(request.messages.first { $0["role"] as? String == "assistant" })
        #expect(assistant["reasoning"] as? String == "the plan")
        let calls = try #require(assistant["tool_calls"] as? [[String: any Sendable]])
        #expect(calls.first?["id"] as? String == call.id.uuidString)

        let tool = try #require(request.messages.first { $0["role"] as? String == "tool" })
        #expect(tool["tool_call_id"] as? String == call.id.uuidString)
    }

    @Test func gemmaSafeSchemaGivesEveryNodeAStringType() throws {
        // The Gemma template renders `value['type'] | upper` unguarded, and swift-jinja (unlike
        // Python) errors on a missing type. MCP schemas are the ones that lack it: union type
        // lists, `anyOf` with no top-level type (DeepWiki's repoName killed the first live run
        // with "upper filter requires string").
        let schema: ToolSchema = [
            "type": "function",
            "function": [
                "name": "ask_question",
                "description": "d",
                "parameters": [
                    "type": "object",
                    "properties": [
                        "repoName": ["anyOf": [
                            ["type": "string"] as [String: any Sendable],
                            ["items": ["type": "string"] as [String: any Sendable],
                             "type": "array"] as [String: any Sendable]
                        ]] as [String: any Sendable],
                        "limit": ["type": ["integer", "null"]] as [String: any Sendable],
                        "mode": ["enum": ["a", "b"]] as [String: any Sendable],
                        "tags": ["type": "array", "items": ["enum": ["x"]] as [String: any Sendable]]
                            as [String: any Sendable]
                    ] as [String: any Sendable]
                ] as [String: any Sendable]
            ] as [String: any Sendable]
        ]
        let safe = Gemma4MessageCodec.gemmaSafeSchema(schema)
        let function = try #require(safe["function"] as? [String: any Sendable])
        let parameters = try #require(function["parameters"] as? [String: any Sendable])
        let properties = try #require(parameters["properties"] as? [String: any Sendable])

        // anyOf with no top-level type adopts the first typed branch.
        #expect((properties["repoName"] as? [String: any Sendable])?["type"] as? String == "string")
        // A union list picks the first concrete member, never "null".
        #expect((properties["limit"] as? [String: any Sendable])?["type"] as? String == "integer")
        // A bare enum defaults to string (the template renders enums for STRING types).
        #expect((properties["mode"] as? [String: any Sendable])?["type"] as? String == "string")
        // Array items are normalized too.
        let tags = try #require(properties["tags"] as? [String: any Sendable])
        #expect((tags["items"] as? [String: any Sendable])?["type"] as? String == "string")
    }

    @Test func gemmaSafeSchemaLeavesWellFormedSchemasAlone() throws {
        // A built-in tool's schema (every node already typed) must pass through unchanged.
        let schema = EchoTool().toolSchema()
        let safe = Gemma4MessageCodec.gemmaSafeSchema(schema)
        #expect(NSDictionary(dictionary: safe) == NSDictionary(dictionary: schema))
    }

    @Test func lfm2AndQwenEncodesStayUnchanged() throws {
        // Regression: the extra fields are gemma-only - the LFM2/qwen templates don't read them,
        // and adding keys would perturb those prompts (and their cached prefixes) for nothing.
        let requests = [
            LFM2MessageCodec().encode(history, systemPrompt: "SYS", tools: [], supportsVision: false),
            Qwen35MessageCodec().encode(history, systemPrompt: "SYS", tools: [], supportsVision: false)
        ]
        for request in requests {
            let assistant = try #require(request.messages.first { $0["role"] as? String == "assistant" })
            #expect(assistant["reasoning"] == nil)
            let calls = try #require(assistant["tool_calls"] as? [[String: any Sendable]])
            #expect(calls.first?["id"] == nil)
            let tool = try #require(request.messages.first { $0["role"] as? String == "tool" })
            #expect(tool["tool_call_id"] == nil)
        }
    }
}
