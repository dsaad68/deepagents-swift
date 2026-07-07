@testable import DeepAgents
@testable import DeepAgentsMacTools
@testable import DeepAgentsMLX
import MLXLMCommon
import Testing

/// The qwen3_5 / Ornith decoder is deliberately thin: the library's `XMLFunctionParser` strips the
/// `<tool_call>` XML and hands each call back as a `Generation.toolCall` event (collected via
/// `ingestToolCall`), so the decoder only has to split `<think>…</think>` reasoning off the visible
/// answer. Ornith's chat template prefills the opening `<think>`, so the generated stream starts
/// *inside* reasoning and emits only the closing `</think>` (the streams below omit the leading tag
/// to mirror that). These tests pin both halves and the `JSONValue` → `AgentJSON` bridge.
struct Qwen35DecoderTests {
    private func tool(_ name: String, _ arguments: [String: JSONValue]) -> ToolCall {
        ToolCall(function: .init(name: name, arguments: arguments))
    }

    @Test func splitsReasoningFromAnswerAndCollectsToolCall() {
        let decoder = Qwen35Decoder()
        _ = decoder.ingest("plan</think>I'll call echo.")
        decoder.ingestToolCall(tool("echo", ["text": .string("x")]))
        let message = decoder.finish().message

        #expect(message.reasoning == "plan")
        #expect(message.text == "I'll call echo.")
        #expect(!message.text.contains("</think>"))
        #expect(message.toolCalls.first?.name == "echo")
        #expect(message.toolCalls.first?.arguments["text"] == .string("x"))
    }

    @Test func reasoningEndTagSplitAcrossChunksNeverLeaksIntoAnswer() {
        let decoder = Qwen35Decoder()
        for chunk in ["pl", "an</thi", "nk>ans", "wer"] { _ = decoder.ingest(chunk) }
        let message = decoder.finish().message

        #expect(message.reasoning == "plan")
        #expect(message.text == "answer")
        #expect(!message.text.contains("think"))
    }

    @Test func reasoningOnlyTurnWithNoCloseIsAllReasoning() {
        // The model spent its whole budget thinking and never closed the block / answered.
        let decoder = Qwen35Decoder()
        _ = decoder.ingest("still reasoning")
        let message = decoder.finish().message

        #expect(message.reasoning == "still reasoning")
        #expect(message.text.isEmpty)
    }

    @Test func multipleToolCallsAreCollectedInOrder() {
        let decoder = Qwen35Decoder()
        _ = decoder.ingest("two steps</think>")
        decoder.ingestToolCall(tool("first", [:]))
        decoder.ingestToolCall(tool("second", ["n": .int(2)]))
        let message = decoder.finish().message

        #expect(message.toolCalls.map(\.name) == ["first", "second"])
        #expect(message.toolCalls.last?.arguments["n"] == .int(2))
    }

    @Test func nestedJSONValueArgumentsBridgeToAgentJSON() {
        let decoder = Qwen35Decoder()
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

/// `ThinkStream` is the shared reasoning splitter the on-device decoders route their visible text
/// through. It must hold a tag split across chunk boundaries, treat an unterminated `<think>` as
/// in-progress reasoning, and (for the Ornith path) start in-think when the prompt prefilled the
/// opening tag. The tags are configurable (Gemma 4 uses its own marker pair); the defaults stay
/// `<think>`/`</think>`. The helper accumulates across `consume`/`finish` so the assertions don't
/// depend on which call happens to flush a given piece.
struct ThinkStreamTests {
    private func run(
        _ chunks: [String], startInThink: Bool = false,
        startTag: String = ThinkStream.startTag, endTag: String = ThinkStream.endTag
    ) -> (answer: String, reasoning: String) {
        var stream = ThinkStream(startInThink: startInThink, startTag: startTag, endTag: endTag)
        var answer = ""
        var reasoning = ""
        for chunk in chunks {
            let split = stream.consume(chunk)
            answer += split.answer
            reasoning += split.reasoning
        }
        let tail = stream.finish()
        return (answer + tail.answer, reasoning + tail.reasoning)
    }

    @Test func plainChunkIsAllAnswer() {
        let result = run(["hello"])
        #expect(result.answer == "hello")
        #expect(result.reasoning.isEmpty)
    }

    @Test func endTagSplitAcrossChunksIsNeverLeaked() {
        let result = run(["<think>reasoning</thi", "nk>answer"])
        #expect(result.reasoning == "reasoning") // the partial `</thi` is buffered, not emitted
        #expect(result.answer == "answer")
        #expect(!result.answer.contains("think"))
    }

    @Test func unterminatedThinkFlushesAsReasoning() {
        let result = run(["<think>still going"])
        #expect(result.reasoning == "still going")
        #expect(result.answer.isEmpty)
    }

    @Test func startInThinkRoutesLeadingContentToReasoning() {
        // Ornith: the opening `<think>` was prefilled into the prompt, so the stream starts in
        // reasoning and only the closing tag is generated.
        let result = run(["the plan</think>the answer"], startInThink: true)
        #expect(result.reasoning == "the plan")
        #expect(result.answer == "the answer")
    }

    @Test func customTagsSplitWithCrossChunkHoldBack() {
        // Gemma 4's marker pair, each split across a chunk boundary - the partial tag must be
        // buffered, never emitted into the wrong channel.
        let result = run(
            ["before<|channel>th", "ought\nplan<chan", "nel|>after"],
            startTag: "<|channel>thought\n", endTag: "<channel|>"
        )
        #expect(result.answer == "beforeafter")
        #expect(result.reasoning == "plan")
    }

    @Test func customTagsDoNotReactToTheDefaultPair() {
        // With Gemma tags configured, a literal `<think>` is ordinary answer text.
        let result = run(
            ["a <think>quote</think> b"],
            startTag: "<|channel>thought\n", endTag: "<channel|>"
        )
        #expect(result.answer == "a <think>quote</think> b")
        #expect(result.reasoning.isEmpty)
    }
}
