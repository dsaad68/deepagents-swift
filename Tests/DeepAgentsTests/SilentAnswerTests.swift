@testable import DeepAgents
import Foundation
import Testing

/// A turn with no tool calls used to end the run, whatever it contained. But "no tool calls" is not
/// "here is the answer": a reasoning model can spend a whole turn inside `<think>` and never reach
/// the visible text, and the run then completed successfully with an empty reply. Observed
/// on-device on a 2.6B - five rounds of real tool work, 6,788 reasoning chunks, zero answer
/// tokens, nothing shown to the user.
struct SilentAnswerTests {
    private func call(_ text: String) -> AgentToolCall {
        AgentToolCall(name: "echo", arguments: ["text": .string(text)])
    }

    /// The message lists handed to the model, so a test can assert what the nudged turn actually
    /// sent - the fabricated turn never reaches `state.messages`, so the model's view is the only
    /// place it is observable.
    private actor SentMessages {
        private(set) var all: [[AgentMessage]] = []
        /// Whether each turn was asked to begin outside the reasoning channel.
        private(set) var outsideReasoning: [Bool] = []
        func record(_ messages: [AgentMessage], outsideReasoning flag: Bool) {
            all.append(messages)
            outsideReasoning.append(flag)
        }

        var last: [AgentMessage] { all.last ?? [] }
    }

    /// A scripted model that also records the conversation it was handed each turn.
    private struct RecordingChatModel: ChatModel {
        let sent: SentMessages
        let turns: [FakeChatModel.Turn]
        var supportsVision = false

        func makeSession() -> any ModelTurnSession {
            RecordingSession(sent: sent, inner: FakeTurnSession(turns: turns))
        }
    }

    private final class RecordingSession: ModelTurnSession {
        private let sent: SentMessages
        private let inner: FakeTurnSession

        init(sent: SentMessages, inner: FakeTurnSession) {
            self.sent = sent
            self.inner = inner
        }

        func nextTurn(
            messages: [AgentMessage],
            systemPrompt: String?,
            tools: [any AgentTool],
            onChunk: @escaping @Sendable (AgentStreamChunk) -> Void
        ) async throws -> AgentMessage {
            try await nextTurn(
                messages: messages, systemPrompt: systemPrompt, tools: tools,
                startingOutsideReasoning: false, onChunk: onChunk
            )
        }

        func nextTurn(
            messages: [AgentMessage],
            systemPrompt: String?,
            tools: [any AgentTool],
            startingOutsideReasoning: Bool,
            onChunk: @escaping @Sendable (AgentStreamChunk) -> Void
        ) async throws -> AgentMessage {
            await sent.record(messages, outsideReasoning: startingOutsideReasoning)
            return try await inner.nextTurn(
                messages: messages, systemPrompt: systemPrompt, tools: tools, onChunk: onChunk
            )
        }
    }

    @Test func aTurnThatOnlyThinksIsNudgedForTheAnswer() async {
        let model = FakeChatModel(turns: [
            .init(text: "", toolCalls: [call("x")]),
            // The model finishes its work, then thinks instead of answering.
            .init(text: "", toolCalls: [], reasoning: "I should summarise what I found"),
            // The nudge lands and it answers.
            .init(text: "here is the answer", toolCalls: [])
        ])
        let agent = createAgent(model: model, tools: [EchoTool()])

        let (ok, events) = await agent.collect([.human("go")])

        #expect(ok)
        #expect(events.finalAnswer == "here is the answer")
        #expect(events.didComplete)
    }

    @Test func theSalvagedThinkingIsBothStreamedAndStored() async {
        let memory = InMemoryCheckpointer()
        let model = FakeChatModel(turns: [
            .init(text: "", toolCalls: [], reasoning: "the branch is clean"),
            // Even the nudged turn stays inside <think>.
            .init(text: "", toolCalls: [], reasoning: "still thinking, no answer")
        ])
        let agent = createAgent(model: model, tools: [EchoTool()], memory: memory)

        let (ok, events) = await agent.collect([.human("go")], threadId: "t")

        #expect(ok)
        // A host builds its visible answer from `.token`, so salvaging into the stored message
        // alone would still leave the screen blank.
        #expect(events.finalAnswer.contains("still thinking, no answer"))
        // …and it is labelled, so working-out is never mistaken for a considered reply.
        #expect(events.finalAnswer.hasPrefix(ReactAgent.salvagedAnswerNote))
        let saved = await memory.load("t")
        #expect(saved.last?.text.contains("still thinking, no answer") == true)
        #expect(saved.last?.role == .ai)
    }

    /// The ask is delivered as a turn at the end of the conversation, carrying the model's own
    /// reasoning back. Leaving it in the system prompt alone put it thousands of tokens from the
    /// generation point and handed the model the identical conversation that just silenced it.
    @Test func theAskIsATrailingTurnCarryingTheReasoningBack() async {
        let seen = SentMessages()
        let model = RecordingChatModel(
            sent: seen,
            turns: [
                .init(text: "", toolCalls: [], reasoning: "the branch is clean and nothing changed"),
                .init(text: "the branch is clean", toolCalls: [])
            ]
        )
        let agent = createAgent(model: model, tools: [EchoTool()])

        let (ok, events) = await agent.collect([.human("go")])

        #expect(ok)
        #expect(events.finalAnswer == "the branch is clean")
        // The nudged turn ends with a human message quoting the reasoning the model had already
        // produced, so answering is a summary rather than a re-derivation.
        let nudged = await seen.last
        #expect(nudged.last?.role == .human)
        #expect(nudged.last?.text.contains("the branch is clean and nothing changed") == true)
        #expect(nudged.last?.text.contains("final answer") == true)
    }

    /// It is a `.human` turn, not a mid-conversation `.system` one: system-role messages inside
    /// `messages` are model-gated (Opus 4.8+; a 400 on Sonnet 5, Haiku, and older models), and this
    /// path runs when a turn has already gone wrong.
    @Test func theAskNeverUsesAModelGatedSystemTurn() async {
        let seen = SentMessages()
        let model = RecordingChatModel(
            sent: seen,
            turns: [.init(text: "", toolCalls: [], reasoning: "thinking"), .init(text: "done", toolCalls: [])]
        )

        _ = await createAgent(model: model, tools: [EchoTool()]).collect([.human("go")])

        #expect(await seen.last.allSatisfy { $0.role != .system })
    }

    /// The forced turn asks to begin in the answer channel. On a model whose template opens
    /// `<think>` every turn, an answer only reaches `text` after the model emits `</think>` - so
    /// asking a turn that has already thought itself silent to "answer in plain text" is asking it
    /// to answer on a channel that is not open. Ordinary rounds are untouched: they may reason.
    @Test func theForcedTurnStartsOutsideTheReasoningChannel() async {
        let seen = SentMessages()
        let model = RecordingChatModel(
            sent: seen,
            turns: [
                .init(text: "", toolCalls: [call("x")]),
                .init(text: "", toolCalls: [], reasoning: "thinking"),
                .init(text: "the answer", toolCalls: [])
            ]
        )

        _ = await createAgent(model: model, tools: [EchoTool()]).collect([.human("go")])

        let asked = await seen.outsideReasoning
        #expect(asked.last == true) // the forced turn
        #expect(asked.dropLast().allSatisfy { $0 == false }) // every ordinary round
    }

    /// The fabricated turn is never stored: the user did not type it, and a saved thread that
    /// contains words they never wrote is worse than the bug it fixes.
    @Test func theAskIsNeverWrittenToTheSavedThread() async {
        let memory = InMemoryCheckpointer()
        let model = FakeChatModel(turns: [
            .init(text: "", toolCalls: [], reasoning: "thinking"),
            .init(text: "the answer", toolCalls: [])
        ])
        let agent = createAgent(model: model, tools: [EchoTool()], memory: memory)

        _ = await agent.collect([.human("go")], threadId: "t")

        let saved = await memory.load("t")
        #expect(saved.map(\.role) == [.human, .ai])
        #expect(saved.first?.text == "go") // the only human turn is the one the user typed
        #expect(saved.last?.text == "the answer")
    }

    /// The nudge is asked once. A model that only ever thinks has to end the run rather than be
    /// asked again and again until the iteration cap.
    @Test func theModelIsNudgedOnlyOnce() async {
        let recorder = RunRecorder()
        let model = FakeChatModel(turns: [
            .init(text: "", toolCalls: [], reasoning: "a"),
            .init(text: "", toolCalls: [], reasoning: "b"),
            .init(text: "", toolCalls: [], reasoning: "c")
        ])
        let agent = createAgent(
            model: model, tools: [EchoTool()], middleware: [RequestRecordingMiddleware(recorder: recorder)]
        )

        let (ok, _) = await agent.collect([.human("go")])

        #expect(ok)
        // One ordinary round; the nudge bypasses the middleware chain, so exactly one request is
        // recorded. Without the guard the loop would keep re-asking a silent model.
        #expect(await recorder.systemPrompts.count == 1)
    }

    /// The floor: a model that produces no text *and* no reasoning, twice. There is nothing to
    /// salvage, so the run must still say something rather than completing with an empty string -
    /// a blank reply tells the user only that something broke.
    @Test func aModelThatProducesNothingAtAllStillSaysSo() async {
        let memory = InMemoryCheckpointer()
        let model = FakeChatModel(turns: [
            .init(text: "", toolCalls: []),
            .init(text: "", toolCalls: [])
        ])
        let agent = createAgent(model: model, tools: [EchoTool()], memory: memory)

        let (ok, events) = await agent.collect([.human("go")], threadId: "t")

        #expect(ok)
        #expect(!events.finalAnswer.isEmpty)
        let saved = await memory.load("t")
        #expect(saved.last?.text.isEmpty == false)
    }

    // MARK: - The other two ways a run can end without an answer

    // `nudgeIfSilent` is not the common case. Across eight on-device runs it fired once, while
    // three of the four empty-answer rescues came through the iteration cap and the duplicate-round
    // guard - both of which force a final turn with no reasoning to hand back. Those paths get the
    // same guarantee: whatever ends the run, the user is not left with an empty reply.

    /// Hitting the cap forces a last turn. When that turn is silent too, the run still answers.
    @Test func theIterationCapNeverEndsOnAnEmptyAnswer() async {
        let model = FakeChatModel(turns: [
            .init(text: "", toolCalls: [call("a")]),
            .init(text: "", toolCalls: [call("b")]),
            .init(text: "", toolCalls: [call("c")]),
            // The forced turn thinks instead of answering - the shape seen on-device.
            .init(text: "", toolCalls: [], reasoning: "I have gathered enough to summarise")
        ])
        let agent = createAgent(model: model, tools: [EchoTool()], maxIterations: 3)

        let (ok, events) = await agent.collect([.human("go")])

        #expect(ok)
        #expect(events.finalAnswer.contains("I have gathered enough to summarise"))
        #expect(events.finalAnswer.hasPrefix(ReactAgent.salvagedAnswerNote))
    }

    /// The duplicate-round guard stops a model repeating one call. Its forced turn is covered too.
    @Test func theDuplicateRoundGuardNeverEndsOnAnEmptyAnswer() async {
        let model = FakeChatModel(turns: [
            .init(text: "", toolCalls: [call("same")]),
            .init(text: "", toolCalls: [call("same")]), // redirected
            .init(text: "", toolCalls: [call("same")]), // stopped
            .init(text: "", toolCalls: [], reasoning: "the echo keeps returning the same thing")
        ])
        let agent = createAgent(model: model, tools: [EchoTool()])

        let (ok, events) = await agent.collect([.human("go")])

        #expect(ok)
        #expect(events.finalAnswer.contains("the echo keeps returning the same thing"))
    }

    /// The cap with nothing at all to salvage - the floor for the path that fires most often.
    @Test func theIterationCapStillSaysSomethingWithNothingToSalvage() async {
        let model = FakeChatModel(turns: [
            .init(text: "", toolCalls: [call("a")]),
            .init(text: "", toolCalls: [call("b")])
        ]) // the forced turn falls off the end of the script: silent, no reasoning
        let agent = createAgent(model: model, tools: [EchoTool()], maxIterations: 2)

        let (ok, events) = await agent.collect([.human("go")])

        #expect(ok)
        #expect(events.finalAnswer == ReactAgent.noAnswerNote)
    }

    /// Whitespace is not an answer either - a model that emits a stray newline has said nothing.
    @Test func whitespaceCountsAsSilence() async {
        let model = FakeChatModel(turns: [
            .init(text: "  \n ", toolCalls: []),
            .init(text: "the real answer", toolCalls: [])
        ])
        let agent = createAgent(model: model, tools: [EchoTool()])

        let (_, events) = await agent.collect([.human("go")])

        #expect(events.finalAnswer.contains("the real answer"))
    }

    /// The ordinary path is untouched: a turn with text ends the run without a second model call.
    @Test func anAnsweredTurnIsNotNudged() async {
        let recorder = RunRecorder()
        let model = FakeChatModel(answer: "done")
        let agent = createAgent(
            model: model, middleware: [RequestRecordingMiddleware(recorder: recorder)]
        )

        let (ok, events) = await agent.collect([.human("go")])

        #expect(ok)
        #expect(events.finalAnswer == "done")
        #expect(await recorder.systemPrompts.count == 1)
    }
}
