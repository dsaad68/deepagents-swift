@testable import DeepAgents
@testable import DeepAgentsMacTools
@testable import DeepAgentsMLX
import Testing

/// A subagent streams its answer and reasoning on separate channels, but the parent's delegate step
/// only sees `.toolProgress` deltas. `SubagentReasoningForwarder` re-inlines the subagent's reasoning
/// as `<think>…</think>` in those deltas so the parent's step (which splits inline think for display)
/// still surfaces it - and keeps `<think>` out of the tool result.
struct SubAgentReasoningForwardingTests {
    private func taskCall(_ type: String) -> AgentToolCall {
        AgentToolCall(name: "task", arguments: ["description": .string("go"), "subagent_type": .string(type)])
    }

    /// The concatenated `task` `.toolProgress` deltas the parent received.
    private func forwarded(_ events: [AgentEvent]) -> String {
        events.compactMap {
            if case .toolProgress(let name, _, let delta, _) = $0, name == "task" { return delta }
            return nil
        }.joined()
    }

    private func run(subagentModel: FakeChatModel) async -> [AgentEvent] {
        let worker = SubAgent(
            name: "worker", description: "does work", systemPrompt: "work", model: subagentModel
        )
        let agent = createDeepAgent(
            model: FakeChatModel(answer: "done", toolCalls: [taskCall("worker")]), subagents: [worker]
        )
        return await agent.collect([.human("delegate")]).events
    }

    @Test func reInlinesReasoningThenAnswer() async {
        let events = await run(subagentModel: FakeChatModel(turns: [
            FakeChatModel.Turn(text: "the answer", toolCalls: [], reasoning: "subagent thinking")
        ]))
        // The parent's delegate stream reconstructs the inline form, which ThinkingSplit recovers.
        let split = ThinkingSplit.split(forwarded(events))
        #expect(split.thinking == "subagent thinking")
        #expect(split.answer == "the answer")
        // The tool result is the clean answer, with no reasoning leaked into it.
        #expect(events.toolCompletedResults.first { $0.name == "task" }?.result == "the answer")
    }

    @Test func answerOnlySubagentForwardsNoThinkTags() async {
        let events = await run(subagentModel: FakeChatModel(answer: "just the answer"))
        #expect(!forwarded(events).contains("<think>"))
        #expect(ThinkingSplit.split(forwarded(events)).answer == "just the answer")
    }

    @Test func aReasoningOnlySubagentStillReturnsSomething() async {
        // A subagent runs the same loop, so it used to hand the parent an empty `task` result when
        // it spent its turn inside `<think>` - the empty-answer bug one level down, and worse there,
        // because the parent then reasons about a delegation that reported nothing. The loop now
        // nudges for the answer and, when even that stays silent, salvages the thinking. The
        // reasoning is still forwarded on its own channel for display.
        let events = await run(subagentModel: FakeChatModel(turns: [
            FakeChatModel.Turn(text: "", toolCalls: [], reasoning: "only reasoning")
        ]))
        #expect(ThinkingSplit.split(forwarded(events)).thinking == "only reasoning")
        // What the parent actually receives is no longer empty, and it is labelled as working-out
        // so the parent doesn't reason about a monologue as though it were the subagent's finding.
        let result = events.toolCompletedResults.first { $0.name == "task" }?.result ?? ""
        #expect(result.contains("only reasoning"))
        #expect(result.hasPrefix(ReactAgent.salvagedAnswerNote))
    }
}
