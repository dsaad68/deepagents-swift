@testable import DeepAgents
import Foundation
import Testing

/// A tool that fails without throwing is still reported as a failure.
///
/// Most built-in tools return their errors (`ToolOutput("Error: no file at …")`) rather than
/// throwing, because the message is meant for the model to recover from. But the run then emitted
/// `.toolCompleted`, so a host drew a success tick on a call that did nothing: `read_file` given a
/// URL rendered as ✓ next to "Error: no file at "https://…"". The text is identical either way -
/// only the event changes.
struct ToolFailureReportingTests {
    private struct FailingTool: AgentTool {
        var name: String { "always_fails" }
        var description: String { "Fails" }
        func execute(_: [String: AgentJSON], _: ToolContext) async throws -> ToolOutput {
            .failure("Error: nothing to do here.")
        }
    }

    private struct WorkingTool: AgentTool {
        var name: String { "always_works" }
        var description: String { "Works" }
        func execute(_: [String: AgentJSON], _: ToolContext) async throws -> ToolOutput {
            ToolOutput("all good")
        }
    }

    private func call(_ name: String) -> AgentToolCall {
        AgentToolCall(name: name, arguments: [:])
    }

    @Test func aReturnedFailureIsReportedAsAFailure() async {
        let model = FakeChatModel(turns: [
            .init(text: "", toolCalls: [call("always_fails")]), .init(text: "done", toolCalls: [])
        ])
        let (_, events) = await createAgent(model: model, tools: [FailingTool()]).collect([.human("go")])

        #expect(events.toolFailedNames == ["always_fails"])
        #expect(events.toolCompletedResults.isEmpty) // never both
    }

    /// The model must still receive the message verbatim - this changes how the run reports the
    /// call, not what the model gets to recover from.
    @Test func theMessageStillReachesTheModelUnchanged() async {
        let memory = InMemoryCheckpointer()
        let model = FakeChatModel(turns: [
            .init(text: "", toolCalls: [call("always_fails")]), .init(text: "done", toolCalls: [])
        ])
        let agent = createAgent(model: model, tools: [FailingTool()], memory: memory)

        _ = await agent.collect([.human("go")], threadId: "t")

        let toolTurn = await memory.load("t").first { $0.role == .tool }
        #expect(toolTurn?.text == "Error: nothing to do here.")
    }

    @Test func anOrdinaryResultIsStillACompletion() async {
        let model = FakeChatModel(turns: [
            .init(text: "", toolCalls: [call("always_works")]), .init(text: "done", toolCalls: [])
        ])
        let (_, events) = await createAgent(model: model, tools: [WorkingTool()]).collect([.human("go")])

        #expect(events.toolCompletedResults.map(\.name) == ["always_works"])
        #expect(events.toolFailedNames.isEmpty)
    }

    /// End to end on the tool from the report: a miss is a failure, not a tick.
    @Test func readingAMissingFileReportsAFailure() async {
        let model = FakeChatModel(turns: [
            .init(text: "", toolCalls: [
                AgentToolCall(name: "read_file", arguments: ["file_path": .string("nope.txt")])
            ]),
            .init(text: "done", toolCalls: [])
        ])
        let agent = createDeepAgent(model: model, backend: StateBackend())

        let (_, events) = await agent.collect([.human("go")])

        #expect(events.toolFailedNames.contains("read_file"))
        #expect(!events.toolCompletedResults.contains { $0.name == "read_file" })
    }

    /// A small model picks its tool from the description, so the one line that stops it passing a
    /// URL to `read_file` has to name the alternative rather than only forbid the mistake.
    @Test func readFileSaysWhatToUseForAWebPage() {
        let description = ReadFileTool(backend: StateBackend()).description
        #expect(description.lowercased().contains("url"))
        #expect(description.lowercased().contains("web"))
    }
}
