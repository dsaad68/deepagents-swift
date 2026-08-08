@testable import DeepAgents
import DeepAgentsMacTools
import Foundation
import Testing

// Tests for lazy tool loading: how tiers expand out of an `AgentToolPolicy`, and how
// `ToolSearchMiddleware` keeps auxiliary tools out of the rendered prompt while leaving them
// dispatchable.
//
// The load-bearing test here is `renderedToolsNeverChange`. The whole design exists to keep the
// prompt prefix - and therefore `MlxChatModel`'s prefix KV cache - byte-identical across a run even
// as the agent discovers and calls tools it could not see. That is a property of the rendered tool
// list, so it is asserted directly on the rendered tool list.

/// A model that walks the real discovery path: search, then call what the search returned, then
/// answer. Counting rounds (rather than inspecting history) keeps it independent of message shape.
private struct ToolSearchScriptModel: ChatModel {
    var supportsVision = false
    /// The tool the script calls in round 2, and how it spells the call.
    let target: String
    let viaRunTool: Bool

    func makeSession() -> any ModelTurnSession { ToolSearchScriptSession(target: target, viaRunTool: viaRunTool) }
}

private final class ToolSearchScriptSession: ModelTurnSession {
    private let target: String
    private let viaRunTool: Bool
    private var round = 0

    init(target: String, viaRunTool: Bool) {
        self.target = target
        self.viaRunTool = viaRunTool
    }

    func nextTurn(
        messages: [AgentMessage],
        systemPrompt: String?,
        tools: [any AgentTool],
        onChunk: @escaping @Sendable (AgentStreamChunk) -> Void
    ) async throws -> AgentMessage {
        round += 1
        switch round {
        case 1:
            return .ai("", toolCalls: [
                AgentToolCall(name: "search_tools", arguments: ["query": .string("echo some text")])
            ])
        case 2 where viaRunTool:
            return .ai("", toolCalls: [
                AgentToolCall(name: "run_tool", arguments: [
                    "name": .string(target),
                    "arguments": .object(["text": .string("hi")])
                ])
            ])
        case 2:
            return .ai("", toolCalls: [
                AgentToolCall(name: target, arguments: ["text": .string("hi")])
            ])
        default:
            return .ai("done")
        }
    }
}

/// Collects `.toolCompleted` events from a run. A class with a lock rather than an actor so the
/// `@Sendable` event closure can record synchronously.
private final class ToolEventSink: @unchecked Sendable {
    private let lock = NSLock()
    private var storage: [(name: String, result: String)] = []

    func record(name: String, result: String) {
        lock.lock()
        storage.append((name, result))
        lock.unlock()
    }

    var completions: [(name: String, result: String)] {
        lock.lock()
        defer { lock.unlock() }
        return storage
    }
}

@Suite("Tool tiers")
struct ToolTierExpansionTests {
    @Test("Tiers are inert until the feature is switched on")
    func offByDefault() {
        let policy = AgentToolPolicy(auxiliaryMiddleware: ["git"], auxiliaryTools: ["fetch"])
        #expect(policy.expand().auxiliaryToolNames.isEmpty)
    }

    @Test("An auxiliary middleware contributes all of its tools")
    func auxiliaryMiddleware() {
        let policy = AgentToolPolicy(toolSearch: true, auxiliaryMiddleware: ["git"])
        let auxiliary = policy.expand().auxiliaryToolNames

        for tool in ["git_status", "git_diff", "git_log", "git_show", "git_blame"] {
            #expect(auxiliary.contains(tool))
        }
        #expect(!auxiliary.contains("read_file")) // a core toolset stays core
    }

    @Test("Individual tools can be tiered without their middleware")
    func auxiliaryTool() {
        let policy = AgentToolPolicy(toolSearch: true, auxiliaryTools: ["curl"])
        let auxiliary = policy.expand().auxiliaryToolNames

        #expect(auxiliary == ["curl"])
        #expect(!auxiliary.contains("fetch")) // its sibling in the `web` middleware is untouched
    }

    @Test("A disabled tool is never offered as auxiliary")
    func disabledBeatsAuxiliary() {
        // Disabled and auxiliary are different things: a disabled tool isn't in the agent's tool list
        // at all, so listing it in the search corpus would advertise something undispatchable.
        let policy = AgentToolPolicy(
            disabledMiddleware: ["git"], toolSearch: true, auxiliaryMiddleware: ["git"]
        )
        let expansion = policy.expand()

        #expect(expansion.disabledToolNames.contains("git_status"))
        #expect(expansion.auxiliaryToolNames.isEmpty)
    }

    @Test("Auxiliary tools are still gated by their approval mode")
    func auxiliaryStillGated() {
        // The tier is about prefill, not permission - an auxiliary write must still ask.
        let policy = AgentToolPolicy(toolSearch: true, auxiliaryMiddleware: ["filesystem"])
        let expansion = policy.expand()

        #expect(expansion.auxiliaryToolNames.contains("write_file"))
        #expect(expansion.interruptOn["write_file"] != nil)
    }

    @Test("MCP servers default to auxiliary; promotions are stored as core")
    func mcpTiers() {
        let servers = [
            MCPServerConfig(name: "deepwiki", kind: .http),
            MCPServerConfig(name: "notes", kind: .stdio)
        ]
        let tiered = AgentToolPolicy(coreMCPServers: ["notes"]).tiered(servers)

        #expect(tiered.first { $0.name == "deepwiki" }?.tier == .auxiliary)
        #expect(tiered.first { $0.name == "notes" }?.tier == .core)
    }

    @Test("A stored policy without the tier keys still decodes")
    func tolerantDecoding() throws {
        // The pre-feature shape on disk. Anything missing must fall back, or `try?` at the load site
        // would silently drop the user's whole policy.
        let json = #"{"disabledMiddleware":["git"],"approvals":{"fetch":"deny"}}"#
        let policy = try JSONDecoder().decode(AgentToolPolicy.self, from: Data(json.utf8))

        #expect(policy.disabledMiddleware == ["git"])
        #expect(policy.toolSearch == false)
        #expect(policy.auxiliaryMiddleware.isEmpty)
        #expect(policy.coreMCPServers.isEmpty)
        #expect(policy.toolSearchLimit == 5)
        #expect(policy.toolSearchModel == nil)
    }
}

@Suite("ToolSearchMiddleware")
struct ToolSearchMiddlewareTests {
    /// An agent whose one auxiliary tool is `echo`, with a recorder registered innermost so it sees
    /// the request exactly as the model does.
    private func makeAgent(viaRunTool: Bool) -> (ReactAgent, RunRecorder) {
        let recorder = RunRecorder()
        let search = ToolSearchMiddleware(auxiliaryTools: [EchoTool()])
        let agent = createAgent(
            model: ToolSearchScriptModel(target: "echo", viaRunTool: viaRunTool),
            tools: [EchoTool()],
            systemPrompt: "base",
            middleware: [search, RequestRecordingMiddleware(recorder: recorder)]
        )
        return (agent, recorder)
    }

    @Test("The rendered tool set is identical on every round, before and after a search")
    func renderedToolsNeverChange() async throws {
        // This is the invariant the prefix KV cache depends on: `MlxChatModel` fingerprints
        // `systemPrompt + toolNames` and resets its cache when either changes, so discovering and
        // calling `echo` must not alter what is rendered.
        let (agent, recorder) = makeAgent(viaRunTool: false)
        _ = await agent.run([.human("echo hi")]) { _ in }

        let sets = await recorder.toolNameSets
        #expect(sets.count >= 3) // search, the call it enabled, the answer
        for names in sets {
            #expect(!names.contains("echo")) // never rendered...
            #expect(names.contains("search_tools"))
            #expect(names.contains("run_tool"))
        }
        #expect(Set(sets.map { $0.sorted() }).count == 1) // ...and never varies

        let prompts = await recorder.systemPrompts
        #expect(Set(prompts.map { $0 ?? "" }).count == 1) // the other half of the fingerprint
    }

    /// Every `(tool, result)` the run completed, in order.
    private func completions(_ agent: ReactAgent) async -> [(name: String, result: String)] {
        let sink = ToolEventSink()
        _ = await agent.run([.human("echo hi")]) { event in
            if case .toolCompleted(let name, let result, _, _, _) = event {
                sink.record(name: name, result: result)
            }
        }
        return sink.completions
    }

    @Test("An auxiliary tool still executes when the model calls it by name")
    func auxiliaryToolDispatches() async throws {
        let (agent, _) = makeAgent(viaRunTool: false)
        let completed = await completions(agent)

        // ...invisible, but executable: dispatch resolves against the agent's full tool list.
        #expect(completed.contains { $0.name == "search_tools" })
        #expect(completed.contains { $0.name == "echo" && $0.result.contains("echo: hi") })
    }

    @Test("run_tool is rewritten into a direct call, so the real tool runs")
    func runToolDispatches() async throws {
        let (agent, _) = makeAgent(viaRunTool: true)
        let completed = await completions(agent)

        // The rewrite lands before dispatch, so everything downstream - the approval gate, the
        // message log, these events - sees `echo` rather than `run_tool`.
        #expect(completed.contains { $0.name == "echo" && $0.result.contains("echo: hi") })
        #expect(!completed.contains { $0.name == "run_tool" })
    }

    @Test("With nothing auxiliary the middleware adds no tools and no prompt")
    func inertWhenEmpty() async throws {
        let middleware = ToolSearchMiddleware(auxiliaryTools: [])
        #expect(middleware.tools.isEmpty)

        let recorder = RunRecorder()
        let agent = createAgent(
            model: FakeChatModel(answer: "hi"),
            tools: [EchoTool()],
            systemPrompt: "base",
            middleware: [middleware, RequestRecordingMiddleware(recorder: recorder)]
        )
        _ = await agent.run([.human("hi")]) { _ in }

        let prompts = await recorder.systemPrompts
        #expect(prompts.first == "base") // no "Finding more tools" section
        let sets = await recorder.toolNameSets
        #expect(sets.first == ["echo"]) // nothing stripped, nothing added
    }

    @Test("renderedTools drives the prompt-overhead estimate, not the full tool list")
    func renderedToolsExcludesHidden() {
        let agent = createAgent(
            model: FakeChatModel(answer: "hi"),
            tools: [EchoTool(), StubNamedTool("grep")],
            middleware: [ToolSearchMiddleware(auxiliaryTools: [StubNamedTool("grep")])]
        )
        let names = agent.renderedTools.map(\.name)

        #expect(names.contains("echo"))
        #expect(!names.contains("grep"))
        #expect(agent.tools.contains { $0.name == "grep" }) // still dispatchable
    }
}

@Suite("Auxiliary guidance is withheld with the schemas")
struct ToolSearchGuidanceTests {
    /// The composed system prompt of one round, with `apple_notes` tiered auxiliary.
    private func prompt(auxiliary: Bool) async -> String {
        let recorder = RunRecorder()
        let notes = AppleNotesMiddleware()
        let policy = AgentToolPolicy(
            toolSearch: auxiliary, auxiliaryMiddleware: auxiliary ? ["apple_notes"] : []
        )
        let agent = createDeepAgent(
            model: FakeChatModel(answer: "hi"),
            middleware: [notes, RequestRecordingMiddleware(recorder: recorder)],
            includeFilesystem: false,
            includeGeneralPurpose: false,
            summarization: nil,
            auxiliaryToolNames: policy.expand().auxiliaryToolNames
        )
        _ = await agent.run([.human("which apple notes tools do you have?")]) { _ in }
        return await recorder.systemPrompts.first.flatMap { $0 } ?? ""
    }

    @Test("Tiering a toolset auxiliary removes its prompt guidance, not just its schemas")
    func guidanceGoesWithTheSchemas() async {
        // The bug this guards: stripping `request.tools` left every middleware's prose in place, so the
        // prompt still said "You can read and write the user's Apple Notes - never claim you can't" and
        // "to save a note you must call `create_note`" while no such schema was rendered. The model
        // recited the tools it had been told about and then reached for an unrelated tool when asked to
        // use one.
        let text = await prompt(auxiliary: true)

        #expect(!text.contains("never claim you can't"))
        #expect(!text.contains("Notes are matched by title"))
        #expect(!text.contains("prefer \"append\""))
    }

    @Test("The same toolset kept core keeps its guidance")
    func coreKeepsGuidance() async {
        // The gate must be about *rendering*, not about the middleware: with the feature off, nothing
        // changes for anyone.
        let text = await prompt(auxiliary: false)
        #expect(text.contains("never claim you can't"))
    }

    @Test("Auxiliary tools are still named in the index, grouped by toolset")
    func indexNamesTheTools() async {
        // Names are the whole point of the index - without them the agent cannot answer "which Apple
        // Notes tools do you have?" or call one directly.
        let text = await prompt(auxiliary: true)

        #expect(text.contains("Finding more tools"))
        #expect(text.contains("Apple Notes: create_note, list_notes, read_note, update_note"))
    }

    @Test("The index is far cheaper than the prose it replaces")
    func indexIsCheaperThanProse() async {
        let auxiliaryPrompt = await prompt(auxiliary: true)
        let corePrompt = await prompt(auxiliary: false)
        #expect(auxiliaryPrompt.count < corePrompt.count)
    }
}

@Suite("Auxiliary tools stay gated")
struct ToolSearchApprovalTests {
    /// Records what the approval gate was asked about.
    private actor ApprovalScript {
        private let decision: ToolApprovalDecision
        private(set) var toolNames: [String] = []

        init(_ decision: ToolApprovalDecision) { self.decision = decision }

        func decide(_ request: ToolApprovalRequest) -> ToolApprovalDecision {
            toolNames.append(request.toolName)
            return decision
        }

        nonisolated var handler: ToolApprovalHandler { { await self.decide($0) } }
    }

    /// An agent whose only tool is auxiliary `echo`, gated behind the approval handler.
    private func run(
        call: AgentToolCall, decision: ToolApprovalDecision
    ) async -> (approved: [String], results: [String]) {
        let script = ApprovalScript(decision)
        let agent = createDeepAgent(
            model: FakeChatModel(answer: "done", toolCalls: [call]),
            tools: [EchoTool()],
            interruptOn: ["echo": InterruptOnConfig()],
            approvalHandler: script.handler,
            includeFilesystem: false,
            includeGeneralPurpose: false,
            summarization: nil,
            auxiliaryToolNames: ["echo"]
        )
        let sink = ToolEventSink()
        _ = await agent.run([.human("echo hi")]) { event in
            switch event {
            case .toolCompleted(let name, let result, _, _, _): sink.record(name: name, result: result)
            case .toolFailed(let name, let error, _): sink.record(name: name, result: error)
            default: break
            }
        }
        return await (script.toolNames, sink.completions.map(\.result))
    }

    private var directCall: AgentToolCall {
        AgentToolCall(name: "echo", arguments: ["text": .string("hi")])
    }

    private var runToolCall: AgentToolCall {
        AgentToolCall(name: "run_tool", arguments: [
            "name": .string("echo"), "arguments": .object(["text": .string("hi")])
        ])
    }

    @Test("A tool called by name is still put to the approval gate")
    func directCallIsGated() async {
        let (approved, _) = await run(call: directCall, decision: .approve)
        #expect(approved == ["echo"]) // the tier is about prefill, not permission
    }

    @Test("A tool reached through run_tool is gated under its own name")
    func runToolCallIsGated() async {
        // The load-bearing check for the rewrite. Had `run_tool` executed the target itself, the inner
        // call would have run inside `run_tool`'s own `wrapToolCall` chain and bypassed this gate
        // entirely - an auxiliary write executing with no approval. It must be `echo` that is asked
        // about, not `run_tool`.
        let (approved, _) = await run(call: runToolCall, decision: .approve)
        #expect(approved == ["echo"])
    }

    @Test("Rejecting an auxiliary call through run_tool stops it running")
    func runToolRejectionIsHonoured() async {
        let (approved, results) = await run(call: runToolCall, decision: .reject(message: "no"))
        #expect(approved == ["echo"])
        // The tool never ran, so no echo output - the rejection is what came back.
        #expect(!results.contains { $0.contains("echo: hi") })
    }
}

@Suite("run_tool rewriting")
struct RunToolRewriteTests {
    private let auxiliary: Set<String> = ["grep", "write_file"]

    @Test("An object payload becomes the target's arguments, keeping the call id")
    func objectPayload() {
        let id = UUID()
        let message = AgentMessage.ai("", toolCalls: [
            AgentToolCall(id: id, name: "run_tool", arguments: [
                "name": .string("grep"),
                "arguments": .object(["pattern": .string("todo"), "ignore_case": .bool(true)])
            ])
        ])
        let rewritten = ToolSearchMiddleware.rewritingRunToolCalls(
            message, auxiliaryToolNames: auxiliary
        )

        #expect(rewritten.toolCalls.count == 1)
        #expect(rewritten.toolCalls[0].name == "grep")
        // The id must survive: the tool result answers this call by id.
        #expect(rewritten.toolCalls[0].id == id)
        #expect(rewritten.toolCalls[0].arguments["pattern"] == .string("todo"))
        #expect(rewritten.toolCalls[0].arguments["ignore_case"] == .bool(true))
    }

    @Test("A stringified-JSON payload is parsed")
    func stringPayload() {
        // Small models routinely emit the nested object as a string; accepting both spellings costs
        // one branch and saves a wasted round.
        let message = AgentMessage.ai("", toolCalls: [
            AgentToolCall(name: "run_tool", arguments: [
                "name": .string("grep"),
                "arguments": .string(#"{"pattern": "todo"}"#)
            ])
        ])
        let rewritten = ToolSearchMiddleware.rewritingRunToolCalls(
            message, auxiliaryToolNames: auxiliary
        )

        #expect(rewritten.toolCalls[0].name == "grep")
        #expect(rewritten.toolCalls[0].arguments["pattern"] == .string("todo"))
    }

    @Test("A call naming an unknown tool is left for run_tool's own error")
    func unknownTarget() {
        let message = AgentMessage.ai("", toolCalls: [
            AgentToolCall(name: "run_tool", arguments: ["name": .string("nope")])
        ])
        let rewritten = ToolSearchMiddleware.rewritingRunToolCalls(
            message, auxiliaryToolNames: auxiliary
        )

        // Rewriting to a nonexistent tool would produce a bare "unknown tool"; leaving it alone lets
        // `run_tool.execute` explain what to do instead.
        #expect(rewritten.toolCalls[0].name == "run_tool")
    }

    @Test("Messages without a run_tool call are untouched")
    func passthrough() {
        let message = AgentMessage.ai("", toolCalls: [AgentToolCall(name: "grep", arguments: [:])])
        let rewritten = ToolSearchMiddleware.rewritingRunToolCalls(
            message, auxiliaryToolNames: auxiliary
        )
        #expect(rewritten.toolCalls[0].name == "grep")
    }
}

@Suite("search_tools results")
struct SearchToolsResultTests {
    private var corpus: [ToolDocument] {
        ToolDocument.corpus(for: [
            StubNamedTool("git_log"), StubNamedTool("read_clipboard"), EchoTool()
        ])
    }

    private func result(query: String) async throws -> String {
        let tool = SearchToolsTool(documents: corpus, retriever: LexicalToolRetriever(), limit: 5)
        let output = try await tool.execute(["query": .string(query)], ToolContext())
        return output.content
    }

    @Test("The result carries a callable signature and the toolsets searched")
    func rendersSignatures() async throws {
        let text = try await result(query: "echo text")

        #expect(text.contains("echo(text: string)")) // enough to call it without another lookup
        #expect(text.contains("Searched toolsets:"))
    }

    @Test("A miss still names the toolsets and invites another search")
    func missPolicy() async throws {
        // Never answer a search with a dead end: the model must be able to see the space it drew from
        // and know that re-searching is allowed.
        let text = try await result(query: "zzzz qqqq")

        #expect(text.contains("search_tools again") || text.contains("different words"))
    }

    @Test("The result stays inside the tool-result truncation limit")
    func withinBudget() async throws {
        // ReactAgent truncates a tool result at `maxToolResultCharacters`; a cut mid-signature would
        // leave the model with half a tool name.
        let many = (0 ..< 40).map { StubNamedTool("tool_number_\($0)") }
        let tool = SearchToolsTool(
            documents: ToolDocument.corpus(for: many), retriever: LexicalToolRetriever(), limit: 40
        )
        let output = try await tool.execute(["query": .string("tool")], ToolContext())

        #expect(output.content.count < ReactAgent.maxToolResultCharacters)
    }

    @Test("A missing query is reported, not searched for")
    func requiresQuery() async throws {
        let tool = SearchToolsTool(documents: corpus, retriever: LexicalToolRetriever(), limit: 5)
        let output = try await tool.execute([:], ToolContext())
        #expect(output.content.contains("required"))
    }
}

@Suite("LexicalToolRetriever")
struct LexicalToolRetrieverTests {
    private let corpus = ToolDocument.corpus(for: [
        StubNamedTool("git_log"), StubNamedTool("read_clipboard"), StubNamedTool("take_screenshot"),
        StubNamedTool("write_file")
    ])

    @Test("A query about the clipboard ranks the clipboard tool first")
    func ranksByRarity() async throws {
        let matches = try await LexicalToolRetriever().search(
            "what is on my clipboard", in: corpus, limit: 3
        )
        #expect(matches.first?.name == "read_clipboard")
    }

    @Test("A query naming the tool wins outright")
    func exactName() async throws {
        let matches = try await LexicalToolRetriever().search("git_log", in: corpus, limit: 2)
        #expect(matches.first?.name == "git_log")
    }

    @Test("Ranking is stable for equal scores")
    func deterministic() async throws {
        let first = try await LexicalToolRetriever().search("file", in: corpus, limit: 4)
        let second = try await LexicalToolRetriever().search("file", in: corpus, limit: 4)
        #expect(first.map(\.name) == second.map(\.name))
    }

    @Test("An empty corpus or a zero limit returns nothing")
    func degenerate() async throws {
        #expect(try await LexicalToolRetriever().search("x", in: [], limit: 3).isEmpty)
        #expect(try await LexicalToolRetriever().search("x", in: corpus, limit: 0).isEmpty)
    }
}
