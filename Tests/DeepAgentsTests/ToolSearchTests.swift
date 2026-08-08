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

    @Test("The prompt names the auxiliary area, never the tools in it")
    func indexNamesAreasNotTools() async {
        // Areas, not names: knowing a notes capability exists is discoverability, and mapping a request
        // onto `list_notes` is the retriever's job. Names would also cost O(tools) in every prompt -
        // scaling with exactly what lazy loading exists to remove.
        let text = await prompt(auxiliary: true)

        #expect(text.contains("core and auxiliary"))
        #expect(text.contains("Apple Notes (4)"))
        for name in ["list_notes", "read_note", "create_note", "update_note"] {
            #expect(!text.contains(name), "\(name) should be discoverable only through search_tools")
        }
    }

    @Test("The section tells the model when to search, not just how")
    func sectionStatesTheObligation() async {
        // The regression this guards: an earlier version described the mechanism only, and cold
        // "list my apple notes" produced no search - the model read an index of names as reference
        // material. Both of these lines are load-bearing.
        let text = await prompt(auxiliary: true)

        #expect(text.contains("call `search_tools` before you answer"))
        #expect(text.contains("Never tell the user you are unable to do something"))
    }

    @Test("The obligation appears only when there are auxiliary tools")
    func noObligationWhenEverythingIsCore() async {
        let text = await prompt(auxiliary: false)
        #expect(!text.contains("core and auxiliary"))
        #expect(!text.contains("search_tools"))
    }

    @Test("One small toolset auxiliary can cost more prompt than it saves")
    func theSectionIsAFixedCost() async {
        // Worth pinning rather than hiding: the section (tiers, the obligation, the refusal-block) is a
        // fixed ~1.4k charge, while the prose it replaces is *per toolset*. With a single toolset
        // auxiliary the charge exceeds the saving - lazy tools pay off across many toolsets, not one.
        // `ToolSearchWholeOutputTests.theIndexBeatsTheProseAcrossManyToolsets` is the case that matters.
        let auxiliaryPrompt = await prompt(auxiliary: true)
        let corePrompt = await prompt(auxiliary: false)
        #expect(auxiliaryPrompt.count > corePrompt.count)
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

/// Whole-output checks over a realistic agent, rather than checks on one mechanism.
///
/// Both bugs that escaped this suite - auxiliary tools' prose surviving in the prompt, and MCP tools
/// rendering with no arguments - were found by using the feature, not by testing it. Each was a
/// *class* of failure (one middleware's guidance, one kind of tool) that a targeted test could not see.
/// These assert properties over every tool an agent actually assembles, so the next member of either
/// class fails here instead.
@Suite("Nothing leaks, nothing is dropped")
struct ToolSearchWholeOutputTests {
    /// Every catalog capability, all tiered auxiliary, plus a schema-only tool standing in for MCP.
    private struct SchemaOnlyTool: AgentTool {
        var name: String { "server__ask_question" }
        var description: String { "Ask a repository a question." }

        func toolSchema() -> ToolSchema {
            [
                "type": "function",
                "function": [
                    "name": name, "description": description,
                    "parameters": [
                        "type": "object",
                        "properties": [
                            "repoName": ["type": "string", "description": "owner/repo."],
                            "question": ["type": "string", "description": "What to ask."]
                        ] as [String: any Sendable],
                        "required": ["repoName", "question"]
                    ] as [String: any Sendable]
                ] as [String: any Sendable]
            ]
        }

        func execute(
            _ arguments: [String: AgentJSON], _ context: ToolContext
        ) async throws -> ToolOutput { ToolOutput("ok") }
    }

    private func makeAgent() -> (ReactAgent, RunRecorder) {
        let recorder = RunRecorder()
        let root = WorkspaceRoot(rootURL: URL(fileURLWithPath: "/tmp"))
        let middleware: [any AgentMiddleware] = [
            WebToolsMiddleware(), SearchToolsMiddleware(root: root), TextToolsMiddleware(root: root),
            GitToolsMiddleware(root: root), ShellToolsMiddleware(root: root),
            ClipboardMiddleware(), AppleNotesMiddleware(), ScreenshotMiddleware(),
            MacToolsMiddleware(root: root)
        ]
        // Everything the catalog knows about, tiered auxiliary, plus the MCP-shaped tool.
        var policy = AgentToolPolicy(toolSearch: true)
        policy.auxiliaryMiddleware = Set(MiddlewareCatalog.all.map(\.id))
        let auxiliary = policy.expand(extraAuxiliary: ["server__ask_question"]).auxiliaryToolNames
        let agent = createDeepAgent(
            model: FakeChatModel(answer: "hi"),
            tools: [SchemaOnlyTool()],
            middleware: middleware + [RequestRecordingMiddleware(recorder: recorder)],
            includeGeneralPurpose: false,
            summarization: nil,
            auxiliaryToolNames: auxiliary,
            toolsetsByTool: ["server__ask_question": "server"]
        )
        return (agent, recorder)
    }

    @Test("No prompt guidance refers to an auxiliary tool")
    func noAuxiliaryToolLeaksIntoThePrompt() async throws {
        // The general form of the Apple Notes guidance leak. Asserting the absence of three known
        // phrases only covered the middleware that had already misbehaved; this covers every one, and
        // any middleware added later that forgets to gate its guidance.
        //
        // Matched on the **backticked** name, which is how every guidance section in this codebase
        // refers to a tool ("call `read_note`", "## Working files with `ls` / `read_file`"). Matching a
        // bare word would trip over tool names that are ordinary English - `open`, `say`, `head`,
        // `diff`, `shell` - and over the index, which lists names unquoted.
        let (agent, recorder) = makeAgent()
        _ = await agent.run([.human("hi")]) { _ in }
        let prompt = try #require(await recorder.systemPrompts.first.flatMap { $0 })
        let auxiliary = Set(agent.tools.map(\.name))
            .subtracting(agent.renderedTools.map(\.name))
        #expect(auxiliary.count > 20) // the fixture is doing its job

        for name in auxiliary.sorted() {
            #expect(!prompt.contains("`\(name)`"), "prompt still instructs the model to use `\(name)`")
        }
    }

    @Test("No auxiliary tool's name appears in the prompt at all")
    func noAuxiliaryToolNameIsInThePrompt() async throws {
        // Stronger than the backtick check: an auxiliary tool must be reachable *only* through search,
        // so its name should be absent entirely - not merely absent from instructions.
        //
        // Restricted to names containing `_`, which cannot collide with ordinary English. Tool names
        // that are also words (`open`, `say`, `head`, `diff`, `shell`) would match the surrounding prose
        // and make this assert nothing; the underscored majority is enough to catch a regression.
        let (agent, recorder) = makeAgent()
        _ = await agent.run([.human("hi")]) { _ in }
        let prompt = try #require(await recorder.systemPrompts.first.flatMap { $0 })
        let auxiliary = Set(agent.tools.map(\.name))
            .subtracting(agent.renderedTools.map(\.name))
            .filter { $0.contains("_") }
        #expect(auxiliary.count > 15) // the fixture is doing its job

        for name in auxiliary.sorted() {
            #expect(!prompt.contains(name), "\(name) is in the prompt; it should require search_tools")
        }
    }

    @Test("Every auxiliary area is named, so the agent knows what to search for")
    func everyAreaIsNamed() async throws {
        // The other half: withholding the names must not withhold the *existence* of the capability, or
        // the agent has nothing to search for.
        let (agent, recorder) = makeAgent()
        _ = await agent.run([.human("hi")]) { _ in }
        let prompt = try #require(await recorder.systemPrompts.first.flatMap { $0 })

        for area in ["Apple Notes", "Git", "Web", "Search", "Clipboard", "System"] {
            #expect(prompt.contains(area), "\(area) is hidden but never mentioned as an area")
        }
    }

    @Test("No tool silently loses its parameters")
    func everyToolKeepsItsParameters() {
        // The general form of the MCP bug: signatures were derived from a field one whole kind of tool
        // never fills in. Compare against the rendered schema, which is what the model would have seen.
        let (agent, _) = makeAgent()
        for tool in agent.tools {
            let function = tool.toolSchema()["function"] as? [String: any Sendable]
            let parameters = function?["parameters"] as? [String: any Sendable]
            let properties = parameters?["properties"] as? [String: any Sendable] ?? [:]
            let specs = ToolDocument.parameterSpecs(of: tool)

            let specNames = specs.map(\.name).sorted()
            let schemaNames = properties.keys.sorted()
            #expect(
                specNames == schemaNames,
                "\(tool.name): signature has \(specNames), schema has \(schemaNames)"
            )
            // And a tool that takes arguments must not be advertised as taking none.
            if !properties.isEmpty {
                #expect(!ToolDocument.signature(of: tool).hasSuffix("()"), "\(tool.name) rendered empty")
            }
        }
    }

    @Test("Across a real toolset the index is much cheaper than the prose")
    func theIndexBeatsTheProseAcrossManyToolsets() async throws {
        // The claim the feature rests on, measured where it applies: the section is a fixed cost, the
        // per-toolset guidance is not, so the saving grows with the number of auxiliary toolsets. (With
        // exactly one, it loses - see `theSectionIsAFixedCost`.)
        let (auxiliaryAgent, auxiliaryRecorder) = makeAgent()
        _ = await auxiliaryAgent.run([.human("hi")]) { _ in }
        let lazy = try #require(await auxiliaryRecorder.systemPrompts.first.flatMap { $0 })

        let eagerRecorder = RunRecorder()
        let root = WorkspaceRoot(rootURL: URL(fileURLWithPath: "/tmp"))
        let eagerAgent = createDeepAgent(
            model: FakeChatModel(answer: "hi"),
            tools: [SchemaOnlyTool()],
            middleware: [
                WebToolsMiddleware(), SearchToolsMiddleware(root: root), TextToolsMiddleware(root: root),
                GitToolsMiddleware(root: root), ShellToolsMiddleware(root: root),
                ClipboardMiddleware(), AppleNotesMiddleware(), ScreenshotMiddleware(),
                MacToolsMiddleware(root: root),
                RequestRecordingMiddleware(recorder: eagerRecorder)
            ],
            includeGeneralPurpose: false,
            summarization: nil
        )
        _ = await eagerAgent.run([.human("hi")]) { _ in }
        let eager = try #require(await eagerRecorder.systemPrompts.first.flatMap { $0 })

        #expect(lazy.count < eager.count / 2, "lazy \(lazy.count) chars vs eager \(eager.count)")
    }

    @Test("A call built from the signature alone passes schema validation")
    func theSignatureIsEnoughToCallCorrectly() {
        // Closes the loop the search result exists to serve: if the model passes exactly the arguments
        // marked required, with a value from any declared enum, the call must not be bounced by
        // `schemaViolation`. Otherwise the signature is describing something uncallable.
        let (agent, _) = makeAgent()
        for tool in agent.tools {
            var arguments: [String: AgentJSON] = [:]
            for spec in ToolDocument.parameterSpecs(of: tool) where spec.isRequired {
                arguments[spec.name] = Self.sampleValue(for: spec)
            }
            let call = AgentToolCall(name: tool.name, arguments: arguments)
            let violation = ReactAgent.schemaViolation(call, tool: tool)
            #expect(violation == nil, "\(tool.name) rejected its own signature: \(violation ?? "")")
        }
    }

    /// A type-appropriate value, honouring a declared enum - which is exactly what the model has to do
    /// from the signature.
    private static func sampleValue(for spec: ToolParameterSpec) -> AgentJSON {
        if let allowed = spec.allowedValues.first { return .string(allowed) }
        switch spec.type {
        case "int": return .int(1)
        case "number": return .double(1)
        case "bool": return .bool(true)
        case "object": return .object([:])
        default:
            return spec.type.hasPrefix("[") ? .array([.string("x")]) : .string("x")
        }
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

@Suite("Signatures tell the model what is required")
struct ToolSignatureTests {
    /// A built-in tool with a required arg, optional args, and an enum-constrained one.
    private struct NoteTool: AgentTool {
        var name: String { "update_note" }
        var description: String { "Modify an existing note." }
        var parameters: [ToolParameter] {
            [
                .required("title", type: .string, description: "Title of the note to update."),
                .required(
                    "mode", type: .string, description: "How to write.",
                    extraProperties: ["enum": ["append", "replace"]]
                ),
                .optional("index", type: .int, description: "Which match to use."),
                .optional("tags", type: .array(elementType: .string), description: "Tags to set.")
            ]
        }

        func execute(
            _ arguments: [String: AgentJSON], _ context: ToolContext
        ) async throws -> ToolOutput { ToolOutput("ok") }
    }

    /// A tool that publishes a server-style JSON Schema and leaves `parameters` empty - exactly what
    /// ``MCPTool`` does.
    private struct SchemaOnlyTool: AgentTool {
        var name: String { "ask_question" }
        var description: String { "Ask a repo a question." }

        func toolSchema() -> ToolSchema {
            [
                "type": "function",
                "function": [
                    "name": name,
                    "description": description,
                    "parameters": [
                        "type": "object",
                        "properties": [
                            "repoName": ["type": "string", "description": "owner/repo."],
                            "question": ["type": "string", "description": "What to ask."],
                            "depth": ["type": "integer", "description": "How deep."]
                        ] as [String: any Sendable],
                        "required": ["repoName", "question"]
                    ] as [String: any Sendable]
                ] as [String: any Sendable]
            ]
        }

        func execute(
            _ arguments: [String: AgentJSON], _ context: ToolContext
        ) async throws -> ToolOutput { ToolOutput("ok") }
    }

    @Test("Required parameters are marked, and marked on both sides")
    func requiredIsExplicit() {
        // `!` and `?` rather than "bare means required": the model should not have to infer which
        // arguments it may omit, and the search result's legend states the convention.
        let signature = ToolDocument.signature(of: NoteTool())

        #expect(signature.contains("title!: string"))
        #expect(signature.contains("index?: int"))
        #expect(!signature.contains("title:")) // never an unmarked name
    }

    @Test("Required parameters come first, then optional ones alphabetically")
    func orderingIsStableAndReadable() {
        // Schema properties arrive as an unordered dictionary, so without a rule the signature - and
        // the document text the retriever caches - would differ run to run.
        let signature = ToolDocument.signature(of: NoteTool())
        let expected = "update_note(title!: string, mode!: string (\"append\"|\"replace\"), "
            + "index?: int, tags?: [string])"
        #expect(signature == expected)
        #expect(ToolDocument.signature(of: NoteTool()) == signature) // and is stable
    }

    @Test("An enum constraint is rendered inline")
    func enumsAreShown() {
        // A value the model cannot guess, and `schemaViolation` rejects a wrong one - so a wasted round
        // unless it is in the signature.
        #expect(ToolDocument.signature(of: NoteTool()).contains("\"append\"|\"replace\""))
    }

    @Test("A tool that only publishes a JSON Schema still gets a full signature")
    func schemaOnlyToolsAreNotEmpty() {
        // The MCP shape: `toolSchema()` is overridden and `parameters` is left empty. Reading
        // `parameters` rendered every MCP tool as `ask_question()` - taking no arguments at all.
        let specs = ToolDocument.parameterSpecs(of: SchemaOnlyTool())
        #expect(specs.map(\.name) == ["repoName", "question", "depth"])
        #expect(specs.filter(\.isRequired).map(\.name) == ["repoName", "question"])

        let signature = ToolDocument.signature(of: SchemaOnlyTool())
        #expect(signature == "ask_question(repoName!: string, question!: string, depth?: int)")
    }

    @Test("A tool with no parameters renders as taking none")
    func noParameters() {
        #expect(ToolDocument.signature(of: StubNamedTool("git_status")) == "git_status()")
    }

    @Test("The search result states the convention and describes the required arguments")
    func resultExplainsRequiredArguments() async throws {
        let corpus = ToolDocument.corpus(for: [NoteTool(), SchemaOnlyTool()])
        let tool = SearchToolsTool(documents: corpus, retriever: LexicalToolRetriever(), limit: 5)
        let text = try await tool.execute(["query": .string("update a note")], ToolContext()).content

        #expect(text.contains("`arg!` must be passed")) // the legend
        #expect(text.contains("update_note(title!: string"))
        #expect(text.contains("title! - Title of the note to update."))
        // Optional parameters stay name-and-type; the budget is spent where a wrong guess fails.
        #expect(!text.contains("index? - "))
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

        // Enough to call it without another lookup - including which arguments are mandatory, which
        // is the part a bare `text: string` left the model to guess at.
        #expect(text.contains("echo(text!: string)"))
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
