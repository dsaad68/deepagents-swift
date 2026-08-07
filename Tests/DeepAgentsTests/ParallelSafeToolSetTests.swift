@testable import DeepAgents
@testable import DeepAgentsMacTools
import Foundation
import Testing

/// Which shipped tools declare ``AgentTool/isParallelSafe``, pinned as an exact set.
///
/// The declaration is load-bearing in a way nothing else checks: marking a tool that writes - or
/// one that needs an earlier call's result - hands it a state snapshot taken before its siblings
/// ran and lets it execute alongside them, silently breaking the round-ordering guarantee
/// `ReactLoopTests` exists to protect. Nothing about that failure is loud: the tool still returns
/// a result, just computed against the wrong state. So the set is asserted whole, and adding or
/// removing a declaration has to be a deliberate edit here too.
struct ParallelSafeToolSetTests {
    /// Every tool the hosts can put in front of a planner: the deep-agent factory's own (todos,
    /// filesystem, `task`) plus each capability toolset, wired the way `RippleDeepAgent` wires them.
    private func allShippedTools() -> [any AgentTool] {
        let root = WorkspaceRoot(rootURL: FileManager.default.temporaryDirectory)
        let agent = createDeepAgent(model: FakeChatModel(answer: "x"), subagents: [])
        let toolsets: [any AgentMiddleware] = [
            SearchToolsMiddleware(root: root),
            TextToolsMiddleware(root: root),
            GitToolsMiddleware(root: root),
            ShellToolsMiddleware(root: root),
            WebToolsMiddleware(),
            UtilityMiddleware(),
            ClipboardMiddleware(),
            MacToolsMiddleware(root: root),
            AppleNotesMiddleware(),
            ScreenshotMiddleware(attachToConversation: false)
        ]
        return agent.tools + toolsets.flatMap(\.tools)
    }

    /// The read-only tools. Each reads and writes nothing another call in the round could care
    /// about, so a batch of them is free of ordering.
    private static let expected: Set<String> = [
        "ls", "read_file",
        "grep", "glob", "tree",
        "head", "tail", "diff",
        "fetch",
        "git_status", "git_diff", "git_log", "git_show", "git_blame",
        "current_datetime", "calculator",
        "mdfind", "read_clipboard"
    ]

    @Test func exactlyTheReadOnlyToolsDeclareThemselvesParallelSafe() {
        let tools = allShippedTools()
        let declared = Set(tools.filter(\.isParallelSafe).map(\.name))

        #expect(declared == Self.expected)
        // Guard the direction that actually corrupts a round: something that mutates fanning out.
        let mutating = ["write_file", "edit_file", "mkdir", "write_todos", "task", "shell",
                        "write_clipboard", "take_screenshot", "create_note", "update_note",
                        "curl", "open", "open_app", "download", "say", "notify"]
        for name in mutating where tools.contains(where: { $0.name == name }) {
            #expect(!declared.contains(name), "\(name) mutates and must stay in the serial order")
        }
    }

    /// The declaration must never reach the model: it is a dispatch property, not something the
    /// planner should see or reason about. (It is also why flipping it cannot change a prompt.)
    @Test func theDeclarationIsAbsentFromTheToolSchema() throws {
        let tool = ReadFileTool(backend: StateBackend())
        #expect(tool.isParallelSafe)

        let schema = tool.toolSchema()
        let data = try JSONSerialization.data(withJSONObject: schema)
        let json = try #require(String(data: data, encoding: .utf8))
        #expect(!json.lowercased().contains("parallel"))
    }
}
