@testable import DeepAgents
import Foundation
import Testing

/// The agent is told where it is.
///
/// Nothing used to state the working folder - the prompts said "your working folder" and the tools
/// resolved paths against a root the model could not see. So it guessed, and a guess outside the
/// root is a refused call and a wasted round. Across 47 on-device runs, 30 contained
/// `outside the allowed folder` errors, 148 refusals in total: a model on macOS reaching for the
/// Linux path `/home/user`, and one reaching for `~/GitHub/mispher` because that is the project's
/// name, while it was running in a worktree called `mispher.parallel-tool-calls`.
struct WorkspaceLocationPromptTests {
    private func prompt(root: URL?) -> String {
        DeepAgentPrompt.system(workspaceRoot: root)
    }

    @Test func thePromptNamesTheWorkingFolder() {
        let root = URL(fileURLWithPath: "/Users/someone/Code/my-worktree")
        let text = prompt(root: root)
        #expect(text.contains("/Users/someone/Code/my-worktree"))
    }

    /// The guidance has to say what happens to a path outside the root, otherwise naming the folder
    /// only tells the model where it is and not that guessing elsewhere fails.
    @Test func thePromptSaysWhatHappensOutsideIt() {
        let text = prompt(root: URL(fileURLWithPath: "/tmp/x"))
        #expect(text.contains("refused"))
        #expect(text.contains("relative"))
    }

    /// With the in-memory scratch filesystem there is no location to name, and inventing one would
    /// be worse than saying nothing.
    @Test func noLocationIsClaimedWithoutARealRoot() {
        let text = prompt(root: nil)
        #expect(!text.contains("Your working folder is"))
    }

    /// The wiring: an agent built on a real-disk backend carries the location, one on the scratch
    /// filesystem does not. This is what actually reaches the model.
    @Test func aLocalBackendPutsTheRootInTheRunningAgentsPrompt() async {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("mispher-root-\(UUID().uuidString)", isDirectory: true)
        try? FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        let recorder = RunRecorder()
        let agent = createDeepAgent(
            model: FakeChatModel(answer: "done"),
            middleware: [RequestRecordingMiddleware(recorder: recorder)],
            backend: LocalFilesystemBackend(rootURL: root)
        )
        _ = await agent.collect([.human("go")])

        let sent = await recorder.systemPrompts.first.flatMap { $0 } ?? ""
        #expect(sent.contains(root.path))
    }

    @Test func theScratchFilesystemAgentNamesNoRoot() async {
        let recorder = RunRecorder()
        let agent = createDeepAgent(
            model: FakeChatModel(answer: "done"),
            middleware: [RequestRecordingMiddleware(recorder: recorder)]
        )
        _ = await agent.collect([.human("go")])

        let sent = await recorder.systemPrompts.first.flatMap { $0 } ?? ""
        #expect(!sent.contains("Your working folder is"))
    }
}
