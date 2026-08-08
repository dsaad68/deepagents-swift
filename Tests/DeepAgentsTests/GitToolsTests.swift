@testable import DeepAgents
@testable import DeepAgentsMacTools
@testable import DeepAgentsMLX
import Foundation
import MLXLMCommon
import Testing

/// The read-only `git` tools over a throwaway repository, plus a clean message when the
/// folder isn't a repo.
struct GitToolsTests {
    struct GitSetupError: Error, CustomStringConvertible {
        let description: String
    }

    /// Run a git command for the fixture, and **fail loudly if it didn't work**. Ignoring the exit
    /// status let a failed `git commit` in `withRepo` surface much later as a puzzling assertion
    /// ("your current branch 'main' does not have any commits yet") in whichever test happened to
    /// draw the broken repo. Both pipes are drained before waiting, so a chatty git can't fill one
    /// and deadlock the test run.
    @discardableResult
    private func git(_ arguments: [String], in dir: URL) throws -> String {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/git")
        process.arguments = ["-C", dir.path] + arguments
        let output = Pipe(), errors = Pipe()
        process.standardOutput = output
        process.standardError = errors
        try process.run()
        let out = output.fileHandleForReading.readDataToEndOfFile()
        let err = errors.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        guard process.terminationStatus == 0 else {
            throw GitSetupError(description:
                "git \(arguments.joined(separator: " ")) failed (\(process.terminationStatus)): "
                    + (String(data: err, encoding: .utf8) ?? ""))
        }
        return String(data: out, encoding: .utf8) ?? ""
    }

    private func withRepo<T>(_ body: (WorkspaceRoot, URL) async throws -> T) async throws -> T {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("mispher-git-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        try git(["init", "-q"], in: dir)
        try git(["config", "user.email", "t@e.st"], in: dir)
        try git(["config", "user.name", "Test"], in: dir)
        // A fixture repo inherits the developer's global config, so on a machine with commit
        // signing on every fixture commit shells out to gpg - which fails intermittently when the
        // suite's tests commit concurrently ("gpg failed to sign the data"). Nothing here is signed.
        try git(["config", "commit.gpgsign", "false"], in: dir)
        try "hello\n".write(to: dir.appendingPathComponent("README.md"), atomically: true, encoding: .utf8)
        try git(["add", "."], in: dir)
        try git(["commit", "-q", "-m", "initial"], in: dir)
        return try await body(WorkspaceRoot(rootURL: dir), dir)
    }

    @Test func statusOnCleanRepoHasNoError() async throws {
        try await withRepo { root, _ in
            let output = try await GitStatusTool(root: root).execute([:], ToolContext())
            #expect(!output.content.contains("Error"))
        }
    }

    // MARK: - Results that say what they are

    /// A clean tree used to answer with a bare `## main` - 8 characters that read like a markdown
    /// heading, which a 2.6B planner responded to by calling `git_status` five rounds running.
    @Test func aCleanTreeSaysItIsClean() async throws {
        try await withRepo { root, _ in
            let output = try await GitStatusTool(root: root).execute([:], ToolContext())
            #expect(output.content.contains("working tree is clean"))
            #expect(output.content.contains("On branch"))
            #expect(!output.content.hasPrefix("## "))
        }
    }

    /// A dirty tree keeps git's own short format - the model needs the file list verbatim - but
    /// behind a line that states the branch and the count.
    @Test func aDirtyTreeSummarizesThenListsTheFiles() async throws {
        try await withRepo { root, dir in
            try "hello\nworld\n".write(
                to: dir.appendingPathComponent("README.md"), atomically: true, encoding: .utf8
            )
            let output = try await GitStatusTool(root: root).execute([:], ToolContext())
            #expect(output.content.contains("1 changed file(s)"))
            #expect(output.content.contains("modified: README.md"))
        }
    }

    /// git's two-column codes are terse enough to be mistaken for part of the path: observed
    /// on-device, a model fed ` m deepagents-swift` straight back to `git_diff` as a `path` and git
    /// rejected it as an ambiguous argument.
    @Test func porcelainCodesAreSpelledOut() {
        #expect(GitTools.describeEntry(" M README.md") == "modified: README.md")
        #expect(GitTools.describeEntry("M  README.md") == "modified (staged): README.md")
        #expect(GitTools.describeEntry(" m deepagents-swift") == "modified: deepagents-swift")
        #expect(GitTools.describeEntry("?? notes.txt") == "untracked: notes.txt")
        #expect(GitTools.describeEntry("A  new.swift") == "added (staged): new.swift")
        #expect(GitTools.describeEntry(" D gone.swift") == "deleted: gone.swift")
        #expect(GitTools.describeEntry("R  old.swift -> new.swift") == "renamed (staged): old.swift -> new.swift")
        // A code we don't know keeps its line rather than being reworded into a guess.
        #expect(GitTools.describeEntry("XY strange.txt") == "XY strange.txt")
    }

    @Test func anEmptyDiffSaysWhichDiffWasEmpty() async throws {
        try await withRepo { root, _ in
            let unstaged = try await GitDiffTool(root: root).execute([:], ToolContext())
            #expect(unstaged.content == "No unstaged changes.")

            let staged = try await GitDiffTool(root: root)
                .execute(["staged": .bool(true)], ToolContext())
            #expect(staged.content == "No staged changes.")

            let scoped = try await GitDiffTool(root: root)
                .execute(["path": .string("README.md")], ToolContext())
            #expect(scoped.content == "No unstaged changes under \"README.md\".")
        }
    }

    /// `nothingToReport` carries a default, so dropping the argument at a call site would compile
    /// and silently restore the bare "(no output)". These pin the two paths that would rot quietly.
    @Test func anEmptyLogAndBlameSayWhatIsMissing() async throws {
        try await withRepo { root, dir in
            // A path git has no history for (an empty log with a zero exit; a repo with no commits
            // at all fails inside git instead, and takes the error path).
            try "x\n".write(to: dir.appendingPathComponent("fresh.txt"), atomically: true, encoding: .utf8)
            let log = try await GitLogTool(root: root)
                .execute(["path": .string("fresh.txt")], ToolContext())
            #expect(log.content == "No commits touch \"fresh.txt\".")

            // An empty tracked file blames to nothing, with a zero exit.
            try "".write(to: dir.appendingPathComponent("empty.txt"), atomically: true, encoding: .utf8)
            try git(["add", "."], in: dir)
            try git(["commit", "-q", "-m", "empty"], in: dir)
            let blame = try await GitBlameTool(root: root)
                .execute(["path": .string("empty.txt")], ToolContext())
            #expect(blame.content.contains("nothing to blame"))
        }
    }

    /// The branch line's ahead/behind counts are worth keeping when git prints them.
    @Test func trackingInformationSurvivesTheRewrite() {
        let clean = GitTools.describeStatus("## main...origin/main [ahead 2]")
        #expect(clean.contains("On branch main [ahead 2]"))
        #expect(clean.contains("working tree is clean"))
    }

    /// Anything that isn't the `--short --branch` shape (an error string, say) passes through
    /// untouched rather than being reworded into a status that never happened.
    @Test func anUnexpectedShapePassesThrough() {
        let error = "Error: /tmp/x is not a git repository."
        #expect(GitTools.describeStatus(error) == error)
    }

    @Test func logShowsTheCommit() async throws {
        try await withRepo { root, _ in
            let output = try await GitLogTool(root: root).execute([:], ToolContext())
            #expect(output.content.contains("initial"))
        }
    }

    @Test func diffShowsUnstagedChange() async throws {
        try await withRepo { root, dir in
            try "hello\nworld\n".write(
                to: dir.appendingPathComponent("README.md"), atomically: true, encoding: .utf8
            )
            let output = try await GitDiffTool(root: root).execute([:], ToolContext())
            #expect(output.content.contains("+world"))
        }
    }

    @Test func nonRepositoryReportsCleanly() async throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("mispher-nogit-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let output = try await GitStatusTool(root: WorkspaceRoot(rootURL: dir)).execute([:], ToolContext())
        #expect(output.content.contains("not a git repository"))
    }
}
