@testable import DeepAgents
@testable import DeepAgentsMacTools
@testable import DeepAgentsMLX
import Foundation
import MLXLMCommon
import Testing

/// The read-only `git` tools over a throwaway repository, plus a clean message when the
/// folder isn't a repo.
struct GitToolsTests {
    @discardableResult
    private func git(_ arguments: [String], in dir: URL) throws -> String {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/git")
        process.arguments = ["-C", dir.path] + arguments
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = Pipe()
        try process.run()
        process.waitUntilExit()
        return String(bytes: pipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""
    }

    private func withRepo<T>(_ body: (WorkspaceRoot, URL) async throws -> T) async throws -> T {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("mispher-git-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        try git(["init", "-q"], in: dir)
        try git(["config", "user.email", "t@e.st"], in: dir)
        try git(["config", "user.name", "Test"], in: dir)
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
