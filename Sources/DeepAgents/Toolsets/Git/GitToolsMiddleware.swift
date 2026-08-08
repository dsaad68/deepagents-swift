import Foundation

/// Git middleware - read-only inspection of the repository in the working folder:
/// `git_status`, `git_diff`, `git_log`, `git_show`, `git_blame`. Every tool runs
/// `git -C <root> …` and never commits, pushes, or mutates the repo.
public struct GitToolsMiddleware: AgentMiddleware {
    let root: WorkspaceRoot

    public init(root: WorkspaceRoot) { self.root = root }

    public var name: String { "git" }
    public var tools: [any AgentTool] {
        [
            GitStatusTool(root: root), GitDiffTool(root: root), GitLogTool(root: root),
            GitShowTool(root: root), GitBlameTool(root: root)
        ]
    }

    public func wrapModelCall(
        _ request: ModelRequest,
        _ handler: (ModelRequest) async throws -> ModelResponse
    ) async throws -> ModelResponse {
        // No guidance for tools whose schemas were withheld this round - see
        // `contributesRenderedTools(to:)`.
        guard contributesRenderedTools(to: request) else { return try await handler(request) }
        let composed = [request.systemPrompt, Self.systemPrompt]
            .compactMap { $0 }
            .joined(separator: "\n\n")
        return try await handler(request.override(systemPrompt: composed))
    }

    public static let systemPrompt = """
    ## Git (read-only) with `git_status` / `git_diff` / `git_log` / `git_show` / `git_blame`
    Inspect the git repository in your working folder: `git_status` for the working-tree \
    state, `git_diff` for unstaged (or staged) changes, `git_log` for recent commits, \
    `git_show` for one commit, and `git_blame` for line-by-line authorship. These only read \
    the repo - they never commit, push, or change anything.
    """
}

/// `git_status`: branch and changed files in the working tree.
public struct GitStatusTool: AgentTool {
    let root: WorkspaceRoot
    public var name: String { "git_status" }
    public var description: String { "Show the git working-tree status (branch and changed files)." }
    /// Every tool here only reads the repository, and `GitTools.run` passes
    /// `--no-optional-locks`, so nothing in a round of them touches the index.
    public var isParallelSafe: Bool { true }

    public func execute(_ arguments: [String: AgentJSON], _ context: ToolContext) async throws -> ToolOutput {
        await ToolOutput(GitTools.describeStatus(GitTools.run(root, ["status", "--short", "--branch"])))
    }
}

/// `git_diff`: unstaged (or staged) changes, optionally limited to one path.
public struct GitDiffTool: AgentTool {
    let root: WorkspaceRoot
    public var name: String { "git_diff" }
    public var description: String {
        "Show git changes as a diff. Set staged to see staged changes; pass path to limit to one file or folder."
    }

    public var isParallelSafe: Bool { true }

    public var parameters: [ToolParameter] {
        [
            .optional("staged", type: .bool, description: "Show staged (index) changes instead of unstaged."),
            .optional("path", type: .string, description: "Limit the diff to this file or folder.")
        ]
    }

    public func execute(_ arguments: [String: AgentJSON], _ context: ToolContext) async throws -> ToolOutput {
        var args = ["diff"]
        let staged = ToolArgs.bool(arguments, "staged")
        if staged { args.append("--staged") }
        var scope = ""
        if let path = ToolArgs.string(arguments, "path") {
            guard let url = try? root.resolve(path) else {
                return ToolOutput("Error: \"\(path)\" is outside the working folder.")
            }
            args += ["--", url.path]
            scope = " under \"\(path)\""
        }
        // An empty diff is a finding, not a blank: say which diff was empty so the model doesn't
        // re-run it looking for the answer it already has.
        let nothing = "No \(staged ? "staged" : "unstaged") changes\(scope)."
        return await ToolOutput(GitTools.run(root, args, nothingToReport: nothing))
    }
}

/// `git_log`: recent commits in one-line form.
public struct GitLogTool: AgentTool {
    let root: WorkspaceRoot
    public var name: String { "git_log" }
    public var description: String { "Show recent commits (one line each)." }
    public var isParallelSafe: Bool { true }

    public var parameters: [ToolParameter] {
        [
            .optional("count", type: .int, description: "How many recent commits (default 20)."),
            .optional("path", type: .string, description: "Limit history to this file or folder.")
        ]
    }

    public func execute(_ arguments: [String: AgentJSON], _ context: ToolContext) async throws -> ToolOutput {
        let count = min(max(1, ToolArgs.int(arguments, "count") ?? 20), 200)
        var args = ["log", "--oneline", "-n", String(count)]
        var scope = ""
        if let path = ToolArgs.string(arguments, "path") {
            guard let url = try? root.resolve(path) else {
                return ToolOutput("Error: \"\(path)\" is outside the working folder.")
            }
            args += ["--", url.path]
            scope = path
        }
        // A repository with no commits at all fails in git rather than returning nothing, so this
        // covers the reachable case: a path git has no history for.
        let nothing = scope.isEmpty
            ? "No commits in this repository yet."
            : "No commits touch \"\(scope)\"."
        return await ToolOutput(GitTools.run(root, args, nothingToReport: nothing))
    }
}

/// `git_show`: a single commit (message + patch).
public struct GitShowTool: AgentTool {
    let root: WorkspaceRoot
    public var name: String { "git_show" }
    public var description: String { "Show one commit: its message and the changes it made." }
    public var isParallelSafe: Bool { true }

    public var parameters: [ToolParameter] {
        [.optional("ref", type: .string, description: "Commit, tag, or ref to show (default HEAD).")]
    }

    public func execute(_ arguments: [String: AgentJSON], _ context: ToolContext) async throws -> ToolOutput {
        let ref = ToolArgs.string(arguments, "ref") ?? "HEAD"
        guard !ToolArgs.looksLikeOption(ref) else { return ToolOutput("Error: invalid ref \"\(ref)\".") }
        return await ToolOutput(GitTools.run(root, ["show", ref]))
    }
}

/// `git_blame`: line-by-line authorship for a file.
public struct GitBlameTool: AgentTool {
    let root: WorkspaceRoot
    public var name: String { "git_blame" }
    public var description: String { "Show line-by-line authorship (last commit per line) for a file." }
    public var isParallelSafe: Bool { true }

    public var parameters: [ToolParameter] {
        [.required("path", type: .string, description: "File to annotate.")]
    }

    public func execute(_ arguments: [String: AgentJSON], _ context: ToolContext) async throws -> ToolOutput {
        guard let path = ToolArgs.string(arguments, "path") else { return ToolOutput("Error: `path` is required.") }
        guard let url = try? root.resolve(path) else {
            return ToolOutput("Error: \"\(path)\" is outside the working folder.")
        }
        return await ToolOutput(GitTools.run(
            root, ["blame", "--", url.path],
            nothingToReport: "\"\(path)\" is empty, so there is nothing to blame."
        ))
    }
}

/// Run a read-only git subcommand in `root`, returning model-ready output or an "Error: …"
/// string (with a clean message when the folder isn't a repository).
enum GitTools {
    /// `git status --short --branch` prints the branch line, then one line per changed file - so a
    /// clean tree yields a bare `## main`, which reads like a stray markdown heading rather than an
    /// answer. Observed on-device: a 2.6B planner re-called `git_status` in five consecutive rounds
    /// because nothing in those 22 characters said "this is the status, and it is clean". Same
    /// lesson as ``ReadClipboardTool``'s JSON envelope: name what the result *is*.
    ///
    /// A dirty tree keeps git's own short format (the model needs the file list verbatim) behind a
    /// summary line, so the branch and the count are stated rather than implied.
    static func describeStatus(_ raw: String) -> String {
        guard raw.hasPrefix("## ") else { return raw } // an error, or a shape we didn't produce
        var lines = raw.split(separator: "\n").map(String.init)
        let header = String(lines.removeFirst().dropFirst(3))
        // `--branch` writes `main...origin/main [ahead 1]`; the branch is everything before "...",
        // and the bracketed ahead/behind counts are worth keeping when git bothered to print them.
        let branch = header.components(separatedBy: "...").first ?? header
        let tracking = header.firstIndex(of: "[").map { " " + String(header[$0...]) } ?? ""
        guard !lines.isEmpty else {
            return "On branch \(branch)\(tracking). The working tree is clean - "
                + "no files added, changed, or deleted."
        }
        return "On branch \(branch)\(tracking) - \(lines.count) changed file(s):\n"
            + lines.map(describeEntry).joined(separator: "\n")
    }

    /// One `--short` entry (`XY path`) as words. The two-column codes are terse enough that a model
    /// reads ` m deepagents-swift` as the path and hands the whole line back to `git_diff` -
    /// observed on-device, where git then rejected it as an ambiguous argument. An unrecognized
    /// code keeps its raw line rather than being guessed at.
    static func describeEntry(_ line: String) -> String {
        guard line.count > 3 else { return line }
        let code = line.prefix(2)
        let path = String(line.dropFirst(3))
        if code == "??" { return "untracked: \(path)" }
        // X is the index (staged) column, Y the working tree; lowercase `m` is a submodule whose
        // contents moved. Report whichever column is set, and say when the change is staged.
        let staged = code.first != " "
        let letter = code.first == " " ? code.last : code.first
        let verb: String
        switch letter {
        case "M", "m": verb = "modified"
        case "A": verb = "added"
        case "D": verb = "deleted"
        case "R": verb = "renamed"
        case "C": verb = "copied"
        case "U": verb = "conflicted"
        default: return line
        }
        return "\(verb)\(staged ? " (staged)" : ""): \(path)"
    }

    /// `nothingToReport` replaces git's silence when a command succeeds with no output - "no staged
    /// changes" is an answer, an empty string is not. See ``describeStatus(_:)``.
    static func run(
        _ root: WorkspaceRoot, _ arguments: [String], nothingToReport: String = "(no output)"
    ) async -> String {
        do {
            // `--no-optional-locks` keeps `status` / `diff` from taking `.git/index.lock` to
            // opportunistically refresh the index. They are read-only either way, but the write
            // is what would make two of them in one round collide - without it a parallel round
            // can surface "Unable to create index.lock" instead of the status the model asked for.
            let result = try await ProcessRunner.run(
                "/usr/bin/git", ["-C", root.rootURL.path, "--no-optional-locks"] + arguments,
                cwd: root.rootURL
            )
            if result.timedOut { return "Error: git timed out." }
            if result.succeeded { return result.stdout.isEmpty ? nothingToReport : result.stdout }
            if result.stderr.lowercased().contains("not a git repository") {
                return "Error: \(root.displayRoot) is not a git repository."
            }
            return "Error: \(result.stderr.isEmpty ? "git exited with status \(result.status)." : result.stderr)"
        } catch {
            return "Error: \(error.localizedDescription)"
        }
    }
}
