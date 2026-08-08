@testable import DeepAgents
import Foundation
import Testing

/// The search tools must never report an absence they did not establish.
///
/// `FileWalk` caps how many files it visits. When that cap was reached, `grep` still answered
/// "No matches for /pattern/ under \".\"" - identical to a genuine miss. On this repository that
/// was not hypothetical: `Ripple/build` holds 23,288 files, 96% of everything under the root, so a
/// walk from the root spent its whole budget inside build output and reached no source file at all.
/// `grep isParallelSafe` reported the symbol did not exist while it sat in
/// `Sources/DeepAgents/Tools/AgentTool.swift`, and the agent burned 11 and 18 rounds trying to
/// reconcile that with a question that presumed otherwise.
struct SearchStarvationTests {
    /// A tree with `fileCount` files in a `bulk/` folder, plus one findable needle in `src/`.
    private func withCrowdedTree<T>(
        files fileCount: Int, bulkDirectory: String = "bulk",
        _ body: (WorkspaceRoot) async throws -> T
    ) async throws -> T {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("mispher-starve-\(UUID().uuidString)", isDirectory: true)
        let bulk = dir.appendingPathComponent(bulkDirectory)
        try FileManager.default.createDirectory(at: bulk, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(
            at: dir.appendingPathComponent("src"), withIntermediateDirectories: true
        )
        for index in 0 ..< fileCount {
            try "filler \(index)\n".write(
                to: bulk.appendingPathComponent("f\(index).txt"), atomically: true, encoding: .utf8
            )
        }
        try "let isParallelSafe = true\n".write(
            to: dir.appendingPathComponent("src/needle.swift"), atomically: true, encoding: .utf8
        )
        defer { try? FileManager.default.removeItem(at: dir) }
        return try await body(WorkspaceRoot(rootURL: dir))
    }

    /// The regression, in all three shapes it takes. One tree over the cap, because building
    /// 20,000 files is the slowest thing in this suite and one is enough to drive every assertion.
    @Test func aTruncatedWalkNeverReportsAnAbsenceItDidNotEstablish() async throws {
        try await withCrowdedTree(files: FileWalk.maxFiles + 50) { root in
            let grep = GrepTool(root: root)

            // 1. Nothing found. Either it reached the needle, or it says it stopped early - but
            //    never a bare "No matches", which is the claim it cannot support.
            let missed = try await grep.execute(["pattern": .string("isParallelSafe")], ToolContext())
            let admitsStopping = missed.content.contains("stopped")
                || missed.content.contains("only the first")
            #expect(missed.content.contains("needle.swift") || admitsStopping)
            #expect(!missed.content.hasPrefix("No matches"))

            // 2. Something found, but the walk still gave up. Results are not a licence to stay
            //    quiet about it: what was returned is a partial answer and has to say so.
            let found = try await grep.execute(["pattern": .string("filler")], ToolContext())
            #expect(found.content.contains("f0.txt") || found.content.contains(".txt"))
            #expect(found.content.contains("only the first"))

            // 3. `glob` walks the same tree and had the same trap.
            let globbed = try await GlobTool(root: root)
                .execute(["pattern": .string("needle.swift")], ToolContext())
            let globAdmits = globbed.content.contains("stopped")
                || globbed.content.contains("only the first")
            #expect(globbed.content.contains("needle.swift") || globAdmits)
            #expect(!globbed.content.hasPrefix("No files match"))
        }
    }

    /// Build output is not descended into, so the budget goes to source. This is what makes the
    /// cap reachable only on genuinely enormous trees rather than on any repo that has been built.
    @Test func buildOutputIsSkippedSoTheNeedleIsFound() async throws {
        try await withCrowdedTree(files: 200, bulkDirectory: "build") { root in
            let output = try await GrepTool(root: root)
                .execute(["pattern": .string("isParallelSafe")], ToolContext())
            #expect(output.content.contains("needle.swift"))
            #expect(!output.content.contains("build/"))
        }
    }

    /// A genuine miss still reads as a genuine miss - the fix must not make every empty result
    /// hedge, or "not found" stops meaning anything.
    @Test func arealMissStillSaysNoMatches() async throws {
        try await withCrowdedTree(files: 5) { root in
            let output = try await GrepTool(root: root)
                .execute(["pattern": .string("nothingHasThisName")], ToolContext())
            #expect(output.content.hasPrefix("No matches"))
            #expect(!output.content.contains("stopped"))
        }
    }

    /// `tree` names a skipped folder rather than descending it, so a layout stays readable without
    /// the folder looking like it isn't there.
    @Test func treeNamesSkippedBuildOutputWithoutDescending() async throws {
        try await withCrowdedTree(files: 30, bulkDirectory: "build") { root in
            let output = try await TreeTool(root: root).execute([:], ToolContext())
            #expect(output.content.contains("build/"))
            #expect(output.content.contains("not listed"))
            #expect(!output.content.contains("f1.txt"))
        }
    }
}
