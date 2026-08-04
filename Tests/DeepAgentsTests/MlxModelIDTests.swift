@testable import DeepAgentsMLX
import Foundation
import Testing

/// ``MlxModelID`` - splitting a catalog id into its Hugging Face repo and, for a conversion that
/// packs several precisions into one repo, the subfolder holding this one. The strictness matters:
/// upstream's `Repo.ID(rawValue:)` splits on the first slash only, so a malformed id is not rejected
/// there - it silently becomes a bogus repo that 404s. These tests pin the parse that prevents it.
struct MlxModelIDTests {
    @Test("A two-component id is a plain repo id with no subfolder")
    func plainRepoID() {
        let id = "LiquidAI/LFM2.5-1.2B-Instruct-MLX-8bit"
        let split = MlxModelID.split(id)
        #expect(split.repo == id)
        #expect(split.subfolder == nil)
    }

    @Test("A three-component id splits into repo + precision subfolder")
    func subfolderID() {
        let split = MlxModelID.split("LiquidAI/LFM2.5-2.6B-MLX/mxfp4")
        #expect(split.repo == "LiquidAI/LFM2.5-2.6B-MLX")
        #expect(split.subfolder == "mxfp4")
    }

    @Test("Malformed and relative ids are treated as plain repo ids, never as subfolders")
    func degenerateIDsNeverYieldASubfolder() {
        // Anything that isn't exactly org/repo/subfolder with real components falls back to the
        // whole string, so it fails loudly at the Hub rather than resolving somewhere unexpected.
        for id in ["a/b/c/d", "a//c", "a/b/", "/b/c", "a/b/.", "a/b/..", "noslash", ""] {
            let split = MlxModelID.split(id)
            #expect(split.repo == id, "\(id) should stay whole")
            #expect(split.subfolder == nil, "\(id) should not yield a subfolder")
        }
    }

    @Test("Every catalog row's repo id is a well-formed org/name pair")
    func catalogRepoIDsAreWellFormed() {
        for model in MlxModel.catalog {
            let components = model.repoId.split(separator: "/")
            #expect(components.count == 2, "\(model.id) resolves to a malformed repo \(model.repoId)")
        }
    }

    @Test("Only the LFM2.5-2.6B rows carry a subfolder")
    func onlyTheSubfolderPackagedRowsCarryOne() {
        // The guard that stops a malformed id landing in the catalog later: a row gains a subfolder
        // only by being packaged that way, never by accident.
        let expected = Set([
            "LiquidAI/LFM2.5-2.6B-MLX/mxfp4",
            "LiquidAI/LFM2.5-2.6B-MLX/mxfp8",
            "LiquidAI/LFM2.5-2.6B-MLX/8bit",
            "LiquidAI/LFM2.5-2.6B-MLX/bf16"
        ])
        let nested = Set(MlxModel.catalog.filter { $0.subfolder != nil }.map(\.id))
        #expect(nested == expected)
        for model in MlxModel.catalog where expected.contains(model.id) {
            #expect(model.repoId == "LiquidAI/LFM2.5-2.6B-MLX")
        }
    }
}
