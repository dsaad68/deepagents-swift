@testable import DeepAgentsMLX
import Foundation
import Testing

/// ``MlxModelLoader/sidecarFilteredView(snapshot:viewRoot:)`` - the symlink view that hides
/// non-indexed sidecar weights (mtp.safetensors / optiq_vision.safetensors in the Qwen3.6 OptiQ
/// conversions) from text-factory loads. Without it, mlx-swift-lm reads every safetensors in the
/// folder and the stray `mtp.` keys trip the qwen3_5 norm-shift heuristic, corrupting all norms.
struct SidecarFilteredViewTests {
    private func makeSnapshot(files: [String], indexed: [String]?) throws -> (snapshot: URL, root: URL) {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("sidecar-test-\(UUID().uuidString)", isDirectory: true)
        let snapshot = root.appendingPathComponent("snapshots/abc123", isDirectory: true)
        try FileManager.default.createDirectory(at: snapshot, withIntermediateDirectories: true)
        for file in files {
            try Data("stub \(file)".utf8).write(to: snapshot.appendingPathComponent(file))
        }
        if let indexed {
            let index = ["weight_map": Dictionary(uniqueKeysWithValues: indexed.enumerated().map {
                ("tensor\($0.offset)", $0.element)
            })]
            let data = try JSONSerialization.data(withJSONObject: index)
            try data.write(to: snapshot.appendingPathComponent("model.safetensors.index.json"))
        }
        return (snapshot, root)
    }

    @Test("Sidecar safetensors are omitted from the view; everything else is linked")
    func sidecarsFiltered() throws {
        let (snapshot, root) = try makeSnapshot(
            files: ["config.json", "tokenizer.json", "model-00001-of-00001.safetensors",
                    "mtp.safetensors", "optiq_vision.safetensors"],
            indexed: ["model-00001-of-00001.safetensors"]
        )
        defer { try? FileManager.default.removeItem(at: root) }
        let view = try #require(
            try MlxModelLoader.sidecarFilteredView(snapshot: snapshot, viewRoot: root.appendingPathComponent("view"))
        )
        let linked = try Set(FileManager.default.contentsOfDirectory(atPath: view.path))
        #expect(linked.contains("model-00001-of-00001.safetensors"))
        #expect(linked.contains("config.json"))
        #expect(linked.contains("tokenizer.json"))
        #expect(linked.contains("model.safetensors.index.json"))
        #expect(!linked.contains("mtp.safetensors"))
        #expect(!linked.contains("optiq_vision.safetensors"))
        // Links resolve to real content (the loader reads through them).
        let text = try String(contentsOf: view.appendingPathComponent("config.json"), encoding: .utf8)
        #expect(text == "stub config.json")
        // Rebuilding (a second load) refreshes the view without erroring.
        #expect(try MlxModelLoader.sidecarFilteredView(
            snapshot: snapshot, viewRoot: root.appendingPathComponent("view")
        ) != nil)
    }

    @Test("A snapshot with no sidecars (or no index) needs no view")
    func cleanSnapshotsPassThrough() throws {
        let (indexedClean, root1) = try makeSnapshot(
            files: ["config.json", "model-00001-of-00001.safetensors"],
            indexed: ["model-00001-of-00001.safetensors"]
        )
        defer { try? FileManager.default.removeItem(at: root1) }
        #expect(try MlxModelLoader.sidecarFilteredView(
            snapshot: indexedClean, viewRoot: root1.appendingPathComponent("view")
        ) == nil)

        let (noIndex, root2) = try makeSnapshot(files: ["config.json", "model.safetensors"], indexed: nil)
        defer { try? FileManager.default.removeItem(at: root2) }
        #expect(try MlxModelLoader.sidecarFilteredView(
            snapshot: noIndex, viewRoot: root2.appendingPathComponent("view")
        ) == nil)
    }
}
