@testable import DeepAgentsMLX
import Foundation
import Testing

/// The Hugging Face cache helpers in ``MlxModelLoader`` once a catalog id can name a precision
/// subfolder of a shared repo. Everything here runs against a fixture directory via the `in:` /
/// `from:` seams, so no test touches the user's real cache or the network.
struct HubCachePathTests {
    private static let repo = "LiquidAI/LFM2.5-2.6B-MLX"

    private func makeFixture() throws -> URL {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("hub-cache-test-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return root
    }

    /// A repo cache directory holding `precisions`, laid out the way the hub does it: every snapshot
    /// entry is a symlink into the content-addressed `blobs/` store, and files byte-identical across
    /// precisions (here `tokenizer.json`) are a *single* shared blob with one link per precision.
    private func makeRepoCache(precisions: [String], in root: URL) throws -> URL {
        let fm = FileManager.default
        let repo = root.appendingPathComponent("models--LiquidAI--LFM2.5-2.6B-MLX", isDirectory: true)
        let blobs = repo.appendingPathComponent("blobs", isDirectory: true)
        let snapshot = repo.appendingPathComponent("snapshots/abc123", isDirectory: true)
        try fm.createDirectory(at: blobs, withIntermediateDirectories: true)
        try fm.createDirectory(at: snapshot, withIntermediateDirectories: true)

        let shared = blobs.appendingPathComponent("sha-shared-tokenizer")
        try Data("shared tokenizer".utf8).write(to: shared)
        for precision in precisions {
            let directory = snapshot.appendingPathComponent(precision, isDirectory: true)
            try fm.createDirectory(at: directory, withIntermediateDirectories: true)
            let weights = blobs.appendingPathComponent("sha-\(precision)-weights")
            try Data("weights for \(precision)".utf8).write(to: weights)
            try fm.createSymbolicLink(
                at: directory.appendingPathComponent("model.safetensors"), withDestinationURL: weights
            )
            try fm.createSymbolicLink(
                at: directory.appendingPathComponent("tokenizer.json"), withDestinationURL: shared
            )
        }
        return repo
    }

    // MARK: - Path resolution

    @Test("A subfolder id and its bare repo resolve to the same cache directory")
    func subfolderIDSharesItsRepoDirectory() {
        // They share one repo, one refs/main, and one blob store - so PrefixKVStore's revision
        // lookup keeps working unchanged for a subfolder id.
        let bare = MlxModelLoader.hubRepoDirectory(Self.repo)
        #expect(MlxModelLoader.hubRepoDirectory("\(Self.repo)/mxfp4") == bare)
        #expect(MlxModelLoader.hubRepoDirectory("\(Self.repo)/bf16") == bare)
        #expect(bare.lastPathComponent == "models--LiquidAI--LFM2.5-2.6B-MLX")
    }

    @Test("Download globs are scoped to the precision subfolder")
    func downloadPatternsScopeToTheSubfolder() {
        // Load-bearing: the hub matches globs with `fnmatch(glob, path, 0)` - no FNM_PATHNAME - so
        // `*` crosses `/`. Unprefixed patterns would match every precision in the repo, turning a
        // 1.6 GB mxfp4 pull into a ~25 GB one.
        #expect(MlxModelLoader.downloadPatterns("LiquidAI/LFM2.5-350M-MLX-8bit")
            == ["*.safetensors", "*.json", "*.jinja"])
        #expect(MlxModelLoader.downloadPatterns("\(Self.repo)/mxfp4")
            == ["mxfp4/*.safetensors", "mxfp4/*.json", "mxfp4/*.jinja"])
    }

    // MARK: - Snapshot resolution

    @Test("The revision in refs/main is offered before any stale snapshot")
    func currentRevisionIsTriedFirst() throws {
        // A load picks the first workable directory, so a stale cached revision winning here would
        // load weights from a different commit than PrefixKVStore keys its KV cache against.
        let root = try makeFixture()
        defer { try? FileManager.default.removeItem(at: root) }
        let fm = FileManager.default
        let repo = root.appendingPathComponent("models--LiquidAI--LFM2.5-2.6B-MLX", isDirectory: true)
        for hash in ["aaa-stale", "zzz-current"] {
            try fm.createDirectory(
                at: repo.appendingPathComponent("snapshots/\(hash)/mxfp4", isDirectory: true),
                withIntermediateDirectories: true
            )
        }
        try fm.createDirectory(at: repo.appendingPathComponent("refs"), withIntermediateDirectories: true)
        try Data("zzz-current\n".utf8).write(to: repo.appendingPathComponent("refs/main"))

        let directories = MlxModelLoader.snapshotDirectories("\(Self.repo)/mxfp4", in: repo)
        #expect(directories.count == 2)
        // Current first, despite sorting after the stale one alphabetically.
        #expect(directories.first?.deletingLastPathComponent().lastPathComponent == "zzz-current")
        // Every entry is the subfolder, never the snapshot root - loadWeights enumerates
        // recursively, so the root would merge both precisions' tensors into one weight dict.
        #expect(directories.allSatisfy { $0.lastPathComponent == "mxfp4" })
    }

    @Test("A plain repo id resolves to snapshot roots, and a missing cache to nothing")
    func plainAndMissingSnapshotResolution() throws {
        let root = try makeFixture()
        defer { try? FileManager.default.removeItem(at: root) }
        let fm = FileManager.default
        let repo = root.appendingPathComponent("models--LiquidAI--LFM2.5-350M-MLX-8bit", isDirectory: true)
        try fm.createDirectory(
            at: repo.appendingPathComponent("snapshots/abc123", isDirectory: true),
            withIntermediateDirectories: true
        )

        let directories = MlxModelLoader.snapshotDirectories("LiquidAI/LFM2.5-350M-MLX-8bit", in: repo)
        #expect(directories.map(\.lastPathComponent) == ["abc123"])
        // No refs/main is fine - ordering is just left alone.
        #expect(MlxModelLoader.snapshotDirectories("x/y", in: root.appendingPathComponent("nope")).isEmpty)
    }

    @Test("Two precisions of one repo get separate sidecar view directories")
    func sidecarViewsDoNotCollideAcrossPrecisions() throws {
        // `sidecarFilteredView` names the view by the snapshot's last path component, which for a
        // nested path is the *subfolder* - so without the hash in the view root both precisions of
        // one repo would build into the same directory and clobber each other's symlinks.
        let root = try makeFixture()
        defer { try? FileManager.default.removeItem(at: root) }
        let fm = FileManager.default
        var views: [URL] = []
        for precision in ["mxfp4", "mxfp8"] {
            let snapshot = root.appendingPathComponent("snapshots/abc123/\(precision)", isDirectory: true)
            try fm.createDirectory(at: snapshot, withIntermediateDirectories: true)
            for file in ["config.json", "model.safetensors", "stray.safetensors"] {
                try Data("stub \(precision) \(file)".utf8).write(to: snapshot.appendingPathComponent(file))
            }
            let index = ["weight_map": ["t0": "model.safetensors"]]
            try JSONSerialization.data(withJSONObject: index)
                .write(to: snapshot.appendingPathComponent("model.safetensors.index.json"))
            let view = try #require(try MlxModelLoader.sidecarFilteredView(
                snapshot: snapshot,
                viewRoot: root.appendingPathComponent("views/\(precision)", isDirectory: true)
            ))
            views.append(view)
        }
        #expect(views[0] != views[1])
        // Each view resolves to its own precision's bytes, and drops the non-indexed sidecar.
        for (view, precision) in zip(views, ["mxfp4", "mxfp8"]) {
            let text = try String(contentsOf: view.appendingPathComponent("config.json"), encoding: .utf8)
            #expect(text == "stub \(precision) config.json")
            #expect(!fm.fileExists(atPath: view.appendingPathComponent("stray.safetensors").path))
        }
    }

    // MARK: - Downloaded-on-disk

    @Test("A downloaded precision never makes its siblings look downloaded")
    func siblingPrecisionsAreTrackedSeparately() throws {
        let root = try makeFixture()
        defer { try? FileManager.default.removeItem(at: root) }
        let repo = try makeRepoCache(precisions: ["bf16"], in: root)

        #expect(MlxModelLoader.hasWeights("\(Self.repo)/bf16", in: repo))
        #expect(!MlxModelLoader.hasWeights("\(Self.repo)/mxfp4", in: repo))
        #expect(!MlxModelLoader.hasWeights("\(Self.repo)/8bit", in: repo))
    }

    @Test("A plain repo id still reports its snapshot weights")
    func plainRepoIDIsUnaffected() throws {
        let root = try makeFixture()
        defer { try? FileManager.default.removeItem(at: root) }
        let fm = FileManager.default
        let repo = root.appendingPathComponent("models--LiquidAI--LFM2.5-350M-MLX-8bit", isDirectory: true)
        let snapshot = repo.appendingPathComponent("snapshots/abc123", isDirectory: true)
        try fm.createDirectory(at: snapshot, withIntermediateDirectories: true)
        try Data("w".utf8).write(to: snapshot.appendingPathComponent("model.safetensors"))

        #expect(MlxModelLoader.hasWeights("LiquidAI/LFM2.5-350M-MLX-8bit", in: repo))
    }

    // MARK: - Removal

    @Test("Removing one precision frees its blobs but leaves its siblings loadable")
    func removingOnePrecisionKeepsSiblingsIntact() throws {
        let root = try makeFixture()
        defer { try? FileManager.default.removeItem(at: root) }
        let fm = FileManager.default
        let repo = try makeRepoCache(precisions: ["mxfp4", "bf16"], in: root)
        let blobs = repo.appendingPathComponent("blobs")
        let snapshot = repo.appendingPathComponent("snapshots/abc123")

        MlxModelLoader.remove("\(Self.repo)/mxfp4", from: repo)

        #expect(!fm.fileExists(atPath: snapshot.appendingPathComponent("mxfp4").path))
        #expect(fm.fileExists(atPath: snapshot.appendingPathComponent("bf16").path))
        // The blob only mxfp4 referenced is gone - the space is actually reclaimed.
        #expect(!fm.fileExists(atPath: blobs.appendingPathComponent("sha-mxfp4-weights").path))
        // The blob bf16 still references survives, and so does the one they *shared*: deleting it
        // would leave bf16 with a dangling tokenizer and it would fail to load.
        #expect(fm.fileExists(atPath: blobs.appendingPathComponent("sha-bf16-weights").path))
        #expect(fm.fileExists(atPath: blobs.appendingPathComponent("sha-shared-tokenizer").path))
        // bf16 still reads through its symlinks.
        let tokenizer = try String(
            contentsOf: snapshot.appendingPathComponent("bf16/tokenizer.json"), encoding: .utf8
        )
        #expect(tokenizer == "shared tokenizer")
        #expect(MlxModelLoader.hasWeights("\(Self.repo)/bf16", in: repo))
        #expect(!MlxModelLoader.hasWeights("\(Self.repo)/mxfp4", in: repo))
    }

    @Test("Removing the last precision drops the whole repo directory")
    func removingTheLastPrecisionDropsTheRepo() throws {
        let root = try makeFixture()
        defer { try? FileManager.default.removeItem(at: root) }
        let repo = try makeRepoCache(precisions: ["mxfp4", "bf16"], in: root)

        MlxModelLoader.remove("\(Self.repo)/mxfp4", from: repo)
        #expect(FileManager.default.fileExists(atPath: repo.path))
        MlxModelLoader.remove("\(Self.repo)/bf16", from: repo)
        #expect(!FileManager.default.fileExists(atPath: repo.path))
    }

    @Test("A plain repo id still drops its whole directory")
    func plainRepoRemovalIsUnchanged() throws {
        let root = try makeFixture()
        defer { try? FileManager.default.removeItem(at: root) }
        let fm = FileManager.default
        let repo = root.appendingPathComponent("models--LiquidAI--LFM2.5-350M-MLX-8bit", isDirectory: true)
        let snapshot = repo.appendingPathComponent("snapshots/abc123", isDirectory: true)
        try fm.createDirectory(at: snapshot, withIntermediateDirectories: true)
        try Data("w".utf8).write(to: snapshot.appendingPathComponent("model.safetensors"))

        MlxModelLoader.remove("LiquidAI/LFM2.5-350M-MLX-8bit", from: repo)
        #expect(!fm.fileExists(atPath: repo.path))
    }
}
