@testable import DeepAgentsMLX
import Foundation
import Testing

/// ``PrefixKVStore`` - the on-disk persistence of the prefix-cache base snapshot: the stable
/// cross-process fingerprint, the token-trace artifact, and the content-addressed candidate
/// scan/selection (exercised against handcrafted safetensors headers). The base-snapshot KV
/// round-trip itself goes through mlx-swift-lm's prompt-cache serialization, which needs Metal
/// (no metallib under `swift test` - only xcodebuild produces one), so it is exercised
/// end-to-end via the built ripple binary instead: two cold `ripple -p` runs write trace ->
/// base, a third resumes from it.
struct PrefixKVStoreTests {
    private func tempDir() -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("prefix-kv-test-\(UUID().uuidString)", isDirectory: true)
        try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    /// A base snapshot file with a valid safetensors *header* but no tensor data - enough for
    /// the header scan (``PrefixKVStore/baseCandidates``), which never loads the body.
    @discardableResult
    private func writeFakeBase(
        directory: URL, modelID: String, tokens: [Int],
        version: String = "2", revision: String = "unknown", age: TimeInterval = 3600
    ) throws -> URL {
        let url = PrefixKVStore.baseURL(modelID: modelID, tokens: tokens, directory: directory)
        // `savePromptCache` namespaces user metadata under `1.` in the header (the Python
        // cache_metadata layout); write the same shape the scanner reads in production.
        let object: [String: Any] = ["__metadata__": [
            "1.version": version, "1.model": modelID, "1.revision": revision,
            "1.tokens": tokens.map(String.init).joined(separator: ",")
        ]]
        let header = try JSONSerialization.data(withJSONObject: object)
        var data = withUnsafeBytes(of: UInt64(header.count).littleEndian) { Data($0) }
        data.append(header)
        try data.write(to: url)
        try FileManager.default.setAttributes(
            [.modificationDate: Date(timeIntervalSinceNow: -age)], ofItemAtPath: url.path
        )
        return url
    }

    // MARK: - Fingerprint

    @Test("The fingerprint is stable across calls (and, by construction, across processes)")
    func fingerprintIsStable() {
        let a = PrefixKVStore.fingerprint(systemPrompt: "sys", toolNames: ["alpha", "beta"])
        let b = PrefixKVStore.fingerprint(systemPrompt: "sys", toolNames: ["alpha", "beta"])
        #expect(a == b)
        // Golden value: a change here breaks every persisted snapshot's key, so it must be
        // deliberate (users' disk caches are orphaned, not corrupted - they just re-warm).
        #expect(a == 6_508_410_457_055_901_089)
    }

    @Test("The fingerprint is order-sensitive and field-delimited")
    func fingerprintSensitivity() {
        let base = PrefixKVStore.fingerprint(systemPrompt: "sys", toolNames: ["a", "b"])
        #expect(base != PrefixKVStore.fingerprint(systemPrompt: "sys", toolNames: ["b", "a"]))
        // The delimiter keeps ("ab", ["c"]) distinct from ("a", ["bc"]) even though the
        // concatenated bytes match.
        #expect(PrefixKVStore.fingerprint(systemPrompt: "ab", toolNames: ["c"])
            != PrefixKVStore.fingerprint(systemPrompt: "a", toolNames: ["bc"]))
        #expect(base != PrefixKVStore.fingerprint(systemPrompt: nil, toolNames: ["a", "b"]))
    }

    // MARK: - Token trace

    @Test("A token trace round-trips, and is keyed by fingerprint and model id")
    func traceRoundTrip() {
        let dir = tempDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        let tokens = [3, 1, 4, 1, 5, 9, 2, 6]
        PrefixKVStore.saveTrace(tokens: tokens, modelID: "test/model", fingerprint: 7, directory: dir)
        #expect(PrefixKVStore.loadTrace(modelID: "test/model", fingerprint: 7, directory: dir) == tokens)
        // A different fingerprint (changed system prompt / tools) misses.
        #expect(PrefixKVStore.loadTrace(modelID: "test/model", fingerprint: 8, directory: dir) == nil)
        // So does a different model id.
        #expect(PrefixKVStore.loadTrace(modelID: "test/other", fingerprint: 7, directory: dir) == nil)
    }

    @Test("A corrupt or empty trace is ignored instead of failing the turn")
    func corruptTraceIgnored() throws {
        let dir = tempDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        PrefixKVStore.saveTrace(tokens: [1, 2], modelID: "test/model", fingerprint: 3, directory: dir)
        let url = try #require(
            try FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)
                .first { $0.pathExtension == "json" }
        )
        try Data("not json".utf8).write(to: url)
        #expect(PrefixKVStore.loadTrace(modelID: "test/model", fingerprint: 3, directory: dir) == nil)
        PrefixKVStore.saveTrace(tokens: [], modelID: "test/model", fingerprint: 4, directory: dir)
        #expect(PrefixKVStore.loadTrace(modelID: "test/model", fingerprint: 4, directory: dir) == nil)
    }

    @Test("Negative fingerprints key files safely (hex of the bit pattern, no minus sign)")
    func negativeFingerprintKeying() {
        let dir = tempDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        PrefixKVStore.saveTrace(tokens: [9, 9], modelID: "test/model", fingerprint: -12345, directory: dir)
        #expect(PrefixKVStore.loadTrace(modelID: "test/model", fingerprint: -12345, directory: dir) == [9, 9])
        // The model id's slash is flattened; the fingerprint renders as unsigned hex.
        let names = (try? FileManager.default.contentsOfDirectory(atPath: dir.path)) ?? []
        #expect(names.count == 1)
        #expect(names.first?.contains("/") == false)
        #expect(names.first?.hasPrefix("test--model-") == true)
    }

    // MARK: - Content-addressed bases

    @Test("Bases are keyed by token content: same tokens share a file, any config")
    func contentAddressing() {
        let dir = tempDir()
        let a = PrefixKVStore.baseURL(modelID: "test/model", tokens: [1, 2, 3], directory: dir)
        let b = PrefixKVStore.baseURL(modelID: "test/model", tokens: [1, 2, 3], directory: dir)
        #expect(a == b) // no config fingerprint in the key - this is what enables sharing
        #expect(a != PrefixKVStore.baseURL(modelID: "test/model", tokens: [1, 2, 4], directory: dir))
        #expect(a != PrefixKVStore.baseURL(modelID: "test/other", tokens: [1, 2, 3], directory: dir))
        // Golden value: a change orphans every persisted base (they re-warm), so it must be
        // deliberate - same contract as the fingerprint golden above.
        #expect(PrefixKVStore.contentKey([1, 2, 3]) == "da2bfb225e0d1f05")
    }

    @Test("The candidate scan validates headers and deletes files that can never load again")
    func candidateScanValidation() throws {
        let dir = tempDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        let good = try writeFakeBase(directory: dir, modelID: "test/model", tokens: [1, 2, 3])
        let oldFormat = try writeFakeBase(
            directory: dir, modelID: "test/model", tokens: [1, 2], version: "1"
        )
        let staleWeights = try writeFakeBase(
            directory: dir, modelID: "test/model", tokens: [7, 8], revision: "cafebabe"
        )
        // Same filename prefix, different model id - must be skipped, never deleted.
        let otherModel = try writeFakeBase(directory: dir, modelID: "test/model-pro", tokens: [5])
        // A corrupt header that is *fresh* is probably a concurrent mid-write: keep it.
        let midWrite = dir.appendingPathComponent("test--model-deadbeef.safetensors")
        try Data("garbage".utf8).write(to: midWrite)

        let candidates = PrefixKVStore.baseCandidates(modelID: "test/model", directory: dir)
        #expect(candidates.map(\.url.lastPathComponent) == [good.lastPathComponent])
        #expect(candidates.first?.tokens == [1, 2, 3])
        let fm = FileManager.default
        #expect(!fm.fileExists(atPath: oldFormat.path)) // v1: deleted on sight
        #expect(!fm.fileExists(atPath: staleWeights.path)) // stale revision: deleted
        #expect(fm.fileExists(atPath: otherModel.path))
        #expect(fm.fileExists(atPath: midWrite.path))
    }

    @Test("Strict-prefix matches are ordered deepest-first; equal/diverging bases are excluded")
    func strictPrefixSelection() throws {
        let dir = tempDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        try writeFakeBase(directory: dir, modelID: "test/model", tokens: [1, 2]) // short match
        try writeFakeBase(directory: dir, modelID: "test/model", tokens: [1, 2, 3, 4]) // deep match
        try writeFakeBase(directory: dir, modelID: "test/model", tokens: [1, 2, 3, 4, 5]) // == prompt
        try writeFakeBase(directory: dir, modelID: "test/model", tokens: [1, 9]) // diverges
        let candidates = PrefixKVStore.baseCandidates(modelID: "test/model", directory: dir)
        let matches = PrefixKVStore.strictPrefixMatches(candidates, promptTokens: [1, 2, 3, 4, 5])
        #expect(matches.map(\.tokens) == [[1, 2, 3, 4], [1, 2]])
    }

    @Test("With no resumable base, seed returns the best divergence trace on record")
    func seedTraceFallback() throws {
        let dir = tempDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        let prompt = [1, 2, 3, 4, 5]
        // No strict-prefix base exists, but a *diverging* base (another config) shares [1, 2, 3]:
        // its tokens make the better trace, so this run's divergence lands on the shared boundary.
        try writeFakeBase(directory: dir, modelID: "test/model", tokens: [1, 2, 3, 9, 9, 9])
        PrefixKVStore.saveTrace(tokens: [1, 8], modelID: "test/model", fingerprint: 42, directory: dir)
        var seed = PrefixKVStore.seed(
            modelID: "test/model", fingerprint: 42, promptTokens: prompt, directory: dir
        )
        #expect(seed.base == nil)
        #expect(seed.trace == [1, 2, 3, 9, 9, 9])
        // When the config's own trace shares more, it wins instead.
        PrefixKVStore.saveTrace(tokens: [1, 2, 3, 4, 7], modelID: "test/model", fingerprint: 42, directory: dir)
        seed = PrefixKVStore.seed(
            modelID: "test/model", fingerprint: 42, promptTokens: prompt, directory: dir
        )
        #expect(seed.base == nil)
        #expect(seed.trace == [1, 2, 3, 4, 7])
        // Nothing sharing even one token: nothing to seed.
        let empty = PrefixKVStore.seed(
            modelID: "test/model", fingerprint: 42, promptTokens: [6, 6], directory: dir
        )
        #expect(empty.base == nil)
        #expect(empty.trace == nil)
    }
}

/// The store's *bounds* and its inventory: which snapshots survive a prune, what the directory
/// reports it is holding, and what removal takes away.
///
/// Serialized because every case here mutates process-global limits, and reset in `deinit` so a
/// failure part-way through cannot leak a 1-snapshot budget into the rest of the suite.
@Suite(.serialized)
final class PrefixKVStoreBoundsTests {
    private let directory: URL

    init() {
        directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("prefix-kv-bounds-\(UUID().uuidString)", isDirectory: true)
        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    }

    deinit {
        PrefixKVStore.maxSnapshotsPerModel = 6
        PrefixKVStore.maxTotalBytes = 4 << 30
        PrefixKVStore.isEnabledOverride = nil
        try? FileManager.default.removeItem(at: directory)
    }

    /// A base snapshot with a valid header, padded to `bytes` so the byte-cap pass has something
    /// real to measure. `age` orders it: larger is older.
    @discardableResult
    private func writeBase(
        modelID: String, tokens: [Int], bytes: Int = 0, age: TimeInterval
    ) throws -> URL {
        let url = PrefixKVStore.baseURL(modelID: modelID, tokens: tokens, directory: directory)
        let object: [String: Any] = ["__metadata__": [
            "1.version": "2", "1.model": modelID, "1.revision": "unknown",
            "1.tokens": tokens.map(String.init).joined(separator: ",")
        ]]
        let header = try JSONSerialization.data(withJSONObject: object)
        var data = withUnsafeBytes(of: UInt64(header.count).littleEndian) { Data($0) }
        data.append(header)
        if bytes > data.count { data.append(Data(repeating: 0, count: bytes - data.count)) }
        try data.write(to: url)
        try FileManager.default.setAttributes(
            [.modificationDate: Date(timeIntervalSinceNow: -age)], ofItemAtPath: url.path
        )
        return url
    }

    private func names() -> Set<String> {
        let files = (try? FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)) ?? []
        return Set(files.map(\.lastPathComponent))
    }

    // MARK: - Prune, pass 1: per-model budgets

    @Test("Each model keeps its own newest snapshots")
    func prunePerModelBudget() throws {
        PrefixKVStore.maxSnapshotsPerModel = 2
        PrefixKVStore.maxTotalBytes = 0 // isolate pass 1
        var kept: Set<String> = []
        for (index, age) in [10.0, 20, 30, 40].enumerated() {
            let a = try writeBase(modelID: "vendor/alpha", tokens: [1, index], age: age)
            let b = try writeBase(modelID: "vendor/beta", tokens: [2, index], age: age)
            if index < 2 { kept.formUnion([a.lastPathComponent, b.lastPathComponent]) }
        }
        PrefixKVStore.pruneNow(directory: directory)
        // Two per model, not two overall - the old global rule would have left two files total.
        #expect(names() == kept)
    }

    @Test("A busy model cannot evict a quiet model's only snapshot")
    func pruneBudgetsAreIndependent() throws {
        PrefixKVStore.maxSnapshotsPerModel = 3
        PrefixKVStore.maxTotalBytes = 0
        for index in 0 ..< 6 { try writeBase(modelID: "vendor/busy", tokens: [1, index], age: Double(index + 1)) }
        let quiet = try writeBase(modelID: "vendor/quiet", tokens: [9], age: 999)
        PrefixKVStore.pruneNow(directory: directory)
        #expect(FileManager.default.fileExists(atPath: quiet.path))
        #expect(names().count == 4) // 3 busy + 1 quiet
    }

    @Test("Grouping reads the header, so one model id prefixing another keeps them apart")
    func pruneGroupsByHeaderNotFilename() throws {
        PrefixKVStore.maxSnapshotsPerModel = 1
        PrefixKVStore.maxTotalBytes = 0
        // `flatID` makes these near-identical file-name prefixes; only the header tells them apart.
        let short = try writeBase(modelID: "vendor/model/8bit", tokens: [1], age: 10)
        let long = try writeBase(modelID: "vendor/model/8bit-extra", tokens: [2], age: 20)
        PrefixKVStore.pruneNow(directory: directory)
        #expect(FileManager.default.fileExists(atPath: short.path))
        #expect(FileManager.default.fileExists(atPath: long.path))
    }

    // MARK: - Prune, pass 2: the byte ceiling

    @Test("The oldest snapshots go until the store fits the byte cap")
    func pruneEvictsOldestUnderTheCap() throws {
        PrefixKVStore.maxSnapshotsPerModel = 99 // isolate pass 2
        let newest = try writeBase(modelID: "vendor/alpha", tokens: [1], bytes: 4000, age: 10)
        let middle = try writeBase(modelID: "vendor/alpha", tokens: [2], bytes: 4000, age: 20)
        let oldest = try writeBase(modelID: "vendor/alpha", tokens: [3], bytes: 4000, age: 30)
        PrefixKVStore.maxTotalBytes = 9000 // room for two
        PrefixKVStore.pruneNow(directory: directory)
        #expect(FileManager.default.fileExists(atPath: newest.path))
        #expect(FileManager.default.fileExists(atPath: middle.path))
        #expect(!FileManager.default.fileExists(atPath: oldest.path))
    }

    @Test("The newest snapshot survives even when it alone exceeds the cap")
    func pruneNeverEvictsTheNewest() throws {
        PrefixKVStore.maxSnapshotsPerModel = 99
        PrefixKVStore.maxTotalBytes = 100
        let newest = try writeBase(modelID: "vendor/alpha", tokens: [1], bytes: 8000, age: 10)
        let older = try writeBase(modelID: "vendor/alpha", tokens: [2], bytes: 8000, age: 20)

        PrefixKVStore.pruneNow(directory: directory)

        // Two snapshots, so eviction genuinely runs - and still stops before the newest. Without
        // that floor a cap below one base size means write-then-delete on every single turn.
        #expect(!FileManager.default.fileExists(atPath: older.path))
        #expect(FileManager.default.fileExists(atPath: newest.path))
    }

    @Test("A lone snapshot is never evicted, whatever the cap")
    func pruneKeepsASingleSnapshot() throws {
        PrefixKVStore.maxSnapshotsPerModel = 99
        PrefixKVStore.maxTotalBytes = 100
        let only = try writeBase(modelID: "vendor/alpha", tokens: [1], bytes: 8000, age: 10)
        PrefixKVStore.pruneNow(directory: directory)
        #expect(FileManager.default.fileExists(atPath: only.path))
    }

    @Test("A cap of zero means unlimited")
    func zeroCapIsUnlimited() throws {
        PrefixKVStore.maxSnapshotsPerModel = 99
        PrefixKVStore.maxTotalBytes = 0
        for index in 0 ..< 4 { try writeBase(modelID: "vendor/alpha", tokens: [index], bytes: 8000, age: Double(index + 1)) }
        PrefixKVStore.pruneNow(directory: directory)
        #expect(names().count == 4)
    }

    @Test("Limits clamp to values that cannot disable the store")
    func limitsClamp() {
        PrefixKVStore.maxSnapshotsPerModel = 0
        #expect(PrefixKVStore.maxSnapshotsPerModel == 1) // keeping nothing would re-prefill every turn
        PrefixKVStore.maxTotalBytes = -1
        #expect(PrefixKVStore.maxTotalBytes == 0)
    }

    // MARK: - Inventory

    @Test("Inventory groups bytes by model and totals them")
    func inventoryGroupsByModel() throws {
        try writeBase(modelID: "vendor/alpha", tokens: [1], bytes: 5000, age: 10)
        try writeBase(modelID: "vendor/alpha", tokens: [2], bytes: 3000, age: 20)
        try writeBase(modelID: "vendor/beta", tokens: [3], bytes: 1000, age: 30)
        let inventory = PrefixKVStore.inventory(directory: directory, knownModelIDs: [])
        #expect(inventory.models.map(\.modelID) == ["vendor/alpha", "vendor/beta"]) // largest first
        let alpha = try #require(inventory.models.first)
        #expect(alpha.snapshotCount == 2)
        #expect(alpha.snapshotBytes == 8000)
        #expect(inventory.totalBytes == 9000)
        #expect(inventory.unattributedCount == 0)
    }

    @Test("A trace is attributed by the model key its payload carries")
    func inventoryAttributesTracesByPayload() throws {
        PrefixKVStore.isEnabledOverride = true
        PrefixKVStore.saveTrace(tokens: [1, 2, 3], modelID: "vendor/alpha", fingerprint: 7, directory: directory)
        // Deliberately no known ids: attribution must come from the payload, not a name match.
        let inventory = PrefixKVStore.inventory(directory: directory, knownModelIDs: [])
        #expect(inventory.models.map(\.modelID) == ["vendor/alpha"])
        #expect(inventory.models.first?.traceCount == 1)
    }

    @Test("A trace written before the model key falls back to the longest matching id")
    func inventoryAttributesLegacyTracesByPrefix() throws {
        // The pre-0.5.0 payload shape: version, revision, tokens - no model.
        let payload = ["version": "2", "revision": "unknown", "tokens": "1,2,3"]
        let data = try JSONEncoder().encode(payload)
        try data.write(to: directory.appendingPathComponent("vendor--model--8bit-1f.json"))
        let inventory = PrefixKVStore.inventory(
            directory: directory, knownModelIDs: ["vendor/model", "vendor/model/8bit"]
        )
        // The longer id wins, so a shorter one never claims a longer sibling's files.
        #expect(inventory.models.map(\.modelID) == ["vendor/model/8bit"])
    }

    @Test("Scanning never deletes what it cannot parse")
    func inventoryIsReadOnly() throws {
        let junk = directory.appendingPathComponent("not-a-real-snapshot.safetensors")
        try Data(repeating: 7, count: 512).write(to: junk)
        let inventory = PrefixKVStore.inventory(directory: directory, knownModelIDs: [])
        #expect(inventory.unattributedCount == 1)
        #expect(inventory.unattributedBytes == 512)
        // `baseCandidates` deletes unreadable files; a scan the user is looking at must not.
        #expect(FileManager.default.fileExists(atPath: junk.path))
    }

    // MARK: - Removal

    @Test("Removing one model's cache spares every other model")
    func removeOneModel() throws {
        try writeBase(modelID: "vendor/alpha", tokens: [1], bytes: 2000, age: 10)
        PrefixKVStore.isEnabledOverride = true
        PrefixKVStore.saveTrace(tokens: [1], modelID: "vendor/alpha", fingerprint: 1, directory: directory)
        let survivor = try writeBase(modelID: "vendor/beta", tokens: [2], bytes: 1000, age: 20)

        let freed = PrefixKVStore.removeAll(modelID: "vendor/alpha", directory: directory, knownModelIDs: [])

        #expect(freed > 2000) // the base plus its trace
        #expect(FileManager.default.fileExists(atPath: survivor.path))
        let inventory = PrefixKVStore.inventory(directory: directory, knownModelIDs: [])
        #expect(inventory.models.map(\.modelID) == ["vendor/beta"])
    }

    @Test("Clearing the store empties it but keeps the directory")
    func removeEverything() throws {
        try writeBase(modelID: "vendor/alpha", tokens: [1], bytes: 2000, age: 10)
        try writeBase(modelID: "vendor/beta", tokens: [2], bytes: 1000, age: 20)
        let freed = PrefixKVStore.removeAll(directory: directory)
        #expect(freed == 3000)
        #expect(PrefixKVStore.inventory(directory: directory, knownModelIDs: []).totalBytes == 0)
        #expect(FileManager.default.fileExists(atPath: directory.path))
    }

    @Test("Inventory and removal work while the store is switched off")
    func reclaimingWorksWhileDisabled() throws {
        try writeBase(modelID: "vendor/alpha", tokens: [1], bytes: 2000, age: 10)
        PrefixKVStore.isEnabledOverride = false
        // Turning the cache off stops it writing; it must not strand what is already on disk.
        #expect(PrefixKVStore.inventory(directory: directory, knownModelIDs: []).totalBytes == 2000)
        #expect(PrefixKVStore.removeAll(directory: directory) == 2000)
    }

    // MARK: - Trace back-compat

    @Test("A trace without the model key still loads")
    func legacyTraceStillLoads() throws {
        PrefixKVStore.isEnabledOverride = true
        let payload = ["version": "2", "revision": "unknown", "tokens": "4,5,6"]
        let data = try JSONEncoder().encode(payload)
        let hex = String(UInt64(bitPattern: Int64(11)), radix: 16)
        try data.write(to: directory.appendingPathComponent("vendor--alpha-\(hex).json"))
        // Requiring the new key would orphan every trace already on a user's disk.
        #expect(PrefixKVStore.loadTrace(modelID: "vendor/alpha", fingerprint: 11, directory: directory) == [4, 5, 6])
    }
}
