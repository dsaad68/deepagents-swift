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
