import Foundation
import MLX
import MLXLMCommon

/// On-disk persistence for ``PrefixCacheSlot``'s *base* snapshot, so the stable prompt prefix
/// (system prompt + tool schemas) survives across processes - not just while a model stays
/// resident. Without it every fresh process (a `ripple -p` one-shot, a `ripple chat` launch, an
/// app relaunch) pays the full multi-second prefill of the ~10k-token agent prompt; with it, a
/// cold start resumes from the saved base and prefills only the conversation suffix.
///
/// Two artifacts, both under ``defaultDirectory``:
/// - `<model>-<fingerprint>.json` - a token *trace* per (model, system+tools fingerprint): the
///   last full prompt's token ids, written after every turn. A base boundary is only *learned*
///   by diffing two prompts, so the next process seeds ``PrefixCacheSlot/tokens`` from the
///   trace: its first turn finds the divergence boundary and establishes (or deepens) the base
///   there, mid-prefill, then persists it.
/// - `<model>-<tokenHash>.safetensors` - a base snapshot via mlx-swift-lm's `savePromptCache` /
///   `loadPromptCache` (which round-trip every cache type our hybrids use, `KVCacheSimple` and
///   the recurrent `MambaCache`). **Content-addressed** by a hash of the tokens it covers, not by
///   config: any config whose prompt starts with those tokens can resume from it. ``seed(modelID:
///   fingerprint:promptTokens:directory:)`` picks the *longest* stored base that strict-prefixes
///   the incoming prompt, so e.g. a run whose MCP server is down reuses the snapshot up to where
///   the missing tool schemas would start, instead of re-prefilling everything.
///
/// Correctness never rests on a hash or a file: a loaded base is only used when its tokens are
/// verified to be a strict prefix of the incoming prompt (same guarantee as the in-memory
/// snapshots), and the metadata must match the model id and the *downloaded revision*
/// (`refs/main`), so updated weights can't silently replay stale KV. Any mismatch or decode
/// failure just means a normal full prefill.
public enum PrefixKVStore {
    /// Runtime switch for the store (ripple's `/config` "Prefill cache" toggle; a host app's
    /// settings). Takes precedence over the env kill switch; `nil` defers to it. Lock-guarded:
    /// set from the UI thread, read from the model containers' threads.
    public static var isEnabledOverride: Bool? {
        get { overrideLock.withLock { overrideStorage } }
        set { overrideLock.withLock { overrideStorage = newValue } }
    }

    private static let overrideLock = NSLock()
    private nonisolated(unsafe) static var overrideStorage: Bool?

    /// Whether the store reads/writes disk: the runtime override when set, else the
    /// `DEEPAGENTS_PREFIX_KV=0` env kill switch. In-memory prefix caching is unaffected.
    static var isEnabled: Bool {
        isEnabledOverride ?? (ProcessInfo.processInfo.environment["DEEPAGENTS_PREFIX_KV"] != "0")
    }

    /// Where snapshots live: `$DEEPAGENTS_PREFIX_KV_DIR`, else `~/.cache/deepagents/prefix-kv`.
    static var defaultDirectory: URL {
        if let dir = ProcessInfo.processInfo.environment["DEEPAGENTS_PREFIX_KV_DIR"], !dir.isEmpty {
            return URL(fileURLWithPath: dir, isDirectory: true)
        }
        return FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".cache/deepagents/prefix-kv", isDirectory: true)
    }

    /// Most-recently-used base snapshots kept after a save; the rest (other models / dead
    /// configs) are pruned so the directory stays bounded (a 9B model's base is a few hundred
    /// MB). Loading a base refreshes its position.
    private static let maxSnapshots = 6
    /// Format version stamped into (and required from) every artifact. v2: bases went from
    /// fingerprint-keyed to content-addressed; v1 files are deleted on sight and re-warm.
    private static let version = "2"

    // MARK: - Fingerprint

    /// Stable identity of a *configuration*'s reusable prefix - keys the token trace (and the
    /// in-memory slot). Replaces `Hasher` (whose seed is randomized per process) so the value can
    /// key files across runs: FNV-1a 64 over the system prompt and the tool names,
    /// order-sensitive to match the rendered prompt.
    static func fingerprint(systemPrompt: String?, toolNames: [String]) -> Int {
        var hash: UInt64 = 0xCBF2_9CE4_8422_2325
        func mix(_ string: String) {
            for byte in string.utf8 {
                hash ^= UInt64(byte)
                hash = hash &* 0x0000_0100_0000_01B3
            }
            mix(byte: 0x1F) // unit separator between fields, so ("ab","c") != ("a","bc")
        }
        func mix(byte: UInt8) {
            hash ^= UInt64(byte)
            hash = hash &* 0x0000_0100_0000_01B3
        }
        mix(systemPrompt ?? "")
        for name in toolNames { mix(name) }
        return Int(bitPattern: UInt(truncatingIfNeeded: hash))
    }

    /// FNV-1a 64 over the token ids (8 little-endian bytes each) - the *content address* of a
    /// base snapshot. Same tokens, same file, whichever config produced them; that is what lets
    /// configs share the prefix they have in common.
    static func contentKey(_ tokens: [Int]) -> String {
        var hash: UInt64 = 0xCBF2_9CE4_8422_2325
        for token in tokens {
            var value = UInt64(bitPattern: Int64(token))
            for _ in 0 ..< 8 {
                hash ^= value & 0xFF
                hash = hash &* 0x0000_0100_0000_01B3
                value >>= 8
            }
        }
        return String(hash, radix: 16)
    }

    // MARK: - Seeding (what a fresh slot resumes from)

    /// What ``seed(modelID:fingerprint:promptTokens:directory:)`` found on disk for a fresh slot.
    /// The two parts are independent and compose: resume from `base`, then use `trace` to learn
    /// where this config's *own* stable prefix ends (it may reach past a shared base).
    struct DiskSeed {
        /// The deepest stored base whose tokens strict-prefix the prompt: resume from its KV.
        var base: (cache: [KVCache], tokens: [Int])?
        /// The best previous prompt on record - this config's trace, or another base's token
        /// list, whichever shares the longest prefix with the prompt. Seeds the divergence diff
        /// so the first turn establishes (or deepens) a base at a stable boundary.
        var trace: [Int]?
    }

    /// The best on-disk starting points for a prompt (see ``DiskSeed``). Both parts empty when
    /// the store is disabled or has nothing relevant.
    static func seed(
        modelID: String, fingerprint: Int, promptTokens: [Int], directory: URL = defaultDirectory
    ) -> DiskSeed {
        var seed = DiskSeed()
        guard isEnabled else { return seed }
        let candidates = baseCandidates(modelID: modelID, directory: directory)
        for candidate in strictPrefixMatches(candidates, promptTokens: promptTokens) {
            guard let (cache, metadata) = try? loadPromptCache(url: candidate.url),
                  decodeTokens(metadata["tokens"]) == candidate.tokens
            else {
                removeUnlessFresh(candidate.url) // truncated or corrupt body: drop and try the next
                continue
            }
            touch(candidate.url) // used = most recently used; pruning spares it
            seed.base = (cache, candidate.tokens)
            break
        }
        var best = loadTrace(modelID: modelID, fingerprint: fingerprint, directory: directory) ?? []
        var bestShared = commonPrefixLength(best, promptTokens)
        for candidate in candidates {
            let shared = commonPrefixLength(candidate.tokens, promptTokens)
            if shared > bestShared {
                best = candidate.tokens
                bestShared = shared
            }
        }
        if bestShared > 0 { seed.trace = best }
        return seed
    }

    /// A validated base snapshot on disk (header only - the KV is loaded lazily by ``seed``).
    struct BaseCandidate {
        let url: URL
        let tokens: [Int]
    }

    /// Every stored base for `modelID` whose header passes validation (version, model, current
    /// downloaded revision, decodable tokens), read without loading tensor data. Files that can
    /// never validate again (old format, stale revision, corrupt header) are deleted - unless
    /// recently modified, which usually means another process is mid-write.
    static func baseCandidates(modelID: String, directory: URL = defaultDirectory) -> [BaseCandidate] {
        let prefix = flatID(modelID) + "-"
        guard let files = try? FileManager.default.contentsOfDirectory(
            at: directory, includingPropertiesForKeys: nil
        ) else { return [] }
        var candidates: [BaseCandidate] = []
        for url in files
            where url.pathExtension == "safetensors" && url.lastPathComponent.hasPrefix(prefix) {
            guard let metadata = headerMetadata(url: url) else {
                removeUnlessFresh(url)
                continue
            }
            guard metadata["model"] == modelID else { continue } // id prefix collision: not ours
            guard metadata["version"] == version,
                  metadata["revision"] == revisionOnDisk(modelID) ?? "unknown",
                  let tokens = decodeTokens(metadata["tokens"]), !tokens.isEmpty
            else {
                removeUnlessFresh(url) // old format or stale weights: never resumable again
                continue
            }
            candidates.append(BaseCandidate(url: url, tokens: tokens))
        }
        return candidates
    }

    /// The candidates whose tokens are a strict prefix of `promptTokens`, deepest first (the
    /// longest prefix leaves the least to re-prefill). Deterministic tie-break by filename.
    static func strictPrefixMatches(
        _ candidates: [BaseCandidate], promptTokens: [Int]
    ) -> [BaseCandidate] {
        candidates.filter { isStrictPrefix($0.tokens, of: promptTokens) }
            .sorted { lhs, rhs in
                lhs.tokens.count != rhs.tokens.count
                    ? lhs.tokens.count > rhs.tokens.count
                    : lhs.url.lastPathComponent < rhs.url.lastPathComponent
            }
    }

    // MARK: - Base snapshot

    /// Persist `cache` (the base snapshot's layer caches) + the tokens it covers, addressed by
    /// the tokens' content hash. When an identical, still-valid base already exists (another
    /// config sharing this prefix), the few-hundred-MB write is skipped and the file's LRU
    /// position refreshed. Failures are silent - the snapshot is an optimization, never a
    /// dependency. Prunes old snapshots after.
    static func saveBase(
        cache: [KVCache], tokens: [Int], modelID: String, directory: URL = defaultDirectory
    ) {
        guard isEnabled, !tokens.isEmpty else { return }
        let fm = FileManager.default
        try? fm.createDirectory(at: directory, withIntermediateDirectories: true)
        let url = baseURL(modelID: modelID, tokens: tokens, directory: directory)
        let revision = revisionOnDisk(modelID) ?? "unknown"
        if let existing = headerMetadata(url: url), existing["version"] == version,
           existing["model"] == modelID, existing["revision"] == revision {
            touch(url)
            return
        }
        let metadata: [String: String] = [
            "version": version,
            "model": modelID,
            "revision": revision,
            "tokens": tokens.map(String.init).joined(separator: ",")
        ]
        do {
            try savePromptCache(url: url, cache: cache, metadata: metadata)
            prune(directory)
        } catch {
            try? fm.removeItem(at: url) // never leave a half-written snapshot behind
        }
    }

    /// Where the base covering exactly `tokens` lives for `modelID`.
    static func baseURL(modelID: String, tokens: [Int], directory: URL = defaultDirectory) -> URL {
        directory.appendingPathComponent("\(flatID(modelID))-\(contentKey(tokens)).safetensors")
    }

    // MARK: - Token trace

    /// Persist the last full prompt's tokens (~tens of KB) so the *next* process can locate this
    /// config's base boundary by divergence on its very first turn - including when it resumes
    /// from a shorter *shared* base and needs to learn how much deeper its own stable prefix
    /// reaches (see ``DiskSeed``). Written after every turn; one small file per (model, config).
    static func saveTrace(
        tokens: [Int], modelID: String, fingerprint: Int, directory: URL = defaultDirectory
    ) {
        guard isEnabled, !tokens.isEmpty else { return }
        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let payload: [String: String] = [
            "version": version,
            "revision": revisionOnDisk(modelID) ?? "unknown",
            "tokens": tokens.map(String.init).joined(separator: ",")
        ]
        guard let data = try? JSONEncoder().encode(payload) else { return }
        try? data.write(to: traceURL(modelID, fingerprint, directory), options: .atomic)
        prune(directory)
    }

    /// The traced prompt tokens for (model, fingerprint), or nil when absent / stale.
    static func loadTrace(
        modelID: String, fingerprint: Int, directory: URL = defaultDirectory
    ) -> [Int]? {
        guard isEnabled,
              let data = try? Data(contentsOf: traceURL(modelID, fingerprint, directory)),
              let payload = try? JSONDecoder().decode([String: String].self, from: data),
              payload["version"] == version,
              payload["revision"] == revisionOnDisk(modelID) ?? "unknown",
              let tokens = decodeTokens(payload["tokens"]), !tokens.isEmpty
        else { return nil }
        return tokens
    }

    // MARK: - Helpers

    /// The downloaded revision of `modelID` (the commit hash in the HF cache's `refs/main`), so a
    /// re-downloaded model invalidates its snapshots. Nil when the ref isn't readable.
    static func revisionOnDisk(_ modelID: String) -> String? {
        let ref = MlxModelLoader.hubRepoDirectory(modelID).appendingPathComponent("refs/main")
        guard let hash = try? String(contentsOf: ref, encoding: .utf8) else { return nil }
        let trimmed = hash.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? nil : trimmed
    }

    /// The *user* metadata of a saved prompt cache, read from the safetensors header without
    /// loading any tensor data (and without MLX, so the candidate scan stays cheap and
    /// unit-testable). Layout: 8-byte little-endian header length, then a JSON object carrying
    /// `__metadata__`. `savePromptCache` flattens the Python-compatible
    /// `cache_metadata = [cache_info, user_metadata, cache_classes]` into that map, so our keys
    /// live under the `1.` namespace - stripped here to mirror what `loadPromptCache` returns.
    static func headerMetadata(url: URL) -> [String: String]? {
        guard let handle = try? FileHandle(forReadingFrom: url) else { return nil }
        defer { try? handle.close() }
        guard let sizeData = try? handle.read(upToCount: 8), sizeData.count == 8 else { return nil }
        var size: UInt64 = 0
        for (index, byte) in sizeData.enumerated() { size |= UInt64(byte) << (8 * UInt64(index)) }
        guard size > 0, size <= 64 << 20,
              let headerData = try? handle.read(upToCount: Int(size)), headerData.count == Int(size),
              let object = try? JSONSerialization.jsonObject(with: headerData) as? [String: Any],
              let raw = object["__metadata__"] as? [String: String]
        else { return nil }
        var user: [String: String] = [:]
        for (key, value) in raw where key.hasPrefix("1.") {
            user[String(key.dropFirst(2))] = value
        }
        return user
    }

    private static func flatID(_ modelID: String) -> String {
        modelID.replacingOccurrences(of: "/", with: "--")
    }

    private static func traceURL(_ modelID: String, _ fingerprint: Int, _ directory: URL) -> URL {
        let hex = String(UInt64(bitPattern: Int64(fingerprint)), radix: 16)
        return directory.appendingPathComponent("\(flatID(modelID))-\(hex).json")
    }

    private static func decodeTokens(_ csv: String?) -> [Int]? {
        guard let csv, !csv.isEmpty else { return nil }
        var tokens: [Int] = []
        for part in csv.split(separator: ",") {
            guard let token = Int(part) else { return nil }
            tokens.append(token)
        }
        return tokens
    }

    private static func isStrictPrefix(_ prefix: [Int], of tokens: [Int]) -> Bool {
        !prefix.isEmpty && prefix.count < tokens.count
            && commonPrefixLength(prefix, tokens) == prefix.count
    }

    private static func commonPrefixLength(_ a: [Int], _ b: [Int]) -> Int {
        let n = min(a.count, b.count)
        var i = 0
        while i < n, a[i] == b[i] { i += 1 }
        return i
    }

    /// Delete a bad snapshot - unless it changed within the last minute, which usually means a
    /// concurrent process is mid-`savePromptCache` (the header lands before the tensor data).
    private static func removeUnlessFresh(_ url: URL) {
        guard modified(url) < Date(timeIntervalSinceNow: -60) else { return }
        try? FileManager.default.removeItem(at: url)
    }

    private static func touch(_ url: URL) {
        try? FileManager.default.setAttributes([.modificationDate: Date()], ofItemAtPath: url.path)
    }

    /// Traces kept per directory - tiny files, bounded anyway so dead configs don't accumulate.
    private static let maxTraces = 16

    /// Keep the ``maxSnapshots`` most-recently-used base files (each can be a few hundred MB)
    /// and the ``maxTraces`` most-recently-written traces.
    private static func prune(_ directory: URL) {
        let fm = FileManager.default
        guard let files = try? fm.contentsOfDirectory(
            at: directory, includingPropertiesForKeys: [.contentModificationDateKey]
        ) else { return }
        let snapshots = files.filter { $0.pathExtension == "safetensors" }
            .sorted { modified($0) > modified($1) }
        for stale in snapshots.dropFirst(maxSnapshots) { try? fm.removeItem(at: stale) }
        let traces = files.filter { $0.pathExtension == "json" }
            .sorted { modified($0) > modified($1) }
        for stale in traces.dropFirst(maxTraces) { try? fm.removeItem(at: stale) }
    }

    private static func modified(_ url: URL) -> Date {
        (try? url.resourceValues(forKeys: [.contentModificationDateKey]))?
            .contentModificationDate ?? .distantPast
    }
}
