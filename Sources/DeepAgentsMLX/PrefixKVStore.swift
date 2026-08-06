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
        get { configLock.withLock { overrideStorage } }
        set { configLock.withLock { overrideStorage = newValue } }
    }

    /// Base snapshots kept **per model** after a save - pass 1 of ``prune(_:)``. A model can never
    /// evict another model's warm base, so alternating between two models no longer leaves both
    /// cold. Values below 1 clamp to 1: a store that keeps nothing would re-prefill every turn.
    ///
    /// Per-model counts say nothing about bytes (one 2.6B base measured 260 MB), which is what
    /// ``maxTotalBytes`` is for. Lock-guarded alongside the other limits.
    public static var maxSnapshotsPerModel: Int {
        get { configLock.withLock { snapshotsPerModelStorage } }
        set { configLock.withLock { snapshotsPerModelStorage = max(1, newValue) } }
    }

    /// Ceiling on the total bytes of stored base snapshots - pass 2 of ``prune(_:)``. `0` (or any
    /// negative value) means no cap. Traces are excluded: they are bounded at ``maxTraces`` files
    /// of tens of KB, against snapshots measured in hundreds of MB.
    public static var maxTotalBytes: Int64 {
        get { configLock.withLock { totalBytesStorage } }
        set { configLock.withLock { totalBytesStorage = max(0, newValue) } }
    }

    private static let configLock = NSLock()
    private nonisolated(unsafe) static var overrideStorage: Bool?
    private nonisolated(unsafe) static var snapshotsPerModelStorage = 6
    private nonisolated(unsafe) static var totalBytesStorage: Int64 = 4 << 30

    /// Whether the store reads/writes disk: the runtime override when set, else the
    /// `DEEPAGENTS_PREFIX_KV=0` env kill switch. In-memory prefix caching is unaffected.
    static var isEnabled: Bool {
        isEnabledOverride ?? (ProcessInfo.processInfo.environment["DEEPAGENTS_PREFIX_KV"] != "0")
    }

    /// Where snapshots live: `$DEEPAGENTS_PREFIX_KV_DIR`, else `~/.cache/deepagents/prefix-kv`.
    /// Public so a host can show the path, and because it is the default argument of every
    /// inventory and removal entry point below.
    public static var defaultDirectory: URL {
        if let dir = ProcessInfo.processInfo.environment["DEEPAGENTS_PREFIX_KV_DIR"], !dir.isEmpty {
            return URL(fileURLWithPath: dir, isDirectory: true)
        }
        return FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".cache/deepagents/prefix-kv", isDirectory: true)
    }

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
        // `model` is additive - `loadTrace` must never require it, or every trace already on a
        // user's disk would orphan. It exists so a trace can be attributed to its model without
        // guessing from the file name (see `owner(of:knownModelIDs:)`).
        let payload: [String: String] = [
            "version": version,
            "model": modelID,
            "revision": revisionOnDisk(modelID) ?? "unknown",
            "tokens": tokens.map(String.init).joined(separator: ",")
        ]
        guard let data = try? JSONEncoder().encode(payload) else { return }
        try? data.write(to: traceURL(modelID, fingerprint, directory), options: .atomic)
        pruneTraces(directory)
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

    // MARK: - Inventory and removal

    //
    // Everything in this section deliberately ignores ``isEnabled``. Turning the store off stops
    // it *writing*; it does not make the hundreds of MB already on disk someone else's problem.
    // A host has to be able to show and reclaim that space either way.

    /// What one model occupies in the store.
    public struct ModelUsage: Sendable, Identifiable, Equatable {
        public let modelID: String
        public let snapshotCount: Int
        public let snapshotBytes: Int64
        public let traceCount: Int
        public let traceBytes: Int64

        public var bytes: Int64 { snapshotBytes + traceBytes }
        public var id: String { modelID }
    }

    /// Everything the store holds, grouped by model.
    public struct Inventory: Sendable, Equatable {
        public let directory: URL
        /// Per model, largest first.
        public let models: [ModelUsage]
        /// Files that name no model - an unreadable header, or a trace from an older build for a
        /// model no longer in the catalog. Reported so the totals add up, never deleted implicitly.
        public let unattributedCount: Int
        public let unattributedBytes: Int64

        public var totalBytes: Int64 { models.reduce(0) { $0 + $1.bytes } + unattributedBytes }

        public static let empty = Inventory(
            directory: PrefixKVStore.defaultDirectory, models: [],
            unattributedCount: 0, unattributedBytes: 0
        )
    }

    /// Scan the store and group it by model.
    ///
    /// Reads safetensors *headers* only - never tensor data - so it stays cheap on a multi-GB
    /// directory. Strictly read-only, unlike ``baseCandidates(modelID:directory:)``, which deletes
    /// what it cannot parse: a scan that quietly ate files while the user was looking at them
    /// would be indefensible. Blocking file I/O; call it off the main thread.
    public static func inventory(
        directory: URL = defaultDirectory,
        knownModelIDs: [String] = MlxModel.catalog.map(\.id)
    ) -> Inventory {
        let fm = FileManager.default
        guard let files = try? fm.contentsOfDirectory(
            at: directory, includingPropertiesForKeys: [.fileSizeKey]
        ) else { return Inventory(directory: directory, models: [], unattributedCount: 0, unattributedBytes: 0) }

        var snapshots: [String: (count: Int, bytes: Int64)] = [:]
        var traces: [String: (count: Int, bytes: Int64)] = [:]
        var unattributedCount = 0
        var unattributedBytes: Int64 = 0

        for url in files {
            let isSnapshot = url.pathExtension == "safetensors"
            guard isSnapshot || url.pathExtension == "json" else { continue }
            let bytes = fileSize(url)
            guard let model = owner(of: url, knownModelIDs: knownModelIDs) else {
                unattributedCount += 1
                unattributedBytes += bytes
                continue
            }
            if isSnapshot {
                let entry = snapshots[model, default: (0, 0)]
                snapshots[model] = (entry.count + 1, entry.bytes + bytes)
            } else {
                let entry = traces[model, default: (0, 0)]
                traces[model] = (entry.count + 1, entry.bytes + bytes)
            }
        }

        var names: Set<String> = Set(snapshots.keys)
        names.formUnion(traces.keys)
        var models: [ModelUsage] = []
        for name in names {
            let snapshot = snapshots[name] ?? (count: 0, bytes: Int64(0))
            let trace = traces[name] ?? (count: 0, bytes: Int64(0))
            models.append(ModelUsage(
                modelID: name,
                snapshotCount: snapshot.count, snapshotBytes: snapshot.bytes,
                traceCount: trace.count, traceBytes: trace.bytes
            ))
        }
        // Largest first: the point of the listing is reclaiming space.
        models.sort { $0.bytes == $1.bytes ? $0.modelID < $1.modelID : $0.bytes > $1.bytes }

        return Inventory(
            directory: directory, models: models,
            unattributedCount: unattributedCount, unattributedBytes: unattributedBytes
        )
    }

    /// Delete every artifact belonging to `modelID` - bases and traces alike. Returns bytes freed.
    @discardableResult
    public static func removeAll(
        modelID: String,
        directory: URL = defaultDirectory,
        knownModelIDs: [String] = MlxModel.catalog.map(\.id)
    ) -> Int64 {
        let fm = FileManager.default
        guard let files = try? fm.contentsOfDirectory(
            at: directory, includingPropertiesForKeys: [.fileSizeKey]
        ) else { return 0 }
        var freed: Int64 = 0
        for url in files where url.pathExtension == "safetensors" || url.pathExtension == "json" {
            guard owner(of: url, knownModelIDs: knownModelIDs) == modelID else { continue }
            let bytes = fileSize(url)
            if (try? fm.removeItem(at: url)) != nil { freed += bytes }
        }
        return freed
    }

    /// Empty the store, leaving the directory itself in place. Returns bytes freed.
    @discardableResult
    public static func removeAll(directory: URL = defaultDirectory) -> Int64 {
        let fm = FileManager.default
        guard let files = try? fm.contentsOfDirectory(
            at: directory, includingPropertiesForKeys: [.fileSizeKey]
        ) else { return 0 }
        var freed: Int64 = 0
        for url in files where url.pathExtension == "safetensors" || url.pathExtension == "json" {
            let bytes = fileSize(url)
            if (try? fm.removeItem(at: url)) != nil { freed += bytes }
        }
        return freed
    }

    /// Apply the current limits now, rather than at the next save - what a host calls after the
    /// user lowers ``maxSnapshotsPerModel`` or ``maxTotalBytes``, so the space comes back while
    /// they are still looking at the setting.
    public static func pruneNow(directory: URL = defaultDirectory) {
        prune(directory)
    }

    // MARK: - Internals

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

    /// The model a stored artifact belongs to, or `nil` when nothing identifies it.
    ///
    /// Never parses the file name. ``flatID(_:)`` maps `/` to `--` and the name then carries a
    /// trailing `-<hex>`, so it cannot be split back without a candidate list - and one model id
    /// can prefix another, the same hazard ``baseCandidates(modelID:directory:)`` already guards
    /// against by comparing the header's `model` key. A base is identified by that key; a trace by
    /// the matching key in its payload, falling back for traces written before that key existed to
    /// the *longest* known id whose `flatID` prefixes the name (longest, so a shorter id can never
    /// claim a longer sibling's files).
    static func owner(of url: URL, knownModelIDs: [String]) -> String? {
        switch url.pathExtension {
        case "safetensors":
            return headerMetadata(url: url)?["model"]
        case "json":
            if let data = try? Data(contentsOf: url),
               let payload = try? JSONDecoder().decode([String: String].self, from: data),
               let model = payload["model"] {
                return model
            }
            let name = url.lastPathComponent
            return knownModelIDs
                .filter { name.hasPrefix("\(flatID($0))-") }
                .max { $0.count < $1.count }
        default:
            return nil
        }
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

    /// Keep the ``maxTraces`` most-recently-written traces. Split out from ``prune(_:)`` because
    /// ``saveTrace(tokens:modelID:fingerprint:directory:)`` runs after *every* turn, and must not
    /// pay for a header read of every multi-hundred-MB snapshot to re-bound a set of files only
    /// ``saveBase(cache:tokens:modelID:directory:)`` can grow.
    private static func pruneTraces(_ directory: URL) {
        let fm = FileManager.default
        guard let files = try? fm.contentsOfDirectory(
            at: directory, includingPropertiesForKeys: [.contentModificationDateKey]
        ) else { return }
        let traces = files.filter { $0.pathExtension == "json" }
            .sorted { modified($0) > modified($1) }
        for stale in traces.dropFirst(maxTraces) { try? fm.removeItem(at: stale) }
    }

    /// One base snapshot as ``prune(_:)`` sees it: where it lives, whose it is, and the two facts
    /// the passes sort and budget on.
    private struct StoredSnapshot {
        let url: URL
        let owner: String?
        let bytes: Int64
        let modified: Date
    }

    /// Bound the store, in two passes over the base snapshots (plus the traces).
    ///
    /// Pass 1 keeps the newest ``maxSnapshotsPerModel`` of *each* model, so a model that is used
    /// constantly cannot evict the warm base of one that is used rarely. Pass 2 then bounds the
    /// directory as a whole against ``maxTotalBytes``, because per-model counts say nothing about
    /// bytes - the two limits do different jobs and both are needed.
    ///
    /// Ordering is by modification date descending, which is most-recently-*used*, not merely
    /// most-recently-written: ``seed(modelID:fingerprint:promptTokens:directory:)`` touches a base
    /// it resumes from and ``saveBase(cache:tokens:modelID:directory:)`` touches one it skipped
    /// rewriting, so recency tracks usefulness.
    private static func prune(_ directory: URL, knownModelIDs: [String] = MlxModel.catalog.map(\.id)) {
        pruneTraces(directory)
        let fm = FileManager.default
        guard let files = try? fm.contentsOfDirectory(
            at: directory, includingPropertiesForKeys: [.contentModificationDateKey, .fileSizeKey]
        ) else { return }
        var snapshots: [StoredSnapshot] = []
        for url in files where url.pathExtension == "safetensors" {
            snapshots.append(StoredSnapshot(
                url: url, owner: owner(of: url, knownModelIDs: knownModelIDs),
                bytes: fileSize(url), modified: modified(url)
            ))
        }
        snapshots.sort { $0.modified > $1.modified }

        // Pass 1 - per-model budget. A snapshot whose header won't read groups under its own file
        // name: a budget of one, so a corrupt file can never consume a real model's allowance,
        // and pass 2 can still reclaim it.
        let limit = maxSnapshotsPerModel
        var kept: [String: Int] = [:]
        var survivors: [(url: URL, bytes: Int64)] = []
        for snapshot in snapshots {
            let key = snapshot.owner ?? snapshot.url.lastPathComponent
            let count = kept[key, default: 0]
            if count < limit {
                kept[key] = count + 1
                survivors.append((snapshot.url, snapshot.bytes))
            } else {
                try? fm.removeItem(at: snapshot.url)
            }
        }

        // Pass 2 - byte ceiling, oldest first. The newest snapshot is exempt: it is the base this
        // run just wrote or just resumed from, and evicting it under a cap smaller than one base
        // would re-prefill and rewrite it every single turn.
        let cap = maxTotalBytes
        guard cap > 0, survivors.count > 1 else { return }
        var total = survivors.reduce(Int64(0)) { $0 + $1.bytes }
        var index = survivors.count - 1
        while total > cap, index > 0 {
            try? fm.removeItem(at: survivors[index].url)
            total -= survivors[index].bytes
            index -= 1
        }
    }

    private static func modified(_ url: URL) -> Date {
        (try? url.resourceValues(forKeys: [.contentModificationDateKey]))?
            .contentModificationDate ?? .distantPast
    }

    private static func fileSize(_ url: URL) -> Int64 {
        Int64((try? url.resourceValues(forKeys: [.fileSizeKey]))?.fileSize ?? 0)
    }
}
