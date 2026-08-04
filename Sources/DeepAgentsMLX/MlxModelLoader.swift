import DeepAgents
import Foundation
import HuggingFace
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import MLXVLM
import Tokenizers

/// The shared, app-agnostic model-loading core: turn a Hugging Face model id into a running MLX
/// container (with the LFM2.5 chat-template / projector repairs) and the HF-cache helpers that back
/// download / delete. The terminal harness loads models through an instance of this directly; the
/// app's `MlxModelManager` builds its residency, transcript, and approval orchestration on top of
/// these statics. Kept in the core so the shared base harness can load models without the app.
@MainActor
public final class MlxModelLoader {
    public init() {}

    private var containers: [String: ModelContainer] = [:]
    /// Human-readable reason the most recent ``loadChatModel(_:progress:)`` returned nil, so a CLI
    /// front-end can say *why* a model failed instead of a blanket "failed to load". Cleared on the
    /// next successful load.
    public private(set) var lastLoadError: String?
    /// The id of the model currently cold-loading from disk (nil when nothing is loading), so a UI
    /// can label the wait as a model load - a lazy reload after an idle-unload otherwise looks like
    /// slow prompt processing.
    public private(set) var loadingModelID: String?
    /// Per-model prefix KV-cache slots, so successive `MlxChatModel`s for a resident model reuse the
    /// system+tools prefix across queries (see ``PrefixCacheSlot``). Dropped in ``unload(_:)``.
    private var prefixSlots: [String: PrefixCacheSlot] = [:]

    private func prefixSlot(for id: String) -> PrefixCacheSlot {
        if let slot = prefixSlots[id] { return slot }
        let slot = PrefixCacheSlot()
        prefixSlots[id] = slot
        return slot
    }

    /// Load one model by Hugging Face id and wrap it as an `MlxChatModel` with that model's
    /// recommended agent sampling. Returns nil if the id isn't in the catalog or the load fails.
    /// The container is cached, so loading the same id again reuses the warm copy - used by the
    /// headless scenario harness and the chat REPL to materialize the planner and each subagent.
    public func loadChatModel(_ id: String) async -> MlxChatModel? {
        await loadChatModel(id, progress: { _ in })
    }

    /// As above, but reports load progress (0...1) so a caller can draw a progress bar. A warm
    /// (cached) container reports 1 immediately; a cold one threads the container loader's
    /// download / verify fraction through.
    public func loadChatModel(
        _ id: String, progress: @escaping @Sendable (Double) -> Void
    ) async -> MlxChatModel? {
        guard let model = MlxModel.catalog.first(where: { $0.id == id }) else {
            lastLoadError = "\(id) is not in the on-device model catalog"
            return nil
        }
        let container: ModelContainer
        if let cached = containers[id] {
            container = cached
            progress(1)
        } else {
            loadingModelID = id
            defer { loadingModelID = nil }
            do {
                let loaded = try await Self.loadContainer(id: id, isVision: model.loadsAsVision, progress: progress)
                containers[id] = loaded
                container = loaded
            } catch {
                lastLoadError = Self.describe(error)
                return nil
            }
        }
        lastLoadError = nil
        return MlxChatModel(
            container: container, supportsVision: model.loadsAsVision,
            modelID: model.id, contextWindowTokens: model.contextWindowTokens,
            generateParameters: model.agentParameters, codecFamily: model.codecFamily, startsInReasoning: model.startsInReasoning,
            prefixCache: prefixSlot(for: id)
        )
    }

    // MARK: - Idle residency

    /// Models with a scheduled idle-unload, by id. Cancelled while a model is in active use and
    /// rearmed once it falls idle.
    private var idleTimers: [String: Task<Void, Never>] = [:]
    /// How many in-flight turns are currently using each model. A model is only idle-unloaded when this
    /// hits zero, so a long-running generation is never freed out from under itself.
    private var activeUses: [String: Int] = [:]

    /// Resolve `id` for an in-flight turn: load it if needed (caching the container), cancel any pending
    /// idle unload, and mark it in active use. Returns nil if the model can't load. Pair every call with
    /// ``endUse(_:idleMinutes:)`` so the idle timer is rearmed when the turn finishes.
    public func beginUse(_ id: String) async -> MlxChatModel? {
        idleTimers[id]?.cancel()
        idleTimers[id] = nil
        activeUses[id, default: 0] += 1
        guard let model = await loadChatModel(id) else {
            // Loading failed: undo the active-use claim so a later turn can still idle-unload.
            endActiveUse(id)
            return nil
        }
        return model
    }

    /// Release one in-flight turn's claim on `id`. When the last one finishes, (re)arm the idle timer -
    /// after `idleMinutes` of no use the cached container is dropped and its weights freed.
    /// `idleMinutes <= 0` keeps the model resident (no timer is scheduled).
    public func endUse(_ id: String, idleMinutes: Int) {
        endActiveUse(id)
        guard activeUses[id, default: 0] == 0, idleMinutes > 0 else { return }
        let seconds = Double(idleMinutes) * 60
        idleTimers[id]?.cancel()
        idleTimers[id] = Task { [weak self] in
            try? await Task.sleep(for: .seconds(seconds))
            guard !Task.isCancelled else { return }
            self?.idleUnload(id)
        }
    }

    private func endActiveUse(_ id: String) {
        if let count = activeUses[id] { activeUses[id] = max(0, count - 1) }
    }

    /// Drop `id`'s container if it's still idle (no active use crept in while the timer ran).
    private func idleUnload(_ id: String) {
        idleTimers[id] = nil
        guard activeUses[id, default: 0] == 0 else { return }
        unload(id)
    }

    /// Drop `id`'s cached container and hand the freed buffers back to the OS. Dropping the only strong
    /// reference frees the weights; `MLX.Memory.clearCache()` then returns the buffers so resident
    /// memory actually falls. The next ``beginUse(_:)`` reloads it.
    public func unload(_ id: String) {
        idleTimers[id]?.cancel()
        idleTimers[id] = nil
        containers[id] = nil
        prefixSlots[id] = nil // its KV is tied to the container just dropped
        MLX.Memory.clearCache()
    }

    // MARK: - Container loading

    public nonisolated static func loadContainer(
        id: String, isVision: Bool, progress: @escaping @Sendable (Double) -> Void
    ) async throws -> ModelContainer {
        let factory: any ModelFactory = isVision ? VLMModelFactory.shared : LLMModelFactory.shared
        var configuration = ModelConfiguration(id: id)
        let handler: @Sendable (Progress) -> Void = { progress($0.fractionCompleted) }
        // Pre-download whenever the factory can't fetch the files itself: a text load (so the
        // sidecar filter can inspect the snapshot first) or *any* precision-subfolder id, whose
        // three-component id is not a repo id - handing it to the factory resolves the wrong repo.
        if !isVision || MlxModelID.subfolder(id) != nil, !isDownloadedOnDisk(id) {
            try await downloadSnapshot(id: id, progress: progress)
        }
        // Repair a stale chat template before the tokenizer reads it (some LFM2.5 conversions ship
        // one that can't render tool calls - see below). This runs *after* the download: on a cold
        // first load there is no snapshot yet, so patching first was a silent no-op and a freshly
        // downloaded model ran its first session on the broken template.
        patchLFM2ChatTemplate(repoId: id)
        if let directory = modelDirectory(id: id, isVision: isVision) {
            configuration = ModelConfiguration(directory: directory)
        }
        do {
            return try await factory.loadContainer(
                from: #hubDownloader(), using: #huggingFaceTokenizerLoader(),
                configuration: configuration, progressHandler: handler
            )
        } catch {
            guard describe(error).contains("multi_modal_projector.layer_norm"),
                  patchProjectorUseLayernorm(repoId: id)
            else { throw error }
            return try await factory.loadContainer(
                from: #hubDownloader(), using: #huggingFaceTokenizerLoader(),
                configuration: configuration, progressHandler: handler
            )
        }
    }

    /// The local directory to load `id` from instead of letting the factory resolve a Hub repo, or
    /// nil to load by repo id as before. Two reasons to override: a text load whose snapshot carries
    /// non-indexed sidecar weights (the symlink view), or a precision-subfolder id, whose files live
    /// in `snapshots/<hash>/<subfolder>/`.
    ///
    /// For a subfolder id it must be the *subfolder*, never the snapshot root: mlx-swift-lm's
    /// `loadWeights` enumerates the model directory **recursively**, so the root would merge every
    /// precision's tensors into one weight dict.
    private nonisolated static func modelDirectory(id: String, isVision: Bool) -> URL? {
        if !isVision, let view = sidecarFilteredSnapshot(repoId: id) { return view }
        guard MlxModelID.subfolder(id) != nil else { return nil }
        return snapshotDirectories(id).first {
            FileManager.default.fileExists(atPath: $0.appendingPathComponent("config.json").path)
        }
    }

    /// A filtered view of `repoId`'s downloaded snapshot for text-factory loads, or nil when the
    /// snapshot has no sidecar weight files and can be loaded in place (the common case - the view
    /// is only needed for conversions like Qwen3.6 OptiQ that ship extra non-indexed safetensors).
    ///
    /// For a precision-subfolder id the view is built from that subfolder, and the snapshot hash is
    /// pushed into the view root: ``sidecarFilteredView(snapshot:viewRoot:)`` names the view by the
    /// snapshot's last path component, which for a nested path is the *subfolder* name, so two
    /// precisions of one repo would otherwise land in the same view directory and clobber each other.
    nonisolated static func sidecarFilteredSnapshot(repoId: String) -> URL? {
        let base = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".cache/deepagents/model-views", isDirectory: true)
            .appendingPathComponent(repoId.replacingOccurrences(of: "/", with: "--"), isDirectory: true)
        let nested = MlxModelID.subfolder(repoId) != nil
        for directory in snapshotDirectories(repoId) {
            let viewRoot = nested
                ? base.appendingPathComponent(
                    directory.deletingLastPathComponent().lastPathComponent, isDirectory: true
                )
                : base
            if let view = try? sidecarFilteredView(snapshot: directory, viewRoot: viewRoot) {
                return view
            }
        }
        return nil
    }

    /// Build (or refresh) a symlink view of `snapshot` that omits every `*.safetensors` not listed
    /// in `model.safetensors.index.json` - matching what Python mlx-lm loads. Returns nil when
    /// nothing needs filtering (no index, or no sidecar files). The view lives under
    /// `viewRoot/<snapshot-hash>/` and is rebuilt on every call, so a re-downloaded revision gets a
    /// fresh view automatically.
    nonisolated static func sidecarFilteredView(snapshot: URL, viewRoot: URL) throws -> URL? {
        let fm = FileManager.default
        guard let data = try? Data(contentsOf: snapshot.appendingPathComponent("model.safetensors.index.json")),
              let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let weightMap = root["weight_map"] as? [String: String]
        else { return nil } // no index: a single-file repo, nothing to filter
        let indexed = Set(weightMap.values)
        let files = try fm.contentsOfDirectory(at: snapshot, includingPropertiesForKeys: nil)
        let sidecars = Set(
            files.filter { $0.pathExtension == "safetensors" && !indexed.contains($0.lastPathComponent) }
                .map(\.lastPathComponent)
        )
        guard !sidecars.isEmpty else { return nil }

        let view = viewRoot.appendingPathComponent(snapshot.lastPathComponent, isDirectory: true)
        try? fm.removeItem(at: view)
        try fm.createDirectory(at: view, withIntermediateDirectories: true)
        for file in files where !sidecars.contains(file.lastPathComponent) {
            // Snapshot entries are themselves relative symlinks into the blob store - link the
            // resolved blob so the view works regardless of depth.
            try fm.createSymbolicLink(
                at: view.appendingPathComponent(file.lastPathComponent),
                withDestinationURL: file.resolvingSymlinksInPath()
            )
        }
        return view
    }

    /// Set `projector_use_layernorm: true` in the cached `config.json` for `repoId`
    /// (the weight error happens after download, so the file is already present).
    /// Returns whether anything was changed. Writes through the cache symlink.
    private nonisolated static func patchProjectorUseLayernorm(repoId: String) -> Bool {
        let fm = FileManager.default
        var patched = false
        for hashDir in snapshotDirectories(repoId) {
            let configURL = hashDir.appendingPathComponent("config.json")
            guard fm.fileExists(atPath: configURL.path),
                  let data = try? Data(contentsOf: configURL),
                  var json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  (json["projector_use_layernorm"] as? Bool) == false
            else { continue }
            json["projector_use_layernorm"] = true
            if let out = try? JSONSerialization.data(withJSONObject: json, options: [.prettyPrinted]),
               (try? out.write(to: configURL, options: [])) != nil {
                patched = true
            }
        }
        return patched
    }

    /// Rewrite a stale LFM2.5 chat template in the cache to the canonical one
    /// (`LFM2ChatTemplate.canonical`) so tool use works. Some community MLX conversions of
    /// LFM2.5 - e.g. `mlx-community/LFM2.5-VL-1.6B-8bit` - ship a `chat_template.jinja` that
    /// has no `render_tool_calls` macro: it injects the tool list into the system prompt but
    /// silently drops an assistant turn's `tool_calls` when re-rendering history, so once the
    /// model makes its first tool call that call vanishes from the next round's prompt and the
    /// ReAct loop's tool use collapses (the larger 1.6B model "stops calling tools"). It also
    /// omits LFM2.5's trained `Today's date: …` framing. swift-transformers' tokenizer loader
    /// reads `chat_template.jinja` and merges it over `tokenizer_config.json`, so we write the
    /// canonical template to both. Only LFM2 repos whose cached template lacks tool-call
    /// rendering are touched - a correct template is left as-is, so this self-heals upstream.
    ///
    /// Note this writes *through* the snapshot symlink, mutating the blob. In a subfolder-packaged
    /// repo a `chat_template.jinja` byte-identical across precisions is a single shared blob, so
    /// patching one precision patches them all - which is what you'd want here (same content, same
    /// repair), but it does desync the blob from the etag the hub recorded for it.
    private nonisolated static func patchLFM2ChatTemplate(repoId: String) {
        guard repoId.lowercased().contains("lfm2") else { return }
        let fm = FileManager.default
        for hashDir in snapshotDirectories(repoId) {
            let configURL = hashDir.appendingPathComponent("tokenizer_config.json")
            // Only the snapshot that holds the tokenizer config is one we load from.
            guard fm.fileExists(atPath: configURL.path) else { continue }

            let jinjaURL = hashDir.appendingPathComponent("chat_template.jinja")
            let jinjaTemplate = try? String(contentsOf: jinjaURL, encoding: .utf8)
            let configJSON = (try? Data(contentsOf: configURL)).flatMap {
                try? JSONSerialization.jsonObject(with: $0) as? [String: Any]
            }
            let configTemplate = configJSON?["chat_template"] as? String

            // The loader prefers the standalone `.jinja`; fall back to the config's key.
            let effective = jinjaTemplate ?? configTemplate
            if LFM2ChatTemplate.rendersToolCalls(effective) { continue } // already good

            // Authoritative for the loader: the standalone `.jinja` file.
            try? Data(LFM2ChatTemplate.canonical.utf8).write(to: jinjaURL, options: [])
            // Mirror into the config for any loader that reads only `tokenizer_config.json`.
            if var json = configJSON {
                json["chat_template"] = LFM2ChatTemplate.canonical
                if let out = try? JSONSerialization.data(
                    withJSONObject: json, options: [.prettyPrinted]
                ) {
                    try? out.write(to: configURL, options: [])
                }
            }
        }
    }

    public nonisolated static func describe(_ error: Error) -> String {
        (error as? LocalizedError)?.errorDescription ?? error.localizedDescription
    }

    // MARK: - HF cache helpers

    /// The Hugging Face hub cache root (`$HF_HOME/hub`, else `~/.cache/huggingface/hub`).
    /// The app isn't sandboxed, so this is where `#hubDownloader()` reads/writes.
    private nonisolated static func hubBase() -> URL {
        if let hfHome = ProcessInfo.processInfo.environment["HF_HOME"] {
            return URL(fileURLWithPath: hfHome).appendingPathComponent("hub")
        }
        return FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".cache/huggingface/hub")
    }

    /// The cache directory for `modelID`'s *repo* (`…/hub/models--org--repo`). A precision-subfolder
    /// id resolves to the same directory as its siblings - they share one repo, one `refs/main`, and
    /// one blob store. Internal so ``PrefixKVStore`` can read the downloaded revision (`refs/main`)
    /// for cache keying.
    nonisolated static func hubRepoDirectory(_ modelID: String) -> URL {
        hubBase().appendingPathComponent(
            "models--" + MlxModelID.repo(modelID).replacingOccurrences(of: "/", with: "--")
        )
    }

    /// Every directory a load could read `modelID` from, best revision first: each cached snapshot
    /// hash, or - for a precision-subfolder id - that subfolder inside each snapshot. `refs/main`'s
    /// commit is ordered first so a stale cached revision never wins over the current one, which
    /// also keeps the load path agreeing with ``PrefixKVStore/revisionOnDisk(_:)``.
    nonisolated static func snapshotDirectories(_ modelID: String) -> [URL] {
        snapshotDirectories(modelID, in: hubRepoDirectory(modelID))
    }

    /// The pure core of ``snapshotDirectories(_:)``, against an explicit repo cache directory so
    /// tests can point it at a fixture instead of the user's real Hugging Face cache.
    nonisolated static func snapshotDirectories(_ modelID: String, in repo: URL) -> [URL] {
        guard let hashes = try? FileManager.default.contentsOfDirectory(
            at: repo.appendingPathComponent("snapshots"), includingPropertiesForKeys: nil
        ) else { return [] }
        // A filter-partition, not `sort` - "is the current revision" isn't a strict weak ordering,
        // and Swift's sort can trap on a predicate that isn't one.
        var ordered = hashes
        if let ref = try? String(
            contentsOf: repo.appendingPathComponent("refs/main"), encoding: .utf8
        ) {
            let current = ref.trimmingCharacters(in: .whitespacesAndNewlines)
            ordered = hashes.filter { $0.lastPathComponent == current }
                + hashes.filter { $0.lastPathComponent != current }
        }
        guard let subfolder = MlxModelID.subfolder(modelID) else { return ordered }
        return ordered.map { $0.appendingPathComponent(subfolder, isDirectory: true) }
    }

    /// Whether `modelID`'s weights are present in the cache (some snapshot - or, for a
    /// precision-subfolder id, that subfolder of some snapshot - holds a `.safetensors` file).
    /// A pure filesystem check - no network.
    public nonisolated static func isDownloadedOnDisk(_ modelID: String) -> Bool {
        hasWeights(modelID, in: hubRepoDirectory(modelID))
    }

    /// The pure core of ``isDownloadedOnDisk(_:)``, against an explicit repo cache directory so
    /// tests can point it at a fixture instead of the user's real Hugging Face cache. A sibling
    /// precision can never satisfy this: only `<subfolder>/` is read, never the snapshot root.
    nonisolated static func hasWeights(_ modelID: String, in repoDirectory: URL) -> Bool {
        let fm = FileManager.default
        let subfolder = MlxModelID.subfolder(modelID)
        guard let hashes = try? fm.contentsOfDirectory(
            at: repoDirectory.appendingPathComponent("snapshots"), includingPropertiesForKeys: nil
        ) else { return false }
        for hash in hashes {
            let directory = subfolder.map { hash.appendingPathComponent($0, isDirectory: true) } ?? hash
            if let files = try? fm.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil),
               files.contains(where: { $0.pathExtension == "safetensors" }) {
                return true
            }
        }
        return false
    }

    /// Remove `modelID`'s local files from the cache.
    public nonisolated static func removeFromDisk(_ modelID: String) {
        remove(modelID, from: hubRepoDirectory(modelID))
    }

    /// The pure core of ``removeFromDisk(_:)``, against an explicit repo cache directory so tests
    /// can point it at a fixture.
    ///
    /// A plain repo id drops the whole `models--org--repo` directory, as before. A
    /// precision-subfolder id must not: its siblings live in the same directory and share one
    /// content-addressed blob store, so every file byte-identical across precisions
    /// (`tokenizer.json`, `tokenizer_config.json`, usually `chat_template.jinja` and
    /// `generation_config.json`) is *one* blob with several snapshot symlinks pointing at it.
    /// Deleting the subfolder alone would free essentially nothing - the bytes live in `blobs/` -
    /// while deleting its blobs unconditionally would break the siblings. So: drop the subfolder
    /// from every snapshot, then delete exactly the blobs no surviving entry still resolves to.
    /// When that leaves no precision behind, the repo directory goes too, rather than accumulating
    /// empty shells.
    nonisolated static func remove(_ modelID: String, from repoDirectory: URL) {
        let fm = FileManager.default
        guard let subfolder = MlxModelID.subfolder(modelID) else {
            try? fm.removeItem(at: repoDirectory)
            return
        }
        let snapshots = repoDirectory.appendingPathComponent("snapshots")
        let blobs = repoDirectory.appendingPathComponent("blobs")
        guard let hashes = try? fm.contentsOfDirectory(at: snapshots, includingPropertiesForKeys: nil)
        else { return }

        // Resolve the blobs this precision references *before* unlinking the symlinks to them.
        var candidates = Set<String>()
        for hash in hashes {
            let directory = hash.appendingPathComponent(subfolder, isDirectory: true)
            candidates.formUnion(resolvedEntries(under: directory, in: blobs))
            try? fm.removeItem(at: directory)
        }
        guard !candidates.isEmpty else { return }

        // Everything the surviving precisions still reference, across every snapshot.
        var live = Set<String>()
        for hash in hashes { live.formUnion(resolvedEntries(under: hash, in: blobs)) }
        for path in candidates.subtracting(live) { try? fm.removeItem(at: URL(fileURLWithPath: path)) }

        let remaining = hashes.contains { (try? fm.contentsOfDirectory(atPath: $0.path))?.isEmpty == false }
        if !remaining { try? fm.removeItem(at: repoDirectory) }
    }

    /// The blob paths that snapshot entries under `directory` resolve to. Entries are relative
    /// symlinks into `blobs/`; an entry that was *copied* instead (a filesystem without symlink
    /// support) resolves to itself, falls outside `blobs/`, and is ignored - it is deleted with its
    /// subfolder and owns no shared bytes.
    private nonisolated static func resolvedEntries(under directory: URL, in blobs: URL) -> [String] {
        let root = blobs.standardizedFileURL.path
        guard let walker = FileManager.default.enumerator(
            at: directory, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles]
        ) else { return [] }
        return walker.compactMap { entry in
            guard let url = entry as? URL else { return nil }
            let resolved = url.resolvingSymlinksInPath().standardizedFileURL.path
            return resolved.hasPrefix(root + "/") ? resolved : nil
        }
    }

    /// Total bytes of the URLSession download temp files written since `startedAt` — the live
    /// in-flight bytes of an active download (each file streams to a `CFNetworkDownload_*.tmp` in
    /// the process temp dir before being moved into the HF cache). The modification-date filter
    /// skips stale leftovers from earlier downloads. No network. This is the only reliable
    /// progress source: the hub's Xet transport reports no incremental progress, so download UIs
    /// poll this instead of (or blended with) the library's progress callback.
    public nonisolated static func inFlightDownloadBytes(since startedAt: Date) -> Int64 {
        let tmp = URL(fileURLWithPath: NSTemporaryDirectory())
        let keys: Set<URLResourceKey> = [.fileSizeKey, .contentModificationDateKey]
        guard let files = try? FileManager.default.contentsOfDirectory(
            at: tmp, includingPropertiesForKeys: Array(keys)
        ) else { return 0 }
        var total: Int64 = 0
        for url in files where url.lastPathComponent.hasPrefix("CFNetworkDownload") {
            guard let values = try? url.resourceValues(forKeys: keys),
                  let modified = values.contentModificationDate, modified >= startedAt
            else { continue }
            total += Int64(values.fileSize ?? 0)
        }
        return total
    }

    /// The hub glob patterns that fetch exactly `modelID`'s files, scoped to its precision subfolder
    /// when it has one.
    ///
    /// The scoping is load-bearing, not tidiness: the hub matches globs with
    /// `fnmatch(glob, path, 0)` - **no** `FNM_PATHNAME` - so `*` crosses `/` and a bare
    /// `*.safetensors` matches *every* precision in a subfolder-packaged repo. On
    /// `LiquidAI/LFM2.5-2.6B-MLX` that turns a 1.6 GB mxfp4 pull into a ~25 GB one. Do not
    /// "simplify" this back to the unprefixed patterns.
    nonisolated static func downloadPatterns(_ modelID: String) -> [String] {
        let extensions = ["*.safetensors", "*.json", "*.jinja"]
        guard let subfolder = MlxModelID.subfolder(modelID) else { return extensions }
        return extensions.map { "\(subfolder)/\($0)" }
    }

    /// Download `id`'s model + tokenizer files to the cache without instantiating it. Reuses
    /// the same `#hubDownloader()` the loader uses; `useLatest: false` returns the cached copy
    /// if already complete. A precision-subfolder id downloads only that subfolder, into the shared
    /// repo cache directory - the Hub is asked for the *repo*, never for the three-component id
    /// (upstream's `Repo.ID` would silently accept it and 404).
    public nonisolated static func downloadSnapshot(
        id: String, progress: @escaping @Sendable (Double) -> Void
    ) async throws {
        let handler: @Sendable (Progress) -> Void = { progress($0.fractionCompleted) }
        _ = try await #hubDownloader().download(
            id: MlxModelID.repo(id), revision: nil,
            matching: downloadPatterns(id),
            useLatest: false, progressHandler: handler
        )
    }
}
