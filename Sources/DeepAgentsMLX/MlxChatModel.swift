import DeepAgents
import Foundation
import MLX
import MLXLMCommon

/// Reusable KV cache for the *stable prompt prefix* (system prompt + tool schemas, which are
/// identical across a run's rounds and across queries for a given model+config), plus the prompt
/// tokens it currently represents. Lets ``RebuildTurnSession`` re-prefill only the suffix that
/// changed instead of the whole ~10k-token prompt every round. Its owner decides the reuse scope: a
/// session makes its own (within-run reuse), or `MlxModelManager` / `MlxModelLoader` hands a
/// per-model one to each ``MlxChatModel`` (cross-query reuse while the model stays resident).
///
/// `@unchecked Sendable`: the MLX-backed caches are only ever touched inside
/// `ModelContainer.perform` (which serialises access per model) and a session is driven from a
/// single task.
public final class PrefixCacheSlot: @unchecked Sendable {
    /// A `copy()`-snapshot of the KV/SSM state after prefilling exactly `tokens`.
    struct Snapshot {
        let cache: [KVCache]
        let tokens: [Int]
    }

    /// Live cache reused by *trimming* - only valid for models whose every layer cache is
    /// trimmable. All current catalog models are attention+Mamba hybrids (non-trimmable), which
    /// use the snapshots below instead.
    var cache: [KVCache]?
    /// The previous round's full prompt tokens - the reference for the next round's common prefix.
    var tokens: [Int] = []
    /// Snapshot at the first observed cross-prompt divergence (in practice the system+tools
    /// boundary), so a *new* query's first round prefills only the conversation.
    var base: Snapshot?
    /// Snapshot just before the end of the latest prompt (only the trailing generation header
    /// changes round-to-round), so the run's next round prefills only its delta.
    var tip: Snapshot?
    var fingerprint: Int?
    var containerID: ObjectIdentifier?
    /// Whether ``PrefixKVStore`` was already consulted for this slot's prompt - the disk is
    /// read at most once per slot life; a reset (config change, image turn) re-arms it.
    var diskChecked = false
    /// Set when a *new* base was established this run (not loaded from disk), so the session
    /// persists it once after generation.
    var baseIsDirty = false
    /// Whether ``tokens`` was seeded from an on-disk trace (vs a live previous prompt). Gates
    /// the one-time base *deepening* after a shared-base resume - live prompts must never deepen
    /// (a re-rendered assistant turn diverges mid-conversation every round, and deepening there
    /// would persist a conversation-laden snapshot per round). Cleared once consumed.
    var diskTrace = false

    public init() {}

    func reset() {
        cache = nil
        tokens = []
        base = nil
        tip = nil
        diskChecked = false
        baseIsDirty = false
        diskTrace = false
    }
}

/// A `ChatModel` over an in-process `mlx-swift-lm` `ModelContainer`.
///
/// Each run gets one `RebuildTurnSession`. `ReactAgent` drives the ReAct loop and, each round, hands
/// the session the full conversation; the session rebuilds the prompt from those structured messages
/// and generates one pass. Rebuilding the prompt every round keeps the chat template faithful
/// (assistant tool calls and tool results included), automatically drops the model's own historical
/// `<think>` blocks, and lets middleware rewrite history between rounds.
///
/// For speed the session reuses the KV cache of the *unchanged token prefix* across rounds (and, when
/// the owner supplies a shared ``PrefixCacheSlot``, across queries on a resident model): only the
/// suffix that differs from the previous prompt is re-prefilled. Because reuse is keyed on the
/// longest common *token* prefix, any change (a dropped `<think>`, a middleware rewrite, a new user
/// turn, or images) simply falls after the shared prefix and is re-prefilled - so caching stays
/// transparent to the codecs.
public struct MlxChatModel: ChatModel {
    let container: ModelContainer
    public let supportsVision: Bool
    /// Hugging Face repo id of the loaded model, recorded on each logged message so a
    /// transcript says which model produced it. `nil` only in tests/previews.
    public var modelID: String?
    /// The model's context window in tokens (from ``MlxModel/contextWindowTokens``), so
    /// summarization's 85% trigger and the context meter measure against the real budget.
    public var contextWindowTokens: Int?
    // Conservative default: Liquid's recommended sampling for the LFM2.5 1.2B instruct
    // models (temperature 0.1, top-k 50, repetition penalty 1.05) for reliable tool use.
    // Production call sites override this with the per-model `MlxModel.agentParameters`
    // (e.g. 8B-A1B wants 0.2 / top-k 80, Thinking adds top-p 0.1). Generous token budget
    // because reasoning models emit long `<think>` blocks and the tool loop re-generates
    // per round, so a small cap truncates the final answer.
    var generateParameters: GenerateParameters = .init(
        maxTokens: 4096, temperature: 0.1, topK: 50, repetitionPenalty: 1.05
    )
    /// Which wire codec drives this model (LFM2 tags vs qwen3_5 / Ornith). Defaults to `.lfm2` (the
    /// historical catalog); set from ``MlxModel/codecFamily`` at the load sites.
    public var codecFamily: MlxCodecFamily = .lfm2
    /// A shared, per-model prefix cache for *cross-query* KV reuse: the owner (`MlxModelManager` /
    /// `MlxModelLoader`) hands the same slot to each `MlxChatModel` it builds for a resident model, so
    /// a new query reuses the previous query's system+tools prefix. `nil` means within-run reuse only
    /// (the session makes its own slot). See ``PrefixCacheSlot``.
    public var prefixCache: PrefixCacheSlot?

    public init(
        container: ModelContainer,
        supportsVision: Bool,
        modelID: String? = nil,
        contextWindowTokens: Int? = nil,
        generateParameters: GenerateParameters,
        codecFamily: MlxCodecFamily = .lfm2,
        prefixCache: PrefixCacheSlot? = nil
    ) {
        self.container = container
        self.supportsVision = supportsVision
        self.modelID = modelID
        self.contextWindowTokens = contextWindowTokens
        self.generateParameters = generateParameters
        self.codecFamily = codecFamily
        self.prefixCache = prefixCache
    }

    public func makeSession() -> any ModelTurnSession {
        RebuildTurnSession(
            container: container,
            supportsVision: supportsVision,
            generateParameters: generateParameters,
            codecFamily: codecFamily,
            prefixCache: prefixCache ?? PrefixCacheSlot(),
            modelID: modelID
        )
    }
}

/// A decoder that collects tool calls the library's built-in parser already extracted from the
/// stream (`Generation.toolCall` events). Adopted by the families that keep the inferred parser
/// active (``Qwen35Decoder``, ``Gemma4Decoder``); LFM2 parses its own tags and never sees these.
public protocol ToolCallIngesting: AnyObject {
    func ingestToolCall(_ call: ToolCall)
}

/// One run's model node: each `nextTurn` rebuilds the prompt from the supplied messages and
/// generates one pass, reusing the KV of the unchanged token prefix via ``prefixCache`` (see
/// ``PrefixCacheSlot`` and ``MlxChatModel``).
///
/// A reference type to satisfy `ModelTurnSession: AnyObject`; `ReactAgent` drives it from a single
/// task, so the per-run cache state it holds through ``prefixCache`` is touched serially and only
/// inside `ModelContainer.perform`.
public final class RebuildTurnSession: ModelTurnSession {
    private let container: ModelContainer
    private let supportsVision: Bool
    private let generateParameters: GenerateParameters
    private let codecFamily: MlxCodecFamily
    /// Prefix KV cache reused across this session's rounds; shared by the owner for cross-query reuse.
    private let prefixCache: PrefixCacheSlot
    /// Hugging Face repo id of the loaded model, keying the on-disk base snapshot
    /// (``PrefixKVStore``); nil (tests/previews) disables disk persistence.
    private let modelID: String?

    public init(
        container: ModelContainer, supportsVision: Bool, generateParameters: GenerateParameters,
        codecFamily: MlxCodecFamily = .lfm2, prefixCache: PrefixCacheSlot = PrefixCacheSlot(),
        modelID: String? = nil
    ) {
        self.container = container
        self.supportsVision = supportsVision
        self.generateParameters = generateParameters
        self.codecFamily = codecFamily
        self.prefixCache = prefixCache
        self.modelID = modelID
    }

    public func nextTurn(
        messages: [AgentMessage],
        systemPrompt: String?,
        tools: [any AgentTool],
        onChunk: @escaping @Sendable (AgentStreamChunk) -> Void
    ) async throws -> AgentMessage {
        // Encode the canonical history to the model's wire shape; the codec owns all format quirks.
        let family = codecFamily
        let request = Self.encode(
            family: family, messages: messages, systemPrompt: systemPrompt,
            tools: tools, supportsVision: supportsVision
        )
        let supportsVision = supportsVision
        let parameters = generateParameters
        // Prefix-cache inputs captured for the `@Sendable` closure: the slot to reuse/update, plus the
        // keys that must match for its KV to stay valid (same resident container, same system+tools).
        let slot = prefixCache
        let containerID = ObjectIdentifier(container)
        let fingerprint = Self.prefixFingerprint(systemPrompt: systemPrompt, tools: tools)
        let modelID = modelID

        // Generate inside the container's lock; resolve image URLs to `UserInput.Image` here (off the
        // message-building path, which stays `Sendable`).
        //
        // Tool-call parsing differs by family. For LFM2 we *suppress* mlx-swift-lm's built-in parser
        // (its Pythonic parser truncates list/dict arguments at the first comma) by forcing a
        // non-matching `.json` `toolCallFormat`, so the raw `<|tool_call_start|>…<|tool_call_end|>`
        // text reaches the LFM2 decoder, which strips those spans and parses the calls itself. For
        // qwen3_5 / Ornith and gemma4 we do the opposite: keep the container's inferred format
        // (`.xmlFunction` / `.gemma4`), so the library's correct parser strips the tool spans and
        // surfaces each call as a `Generation.toolCall` event - the decoders there only split the
        // reasoning channel.
        return try await container.perform { context in
            let images: [UserInput.Image] = supportsVision ? request.imageURLs.map { .url($0) } : []
            // Gemma 4's reasoning is opt-in via the chat template: `enable_thinking` opens the
            // first system turn with `<|think|>`. Constant per family, so prefix fingerprints
            // and cached prompt prefixes are unaffected.
            let userInput = UserInput(
                messages: request.messages, images: images,
                tools: request.toolSpecs.isEmpty ? nil : request.toolSpecs,
                additionalContext: family == .gemma4 ? ["enable_thinking": true] : nil
            )
            let lmInput = try await context.processor.prepare(input: userInput)

            var configuration = context.configuration
            if family == .lfm2 {
                configuration.toolCallFormat = .json // ≠ LFM2's tags → built-in parser stands down
            }

            // Drop the reusable prefix KV if the resident model was swapped or the system+tools prefix
            // changed (the cached tokens would no longer be a valid prefix of this prompt).
            if slot.fingerprint != fingerprint || slot.containerID != containerID {
                slot.reset()
                slot.fingerprint = fingerprint
                slot.containerID = containerID
            }
            let iterator = try Self.makeIterator(
                lmInput: lmInput, hasImages: !images.isEmpty,
                slot: slot, model: context.model, parameters: parameters, modelID: modelID
            )
            // `tools:` gives the built-in parser the schemas it types parameter values with -
            // without them the qwen3_5 XML parser leaves every nested array/object argument as a
            // raw string (e.g. ask_user's `questions` arrived as one JSON string and parsed to
            // zero questions). Inert for LFM2, whose forced `.json` format never matches.
            let (stream, task) = generateTask(
                promptTokenCount: lmInput.text.tokens.size,
                modelConfiguration: configuration,
                tokenizer: context.tokenizer,
                iterator: iterator,
                tools: request.toolSpecs.isEmpty ? nil : request.toolSpecs
            )

            let decoder = Self.makeDecoder(family: family)
            for await generation in stream {
                switch generation {
                case .chunk(let chunk):
                    for piece in decoder.ingest(chunk) { onChunk(piece) }
                case .toolCall(let call):
                    // The qwen3_5 and gemma4 paths leave the built-in parser active; their decoders
                    // collect the already-parsed call. (No-op for LFM2, which never sees `.toolCall`.)
                    (decoder as? any ToolCallIngesting)?.ingestToolCall(call)
                default:
                    break
                }
            }
            let (trailing, message) = decoder.finish()
            for piece in trailing { onChunk(piece) }
            await task.value
            // After generation (still serialized by `perform`): persist a newly established base
            // snapshot, or - until one exists - the prompt-token trace the next process needs to
            // find the base boundary. One ~O(100MB) write per shared prefix, ever (bases are
            // content-addressed, so configs sharing a prefix share the file).
            Self.persistPrefix(slot: slot, modelID: modelID)
            return message
        }
    }

    /// Write the slot's freshly established base snapshot (if any) and this config's token trace
    /// through ``PrefixKVStore``. The trace is written every turn: it is how a later process
    /// learns this config's own prefix boundary, even after resuming from a shorter shared base.
    /// No-ops for anonymous models (tests/previews).
    private static func persistPrefix(slot: PrefixCacheSlot, modelID: String?) {
        guard let modelID, let fingerprint = slot.fingerprint else { return }
        if let base = slot.base, slot.baseIsDirty {
            PrefixKVStore.saveBase(cache: base.cache, tokens: base.tokens, modelID: modelID)
            slot.baseIsDirty = false
        }
        if !slot.tokens.isEmpty {
            PrefixKVStore.saveTrace(tokens: slot.tokens, modelID: modelID, fingerprint: fingerprint)
        }
    }

    /// Encode the history with the codec for `family`. All codecs share `LFM2Request`, so the
    /// container-side transport below is family-agnostic.
    private static func encode(
        family: MlxCodecFamily, messages: [AgentMessage], systemPrompt: String?,
        tools: [any AgentTool], supportsVision: Bool
    ) -> LFM2Request {
        switch family {
        case .lfm2:
            return LFM2MessageCodec().encode(
                messages, systemPrompt: systemPrompt, tools: tools, supportsVision: supportsVision
            )
        case .qwen35:
            return Qwen35MessageCodec().encode(
                messages, systemPrompt: systemPrompt, tools: tools, supportsVision: supportsVision
            )
        case .gemma4:
            return Gemma4MessageCodec().encode(
                messages, systemPrompt: systemPrompt, tools: tools, supportsVision: supportsVision
            )
        }
    }

    /// A fresh per-turn decoder for `family`.
    private static func makeDecoder(family: MlxCodecFamily) -> any TurnDecoder<String> {
        switch family {
        case .lfm2: return LFM2MessageCodec().makeDecoder()
        case .qwen35: return Qwen35MessageCodec().makeDecoder()
        case .gemma4: return Gemma4MessageCodec().makeDecoder()
        }
    }

    /// The tip snapshot sits this many tokens before the prompt's end: between rounds only the
    /// trailing generation header (and the re-rendered last assistant turn) changes, so a snapshot
    /// taken just short of the end stays a valid prefix of the next round's prompt.
    private static let tipMargin = 64
    /// Snapshots shorter than this aren't worth the copy.
    private static let minSnapshotTokens = 64

    /// Build this round's token iterator, reusing the KV/SSM state of the prompt's unchanged
    /// prefix via `slot`. Trimmable caches are trimmed back to the shared prefix; models that
    /// can't rewind - attention+Mamba hybrids (their recurrent state has no history) and
    /// sliding-window models like Gemma 4 (a `RotatingKVCache` claims trim success even after the
    /// window rotated the tokens away, so trimming would silently resume from a corrupt prefix) -
    /// resume from `copy()`-snapshots instead (see ``PrefixCacheSlot``). Falls back to a fresh
    /// full prefill on an image turn or when nothing matches. Updates `slot` in place.
    private static func makeIterator(
        lmInput: LMInput, hasImages: Bool, slot: PrefixCacheSlot,
        model: any LanguageModel, parameters: GenerateParameters, modelID: String? = nil
    ) throws -> TokenIterator {
        let promptTokens = lmInput.text.tokens.asArray(Int.self)

        // Image turns can't reuse token-prefix KV: the VLM processor injects image embeddings that
        // aren't captured by the text token ids. Start fresh and clear the slot.
        if hasImages {
            slot.reset()
            return try TokenIterator(input: lmInput, model: model, cache: nil, parameters: parameters)
        }

        if supportsTrimReuse(model.newCache(parameters: parameters)) {
            return try makeTrimmingIterator(
                lmInput: lmInput, promptTokens: promptTokens, slot: slot,
                model: model, parameters: parameters
            )
        }
        return try makeSnapshotIterator(
            promptTokens: promptTokens, prototype: lmInput.text.tokens, slot: slot,
            model: model, parameters: parameters, modelID: modelID
        )
    }

    /// True when every cache can be *safely* rewound with `trim` to reuse a shared prefix.
    /// `isTrimmable` alone is not enough: a `RotatingKVCache` (Gemma 4's sliding-window layers)
    /// answers true while its window hasn't filled, and its `trim` reports the full requested
    /// count even after rotation has discarded the tokens - so the trim-mismatch fallback in
    /// `trimToPrefix` would never engage and the turn would silently resume from a corrupt
    /// prefix. Any model carrying one is routed to the `copy()`-snapshot path, which only ever
    /// moves forward.
    static func supportsTrimReuse(_ caches: [KVCache]) -> Bool {
        caches.allSatisfy { $0.isTrimmable && !($0 is RotatingKVCache) }
    }

    /// Rebuild a token slice as an `MLXArray` with the same rank as the original prompt array -
    /// VLM processors emit 2-D `(1, seqLen)` prompts and their models index the batch dimension,
    /// while plain LLMs use 1-D prompts.
    private static func tokenArray(_ tokens: ArraySlice<Int>, like prototype: MLXArray) -> MLXArray {
        let array = MLXArray(tokens.map { Int32($0) }).asType(prototype.dtype)
        return prototype.ndim == 2 ? array.expandedDimensions(axis: 0) : array
    }

    /// Reuse for fully-trimmable caches: trim the live cache back to the longest token prefix
    /// shared with the new prompt and prefill only the differing suffix.
    private static func makeTrimmingIterator(
        lmInput: LMInput, promptTokens: [Int], slot: PrefixCacheSlot,
        model: any LanguageModel, parameters: GenerateParameters
    ) throws -> TokenIterator {
        if let cache = slot.cache, !slot.tokens.isEmpty {
            let reuseLen = min(commonPrefixLength(slot.tokens, promptTokens), promptTokens.count - 1)
            if reuseLen > 0, trimToPrefix(cache: cache, reuseLen: reuseLen) {
                let iterator = try TokenIterator(
                    input: LMInput(tokens: tokenArray(promptTokens[reuseLen...], like: lmInput.text.tokens)),
                    model: model, cache: cache, parameters: parameters
                )
                slot.tokens = promptTokens
                return iterator
            }
        }
        // First round or unclean trim: prefill the whole prompt against a fresh cache.
        let cache = model.newCache(parameters: parameters)
        let iterator = try TokenIterator(input: lmInput, model: model, cache: cache, parameters: parameters)
        slot.cache = cache
        slot.tokens = promptTokens
        return iterator
    }

    /// Reuse for hybrid models: resume from the deepest `copy()`-snapshot that is still a strict
    /// prefix of this prompt (`tip`, else `base`), then advance the tip to just before this
    /// prompt's end so the next round prefills only its delta. On a miss, the divergence point
    /// against the previous prompt becomes the new `base` (the cross-query-stable boundary).
    private static func makeSnapshotIterator(
        promptTokens: [Int], prototype: MLXArray, slot: PrefixCacheSlot,
        model: any LanguageModel, parameters: GenerateParameters, modelID: String? = nil
    ) throws -> TokenIterator {
        // A fresh slot (new process, or reset by a config change / image turn) first consults the
        // on-disk store: the longest persisted base that strict-prefixes this prompt resumes
        // directly, whichever config wrote it (bases are content-addressed). The best previous
        // prompt on record (this config's trace, or another base's tokens) seeds `tokens`, so the
        // divergence math below can establish a base - or deepen a resumed shared one - at a
        // boundary future runs can reuse.
        if slot.base == nil, slot.tokens.isEmpty, !slot.diskChecked, let modelID,
           let fingerprint = slot.fingerprint {
            slot.diskChecked = true
            let seed = PrefixKVStore.seed(
                modelID: modelID, fingerprint: fingerprint, promptTokens: promptTokens
            )
            if let base = seed.base { slot.base = .init(cache: base.cache, tokens: base.tokens) }
            if let trace = seed.trace {
                slot.tokens = trace
                slot.diskTrace = true
            }
        }

        let working: [KVCache]
        var start = 0

        if let tip = slot.tip, isStrictPrefix(tip.tokens, of: promptTokens) {
            working = snapshotCopy(tip.cache)
            start = tip.tokens.count
        } else if let base = slot.base, isStrictPrefix(base.tokens, of: promptTokens) {
            working = snapshotCopy(base.cache)
            start = base.tokens.count
            slot.tip = nil
            // The resumed base may be a *shared* prefix from another config (disk bases are
            // content-addressed) - shorter than this config's own stable prefix. If the trace
            // on record shares meaningfully more than the base covers, deepen the base to that
            // divergence, so future runs of this config resume past the shared part too. Only a
            // disk trace qualifies: live previous prompts diverge mid-conversation (re-rendered
            // assistant turns) and must not be snapshotted.
            let baseLen = min(commonPrefixLength(slot.tokens, promptTokens), promptTokens.count - 1)
            if slot.diskTrace, baseLen - start >= minSnapshotTokens {
                try prefill(promptTokens[start ..< baseLen], like: prototype, into: working, model: model, parameters: parameters)
                slot.base = .init(cache: snapshotCopy(working), tokens: Array(promptTokens[..<baseLen]))
                slot.baseIsDirty = true
                start = baseLen
            }
        } else {
            working = model.newCache(parameters: parameters)
            slot.tip = nil
            // Where this prompt diverges from the previous one is, by observation, the boundary
            // that stays stable across prompts (system + tools) - snapshot it as the new base.
            let baseLen = min(commonPrefixLength(slot.tokens, promptTokens), promptTokens.count - 1)
            if baseLen >= minSnapshotTokens {
                try prefill(promptTokens[..<baseLen], like: prototype, into: working, model: model, parameters: parameters)
                slot.base = .init(cache: snapshotCopy(working), tokens: Array(promptTokens[..<baseLen]))
                slot.baseIsDirty = true // established here (not read from disk): persist after the turn
                start = baseLen
            } else {
                slot.base = nil
            }
        }

        // Advance the tip to just before this prompt's end for the next round.
        let tipLen = promptTokens.count - tipMargin
        if tipLen > start, tipLen >= minSnapshotTokens {
            try prefill(promptTokens[start ..< tipLen], like: prototype, into: working, model: model, parameters: parameters)
            slot.tip = .init(cache: snapshotCopy(working), tokens: Array(promptTokens[..<tipLen]))
            start = tipLen
        }

        slot.tokens = promptTokens
        slot.diskTrace = false // consumed: from here on `tokens` is a live prompt
        return try TokenIterator(
            input: LMInput(tokens: tokenArray(promptTokens[start...], like: prototype)),
            model: model, cache: working, parameters: parameters
        )
    }

    /// Prefill `tokens` into `cache` without generating: `TokenIterator`'s init runs the whole
    /// prefill (chunked by `parameters.prefillStepSize`), so the iterator itself is discarded.
    /// Afterwards the cache state covers exactly `tokens`.
    private static func prefill(
        _ tokens: ArraySlice<Int>, like prototype: MLXArray, into cache: [KVCache],
        model: any LanguageModel, parameters: GenerateParameters
    ) throws {
        _ = try TokenIterator(
            input: LMInput(tokens: tokenArray(tokens, like: prototype)),
            model: model, cache: cache, parameters: parameters
        )
    }

    /// Independent deep copies of every layer cache (supported by all cache types, `MambaCache`
    /// included) - the snapshot survives while generation mutates the original.
    private static func snapshotCopy(_ cache: [KVCache]) -> [KVCache] {
        cache.map { $0.copy() }
    }

    /// True when `prefix` is a non-empty strict token prefix of `tokens`.
    static func isStrictPrefix(_ prefix: [Int], of tokens: [Int]) -> Bool {
        !prefix.isEmpty && prefix.count < tokens.count
            && commonPrefixLength(prefix, tokens) == prefix.count
    }

    /// Trim `cache` back to its first `reuseLen` tokens. Returns false if any entry can't be
    /// trimmed exactly, so the caller falls back to a full prefill rather than risk a corrupt
    /// cache. `drop` is never negative (`reuseLen <= offset`).
    private static func trimToPrefix(cache: [KVCache], reuseLen: Int) -> Bool {
        let drop = cache[0].offset - reuseLen
        if drop > 0 {
            for entry in cache where entry.trim(drop) != drop { return false }
        }
        return true
    }

    /// Length of the longest shared prefix of two token sequences.
    static func commonPrefixLength(_ a: [Int], _ b: [Int]) -> Int {
        let n = min(a.count, b.count)
        var i = 0
        while i < n, a[i] == b[i] { i += 1 }
        return i
    }

    /// Identity of the reusable prefix: its KV is valid only while the system prompt and tool set are
    /// unchanged. Order-sensitive over tool names, matching the rendered prompt. Stable across
    /// processes (unlike `Hasher`) because it also keys ``PrefixKVStore``'s on-disk snapshots.
    private static func prefixFingerprint(systemPrompt: String?, tools: [any AgentTool]) -> Int {
        PrefixKVStore.fingerprint(systemPrompt: systemPrompt, toolNames: tools.map(\.name))
    }
}
