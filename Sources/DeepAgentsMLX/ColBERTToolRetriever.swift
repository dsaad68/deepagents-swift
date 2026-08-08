import DeepAgents
import Foundation
import HuggingFace
import MLX
import MLXEmbedders
import MLXHuggingFace
import MLXLMCommon
import Tokenizers

/// The retrieval models `search_tools` can run. Both are LFM2.5-ColBERT-350M - a bidirectional LFM2
/// encoder with a per-token Dense projection to 128 dimensions, scored by MaxSim late interaction.
///
/// 8-bit is the default: retrieval quality is close to bf16 for this ranking job and it halves the
/// resident footprint, which matters because the retriever is co-resident with the planner in unified
/// memory.
public enum ToolSearchModel: String, Sendable, CaseIterable, Codable {
    case colbert350m8bit = "mlx-community/LFM2.5-ColBERT-350M-8bit"
    case colbert350mBF16 = "mlx-community/LFM2.5-ColBERT-350M-bf16"

    public static let `default` = ToolSearchModel.colbert350m8bit

    /// Label for the `/config` editor.
    public var label: String {
        switch self {
        case .colbert350m8bit: "ColBERT 350M (8-bit)"
        case .colbert350mBF16: "ColBERT 350M (bf16)"
        }
    }

    /// The Hugging Face repo id. Also a ``MlxModel/retrieverCatalog`` id, which is what makes these
    /// downloadable and deletable from `/model` → Local like any other model; size lives there.
    public var repoID: String { rawValue }

    /// This model's catalog row, for its size and download state.
    public var catalogEntry: MlxModel? {
        MlxModel.retrieverCatalog.first { $0.id == repoID }
    }
}

/// ColBERT late-interaction retrieval over the auxiliary tool corpus, on device.
///
/// ## How it scores
///
/// Query and documents are each encoded to **per-token** 128-d vectors (`LFM2BidirectionalModel`'s
/// `colbert` head), L2-normalized, and scored by MaxSim - for every query token, the best-matching
/// document token, summed:
///
/// ```text
/// score(Q, D) = (1/|Q|) Σ_i max_j (q_i · d_j)
/// ```
///
/// Dividing by the query length is cosmetic (it keeps scores in `[-1, 1]` so they read like
/// similarities); ranking within one search is unaffected.
///
/// ## Three things the library leaves to us
///
/// `MLXEmbedders` gives raw projections and nothing else, by design - its own comment notes that the
/// encoder "does NOT mask or filter outputs" because a `0` in the attention mask means padding for a
/// document but a kept expansion token for a query, so only the retrieval layer can tell them apart.
/// So: we normalize per token, we inject the query/document prefixes the config declares, and we
/// implement MaxSim.
///
/// We also **encode one text at a time**. A batch would need padding, and padding is exactly what the
/// encoder refuses to mask - a padded document would contribute junk tokens that MaxSim would happily
/// match against. One text per forward pass makes the mask question disappear. The corpus is a few
/// dozen short strings and its vectors are cached after the first pass, so the cost is a one-time
/// warm-up rather than a per-search charge.
///
/// The cache is per process and in memory, so the *first* search of a session encodes the whole
/// corpus (a few dozen short strings) and every later one encodes just the query. Persisting the
/// document vectors, the way `PrefixKVStore` persists prompt prefixes, is the obvious follow-up.
///
/// An `actor` because it owns that cache and the loaded model container. Document vectors are held as
/// plain `[[Float]]` rather than `MLXArray`: MLX arrays are not `Sendable` and could not cross into
/// the `@Sendable` closure `EmbedderModelContainer.perform` requires, and MaxSim over a corpus this
/// size is a few million multiplications - immaterial next to a model forward pass.
public actor ColBERTToolRetriever: ToolRetriever {
    private let model: ToolSearchModel
    /// Loaded on first use, then kept - the model stays resident alongside the planner.
    private var container: EmbedderModelContainer?
    /// Per-token vectors per document `indexText`, keyed by the text itself so a corpus change
    /// re-encodes only what actually changed.
    private var documentVectors: [String: [[Float]]] = [:]
    /// Prefixes and length caps read from the model's own `config.json` (`mlx` block), so we follow
    /// the trained convention instead of guessing at it.
    private var queryPrefix = ""
    private var documentPrefix = ""
    private var queryLength = 32
    private var documentLength = 180

    public init(model: ToolSearchModel = .default) {
        self.model = model
    }

    public func search(_ query: String, in corpus: [ToolDocument], limit: Int) async throws -> [ToolMatch] {
        guard limit > 0, !corpus.isEmpty else { return [] }
        let documents = try await vectors(for: corpus)
        let queryVectors = try await encode(query, isQuery: true)
        guard !queryVectors.isEmpty else { return [] }

        var scored: [ToolMatch] = []
        for document in corpus {
            guard let vectors = documents[document.indexText], !vectors.isEmpty else { continue }
            scored.append(
                ToolMatch(name: document.name, score: Self.maxSim(queryVectors, vectors))
            )
        }
        // Name as the tiebreak so equal scores rank the same way on every run.
        return scored
            .sorted { $0.score == $1.score ? $0.name < $1.name : $0.score > $1.score }
            .prefix(limit)
            .map { $0 }
    }

    /// Cached per-token vectors for every document, encoding only the ones not already held.
    private func vectors(for corpus: [ToolDocument]) async throws -> [String: [[Float]]] {
        for document in corpus where documentVectors[document.indexText] == nil {
            documentVectors[document.indexText] = try await encode(document.indexText, isQuery: false)
        }
        // Drop anything no longer in the corpus, so a policy change doesn't leak vectors forever.
        let live = Set(corpus.map(\.indexText))
        documentVectors = documentVectors.filter { live.contains($0.key) }
        return documentVectors
    }

    /// One forward pass: prefix, tokenize, truncate, project, L2-normalize per token.
    private func encode(_ text: String, isQuery: Bool) async throws -> [[Float]] {
        let container = try await loadedContainer()
        let prefixed = (isQuery ? queryPrefix : documentPrefix) + text
        let cap = isQuery ? queryLength : documentLength
        return await container.perform { context in
            var tokens = context.tokenizer.encode(text: prefixed)
            if tokens.count > cap { tokens = Array(tokens.prefix(cap)) }
            guard !tokens.isEmpty else { return [] }
            let input = MLXArray(tokens.map { Int32($0) }).reshaped(1, -1)
            let output = context.model.callAsFunction(
                input, positionIds: nil, tokenTypeIds: nil, attentionMask: nil
            )
            guard let hidden = output.hiddenStates else { return [] }
            // Normalize along the projection axis by hand rather than through `Pooling`: with
            // strategy `.none` the output is 3-D `(1, L, 128)`, and `Pooling`'s optional `dimension`
            // truncation would slice the *token* axis of a 3-D array, not the vector axis.
            let scale = MLX.sqrt(MLX.sum(hidden * hidden, axis: -1, keepDims: true)) + 1e-9
            let normalized = hidden / scale
            MLX.eval(normalized)
            let dimension = normalized.dim(-1)
            let flat = normalized.asArray(Float.self)
            return stride(from: 0, to: flat.count, by: dimension).map {
                Array(flat[$0 ..< min($0 + dimension, flat.count)])
            }
        }
    }

    /// Load the encoder once, and read the ColBERT conventions off the model it produced.
    private func loadedContainer() async throws -> EmbedderModelContainer {
        if let container { return container }
        let loaded = try await EmbedderModelFactory.shared.loadContainer(
            from: #hubDownloader(), using: #huggingFaceTokenizerLoader(),
            configuration: ModelConfiguration(id: model.repoID)
        )
        // `queryPrefix` / `documentPrefix` / the length caps live in the custom `mlx` block of the
        // model's config.json. The library parses them but deliberately does not apply them.
        if let head = await loaded.perform({ context in
            (context.model as? LFM2BidirectionalModel)?.configuration.mlx
        }) {
            queryPrefix = head.queryPrefix ?? ""
            documentPrefix = head.documentPrefix ?? ""
            queryLength = head.queryLength ?? queryLength
            documentLength = head.documentLength ?? documentLength
        }
        container = loaded
        return loaded
    }

    /// MaxSim over already-normalized vectors: each query token takes its best document token.
    static func maxSim(_ query: [[Float]], _ document: [[Float]]) -> Double {
        var total = 0.0
        for queryVector in query {
            var best = -Float.greatestFiniteMagnitude
            for documentVector in document {
                var dot: Float = 0
                for index in 0 ..< min(queryVector.count, documentVector.count) {
                    dot += queryVector[index] * documentVector[index]
                }
                if dot > best { best = dot }
            }
            if best > -Float.greatestFiniteMagnitude { total += Double(best) }
        }
        return query.isEmpty ? 0 : total / Double(query.count)
    }
}
