@testable import DeepAgentsMLX
import Foundation
import Testing

// MaxSim is the whole of ColBERT's scoring, and it is the kind of function where picking the wrong
// axis - mean instead of max, or summing without normalising by query length - still returns plausible
// numbers and silently reorders every result. So it is pinned here on hand-built unit vectors, where
// the right answer is known by inspection. (The encode path needs the 350M model on disk and belongs
// in the integration suite.)
@Suite("ColBERT MaxSim")
struct ColBERTMaxSimTests {
    /// Unit vectors in the first two dimensions, so dot products are exactly 1, 0, or -1.
    private let x: [Float] = [1, 0]
    private let y: [Float] = [0, 1]
    private let minusX: [Float] = [-1, 0]

    private func score(_ query: [[Float]], _ document: [[Float]]) -> Double {
        ColBERTToolRetriever.maxSim(query, document)
    }

    @Test("An exact match scores 1")
    func exactMatch() {
        #expect(abs(score([x], [x]) - 1.0) < 1e-6)
    }

    @Test("An orthogonal document scores 0")
    func orthogonal() {
        #expect(abs(score([x], [y])) < 1e-6)
    }

    @Test("An opposed document scores -1")
    func opposed() {
        #expect(abs(score([x], [minusX]) + 1.0) < 1e-6)
    }

    @Test("Each query token takes its BEST document token, not the first or the average")
    func takesTheMaximum() {
        // The distinguishing case: a document whose first token is irrelevant and whose second is a
        // perfect match must score 1, not 0 (first) and not 0.5 (mean over document tokens).
        #expect(abs(score([x], [y, x]) - 1.0) < 1e-6)
        #expect(abs(score([x], [x, y]) - 1.0) < 1e-6) // and order must not matter
    }

    @Test("Query tokens are averaged, so the scale stays comparable across queries")
    func averagesOverQueryTokens() {
        // Two perfectly matched query tokens score 1, not 2 - otherwise a longer query would outscore
        // a shorter one against the same document and the numbers would stop reading as similarities.
        #expect(abs(score([x, y], [x, y]) - 1.0) < 1e-6)
        // One matched, one unmatched: the unmatched token drags the average down rather than being
        // ignored, which is what makes a partially-relevant tool rank below a fully-relevant one.
        #expect(abs(score([x, y], [x]) - 0.5) < 1e-6)
    }

    @Test("A longer query is not rewarded for length")
    func lengthIsNotRewarded() {
        let short = score([x], [x, y])
        let padded = score([x, x, x], [x, y])
        #expect(abs(short - padded) < 1e-6)
    }

    @Test("Empty inputs score 0 instead of trapping")
    func degenerate() {
        // Reached whenever a text tokenises to nothing; a crash here would take down a tool call.
        #expect(score([], [x]) == 0)
        #expect(score([x], []) == 0)
        #expect(score([], []) == 0)
    }

    @Test("Ranking follows relevance")
    func ranksSensibly() {
        // The property that actually matters downstream: a document containing the query token
        // outranks one that does not.
        #expect(score([x], [x, y]) > score([x], [y, y]))
    }
}

@Suite("Tool search model catalog")
struct ToolSearchModelTests {
    @Test("Both retrievers are catalogued, so they list and download like any other model")
    func bothAreCatalogued() {
        for model in ToolSearchModel.allCases {
            let entry = model.catalogEntry
            #expect(entry != nil, "\(model.rawValue) is missing from MlxModel.catalog")
            #expect(entry?.kind == .retriever)
        }
    }

    @Test("A retriever can never be chosen as the planner")
    func neverAPlanner() {
        // `languageCatalog` used to filter `!isVision`, which would have admitted these - and an
        // encoder with no LM head fails at load, not at selection.
        let plannerIDs = Set(MlxModel.languageCatalog.map(\.id))
        for model in ToolSearchModel.allCases {
            #expect(!plannerIDs.contains(model.repoID))
        }
    }

    @Test("A retriever is not offered as a vision model either")
    func neverVision() {
        for model in ToolSearchModel.allCases {
            #expect(model.catalogEntry?.acceptsImages == false)
        }
    }

    @Test("The default is the 8-bit model")
    func defaultIsQuantized() {
        // Resident alongside the planner in unified memory, so the cheaper one is the default.
        #expect(ToolSearchModel.default == .colbert350m8bit)
    }
}
