@testable import DeepAgents
@testable import DeepAgentsMLX
import Foundation
import Testing

/// The one test that actually *runs* the ColBERT retriever end to end: load the encoder, inject the
/// query/document prefixes off the model's own config, tokenize, forward, L2-normalize per token, and
/// rank by MaxSim.
///
/// Everything else about the retriever is unit-tested on hand-built vectors
/// (`ColBERTMaxSimTests`), which pins the scoring but never executes the encode path - so a wrong
/// normalization axis, a mis-shaped `[1, L, 128]` slice, or prefixes read from the wrong place would
/// all pass the unit suite and fail here.
///
/// ## Why this is opt-in
///
/// It needs `DEEPAGENTS_MLX_TESTS=1`, and not because of the weights. **No test in this package can
/// run real MLX inference under `swift test`:** MLX's Metal shader library (`default.metallib`) is
/// only emitted by Xcode's build system, so the first GPU op aborts the whole process with "Failed to
/// load the default metallib" - it takes the test run down, it does not fail one test. (Copying the
/// bundle Xcode emitted next to the `.xctest` executable does not satisfy its search paths either.)
/// That is why `ripple` is built with xcodebuild and why every other test here drives `FakeChatModel`.
///
/// So this is left switched off by default rather than deleted: the assertions are the ones worth
/// making about the encode path, and they run the moment there is a metallib-capable runner (an Xcode
/// test target in one of the workspaces, or a wrapper that co-locates the bundle the way `just build`
/// does for the CLI). Until then the encode path is exercised by using the feature, not by CI - which
/// is worth knowing rather than assuming.
///
/// Also requires the 8-bit weights on disk, so enabling the flag on a fresh machine skips instead of
/// triggering a silent multi-hundred-MB download inside a test.
@Suite(.serialized)
struct ColBERTRetrievalTests {
    private static let model = ToolSearchModel.colbert350m8bit

    private static var canRun: Bool {
        ProcessInfo.processInfo.environment["DEEPAGENTS_MLX_TESTS"] == "1"
            && MlxModelLoader.isDownloadedOnDisk(model.repoID)
    }

    /// A small corpus written the way ``ToolDocument/corpus(for:)`` writes one: both the human-facing
    /// summary and the model-facing description, so retrieval has both voices to match against.
    private var corpus: [ToolDocument] {
        [
            document(
                "read_clipboard", toolset: "Clipboard",
                text: "read clipboard. Read clipboard. Read the current pasteboard contents."
            ),
            document(
                "git_log", toolset: "Git",
                text: "git log. Git log. Show recent commits in the repository's history."
            ),
            document(
                "take_screenshot", toolset: "Screen Capture",
                text: "take screenshot. Take screenshot. Capture the full screen as an image."
            ),
            document(
                "create_note", toolset: "Apple Notes",
                text: "create note. Create note. Create a new note in Apple Notes with a title and body."
            )
        ]
    }

    private func document(_ name: String, toolset: String, text: String) -> ToolDocument {
        ToolDocument(
            name: name, toolset: toolset.lowercased(), toolsetDisplayName: toolset,
            signature: "\(name)()", summary: text, indexText: text
        )
    }

    /// (query, the tool it should surface) pairs. Grow this fixture when tuning
    /// ``ToolDocument/corpus(for:)``'s index text - it is what makes that tuning safe.
    private static let fixture = [
        ("what is on my clipboard right now", "read_clipboard"),
        ("show me the recent commits", "git_log"),
        ("look at what is on the screen", "take_screenshot"),
        ("save this as a new note", "create_note")
    ]

    /// Recall@k, not "is it first". `search_tools` hands the model a *list* (5 by default) and the
    /// model picks from it, so the bar that matters is whether the right tool is in that list at all.
    ///
    /// Measured on this fixture: **recall@2 = 4/4, recall@1 = 3/4.** The top-1 miss is
    /// "look at what is on the screen", which ranks `read_clipboard` first - "what is on my …" pulls
    /// hard toward the clipboard document. It is a genuine ranking weakness, not a defect: the tool is
    /// still returned, and the model sees it. Worth revisiting if the index text changes (upstream
    /// ColBERT also drops punctuation tokens from documents via a skiplist, which we do not).
    @Test(.enabled(if: canRun))
    func recallAtTwoIsPerfectOnTheFixture() async throws {
        let retriever = ColBERTToolRetriever(model: Self.model)
        var topOne = 0
        for (query, expected) in Self.fixture {
            let matches = try await retriever.search(query, in: corpus, limit: 4)
            #expect(matches.count == 4)
            let names = matches.map(\.name)
            #expect(
                names.prefix(2).contains(expected),
                "\"\(query)\" wanted \(expected) in the top 2, ranked \(names)"
            )
            if names.first == expected { topOne += 1 }
            // Cosine over L2-normalized vectors, averaged over query tokens: every score has to land
            // in [-1, 1]. Outside that, the normalization did not happen on the projection axis.
            for match in matches {
                #expect(match.score >= -1.0001 && match.score <= 1.0001, "\(match.name) → \(match.score)")
            }
            #expect((matches.first?.score ?? 0) > 0.3) // a real match, not noise
        }
        // Pinned so a regression in ranking shows up as a failure rather than as a quiet drift.
        #expect(topOne >= 3, "recall@1 dropped to \(topOne)/\(Self.fixture.count)")
    }

    @Test(.enabled(if: canRun))
    func rankingIsStableAcrossCallsOnTheCachedCorpus() async throws {
        // The second search reuses the cached document vectors and re-encodes only the query; it must
        // produce the same ranking as the first, which encoded everything.
        let retriever = ColBERTToolRetriever(model: Self.model)
        let first = try await retriever.search("what is on my clipboard", in: corpus, limit: 4)
        let second = try await retriever.search("what is on my clipboard", in: corpus, limit: 4)

        #expect(first.map(\.name) == second.map(\.name))
        for (a, b) in zip(first, second) { #expect(abs(a.score - b.score) < 1e-5) }
    }

    @Test(.enabled(if: canRun))
    func beatsTheLexicalBaselineOnAParaphrase() async throws {
        // The reason to spend 350 MB on this at all: a query sharing no words with the tool's text.
        // "copied" / "pasteboard" have no lexical overlap, so term matching cannot get there.
        let query = "what did I copy earlier"
        let lexical = try await LexicalToolRetriever().search(query, in: corpus, limit: 4)
        let colbert = try await ColBERTToolRetriever(model: Self.model).search(query, in: corpus, limit: 4)

        #expect(colbert.first?.name == "read_clipboard")
        // Recorded rather than asserted: if the lexical retriever also gets this one, the paraphrase
        // was not as hard as intended and the test should be sharpened, not deleted.
        if lexical.first?.name == "read_clipboard" {
            Issue.record("the lexical baseline also ranked read_clipboard first - pick a harder paraphrase")
        }
    }
}
