import Foundation

/// IDF-weighted token overlap between the query and each tool's index text - the retriever that
/// needs no model.
///
/// It is the default for three situations, not just a placeholder: the feature is on but the ColBERT
/// weights have not downloaded yet, the host has no MLX (the pure framework, and the unit tests),
/// and the user picked "lexical" to spend nothing on retrieval. Term rarity carries most of the
/// signal here - "clipboard" appears in one tool's text and "file" in a dozen, so weighting by
/// inverse document frequency is what separates them.
public struct LexicalToolRetriever: ToolRetriever {
    public init() {}

    /// Words too common in tool descriptions to discriminate. Deliberately short: an aggressive list
    /// would drop terms that really do select a tool ("open", "read", "write").
    static let stopWords: Set<String> = [
        "a", "an", "and", "are", "as", "at", "be", "by", "can", "for", "from", "in", "into", "is",
        "it", "its", "of", "on", "or", "that", "the", "then", "this", "to", "up", "use", "used",
        "using", "with", "you", "your", "so", "if", "when", "what", "how", "do", "does", "i",
        "me", "my", "tool", "tools"
    ]

    public func search(_ query: String, in corpus: [ToolDocument], limit: Int) async throws -> [ToolMatch] {
        guard limit > 0, !corpus.isEmpty else { return [] }
        let queryTerms = Set(Self.tokenize(query))
        let documents = corpus.map { (document: $0, terms: Self.tokenize($0.indexText)) }

        // Document frequency over the corpus, so a term's weight reflects how much it narrows down.
        var frequency: [String: Int] = [:]
        for entry in documents {
            for term in Set(entry.terms) { frequency[term, default: 0] += 1 }
        }
        let total = Double(documents.count)

        var scored: [ToolMatch] = []
        for entry in documents {
            var counts: [String: Int] = [:]
            for term in entry.terms { counts[term, default: 0] += 1 }
            var score = 0.0
            for term in queryTerms {
                guard let count = counts[term], let document = frequency[term] else { continue }
                // Smoothed IDF, damped TF: a tool that says "file" six times is not six times as
                // relevant as one that says it once.
                let idf = log(1 + total / Double(document))
                score += idf * (1 + log(Double(count)))
            }
            // Longer texts accumulate matches by sheer length; divide it back out.
            if !entry.terms.isEmpty { score /= sqrt(Double(entry.terms.count)) }
            // A query naming the tool outright should win regardless of the statistics.
            if queryTerms.contains(where: { entry.document.name.contains($0) }) { score += 1 }
            if score > 0 { scored.append(ToolMatch(name: entry.document.name, score: score)) }
        }
        // Sort by score, then by name so equal scores rank deterministically across runs.
        return scored
            .sorted { $0.score == $1.score ? $0.name < $1.name : $0.score > $1.score }
            .prefix(limit)
            .map { $0 }
    }

    /// Lowercase alphanumeric runs, minus stop words and single characters. Underscores split, so
    /// `write_clipboard` indexes as `write` + `clipboard`.
    static func tokenize(_ text: String) -> [String] {
        text.lowercased()
            .split { !$0.isLetter && !$0.isNumber }
            .map(String.init)
            .filter { $0.count > 1 && !stopWords.contains($0) }
    }
}
