import Foundation

/// Splits a streaming generation into visible answer text and `<think>…</think>` reasoning, holding
/// back partial tags across chunk boundaries so a tag split over two chunks never leaks into the
/// answer. A trailing unterminated `<think>` (still streaming) is treated as in-progress reasoning.
///
/// The `<think>`/`</think>` convention is shared across most on-device reasoning models this adapter
/// drives (LFM2.5 Thinking, Ornith/qwen3_5, …), so this lives outside any one codec: ``LFM2Decoder``,
/// ``Qwen35Decoder``, and ``Gemma4Decoder`` route their visible text through it to surface reasoning
/// on its own channel. The tags are configurable for models with a different marker pair - Gemma 4
/// wraps reasoning in a `<|channel>thought\n…<channel|>` block instead.
///
/// `startInThink` controls the initial state. LFM2 generates the whole `<think>…</think>` itself, so
/// it starts in the answer (the default). Ornith's chat template prefills the opening `<think>` into
/// the generation prompt, so the model's stream begins *inside* reasoning and only emits the closing
/// `</think>`; its decoder starts in-think so that pre-`</think>` content is routed to reasoning, not
/// leaked into the answer.
struct ThinkStream {
    static let startTag = "<think>"
    static let endTag = "</think>"

    private let startTag: String
    private let endTag: String
    private var buffer = ""
    private var inThink: Bool

    init(startInThink: Bool = false, startTag: String = ThinkStream.startTag, endTag: String = ThinkStream.endTag) {
        inThink = startInThink
        self.startTag = startTag
        self.endTag = endTag
    }

    mutating func consume(_ chunk: String) -> (answer: String, reasoning: String) {
        buffer += chunk
        var answer = ""
        var reasoning = ""
        while true {
            if !inThink {
                if let range = buffer.range(of: startTag) {
                    answer += buffer[..<range.lowerBound]
                    buffer = String(buffer[range.upperBound...])
                    inThink = true
                } else {
                    let cut = Self.safeEmitEnd(of: buffer, tag: startTag)
                    answer += buffer[..<cut]
                    buffer = String(buffer[cut...])
                    break
                }
            } else {
                if let range = buffer.range(of: endTag) {
                    reasoning += buffer[..<range.lowerBound]
                    buffer = String(buffer[range.upperBound...])
                    inThink = false
                } else {
                    let cut = Self.safeEmitEnd(of: buffer, tag: endTag)
                    reasoning += buffer[..<cut]
                    buffer = String(buffer[cut...])
                    break
                }
            }
        }
        return (answer, reasoning)
    }

    mutating func finish() -> (answer: String, reasoning: String) {
        defer { buffer = ""; inThink = false }
        return inThink ? ("", buffer) : (buffer, "")
    }

    /// The index up to which `s` can be emitted without splitting a possible `tag`: holds back the
    /// longest suffix of `s` that is a proper prefix of `tag`.
    private static func safeEmitEnd(of s: String, tag: String) -> String.Index {
        let maxHold = min(s.count, tag.count - 1)
        if maxHold > 0 {
            for hold in stride(from: maxHold, through: 1, by: -1) where tag.hasPrefix(s.suffix(hold)) {
                return s.index(s.endIndex, offsetBy: -hold)
            }
        }
        return s.endIndex
    }
}
