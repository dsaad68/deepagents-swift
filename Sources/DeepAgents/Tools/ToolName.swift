/// The single spelling rule for a tool name, applied in *both* directions: a name is published to
/// the model through it, and every name the model sends back is put through it before being
/// matched. One function, both sides.
///
/// It used to be enforced on egress only - `MCPTool` sanitized the name it exposed, and the name
/// that came back was compared raw. That is not a convention, it is a hope: a `parallel-search`
/// server published `parallel_search__web_search`, the model emitted `parallel-search__web_search`
/// from the description it had read, and dispatch called it an unknown tool. Every consumer
/// downstream - dispatch, the approval gate, the duplicate-round signature, the message log - then
/// had to reconcile two spellings of one tool on its own, and each place that invented its own
/// leniency was a place the others disagreed with.
///
/// The rule: ASCII letters, digits and `_` survive (lowercased); everything else becomes `_`.
/// Models are trained on `[A-Za-z0-9_]` function names and normalize other punctuation away when
/// emitting a call, so a name outside that set cannot round-trip and must not be published.
public enum ToolName {
    /// `name` reduced to the canonical spelling. Never returns an empty string.
    public static func normalized(_ name: String) -> String {
        let mapped = name.map { character -> Character in
            guard character.isASCII, character.isLetter || character.isNumber || character == "_" else {
                return "_"
            }
            return Character(character.lowercased())
        }
        return mapped.isEmpty ? "_" : String(mapped)
    }

    /// The tool in `tools` that `name` refers to: an exact hit, else the one whose normalized name
    /// matches. Ambiguity is never guessed at - if several tools share a normalized name and none
    /// matches exactly, the call names none of them.
    static func resolve(_ name: String, in tools: [any AgentTool]) -> (any AgentTool)? {
        if let exact = tools.first(where: { $0.name == name }) { return exact }
        let wanted = normalized(name)
        let matches = tools.filter { normalized($0.name) == wanted }
        return matches.count == 1 ? matches[0] : nil
    }
}
