import Foundation

/// One auxiliary tool as a retrieval document: the text a retriever scores a query against, plus
/// the signature ``ToolSearchMiddleware`` renders into the search result.
///
/// The index text is deliberately assembled from *both* voices a tool has - the human-facing
/// `displayName`/`summary` the settings UI shows (``ToolDescriptor``) and the model-facing
/// `description` the schema carries - because a user's query ("look at my screen") tends to match
/// the former while the model's query ("capture a screenshot") tends to match the latter.
public struct ToolDocument: Sendable, Equatable {
    /// The tool's dispatch name - what the model must emit to call it.
    public let name: String
    /// The owning toolset: an ``AgentMiddleware/name`` for a built-in, or the server name for MCP.
    public let toolset: String
    /// Human label for the toolset, for the "searched N toolsets" footer.
    public let toolsetDisplayName: String
    /// The rendered signature, e.g. `grep(pattern: string, path?: string)`.
    public let signature: String
    /// The one-line description shown under the signature.
    public let summary: String
    /// What the retriever scores against.
    public let indexText: String

    public init(
        name: String, toolset: String, toolsetDisplayName: String,
        signature: String, summary: String, indexText: String
    ) {
        self.name = name
        self.toolset = toolset
        self.toolsetDisplayName = toolsetDisplayName
        self.signature = signature
        self.summary = summary
        self.indexText = indexText
    }
}

extension ToolDocument {
    /// Build the retrieval corpus for `tools`, enriching each with its catalog entry when it has one.
    ///
    /// MCP tools are not in the catalog, so they fall back to the tool's own name and description and
    /// take their toolset from `toolsetsByTool` (which ripple/the app populate from the server that
    /// contributed each tool).
    public static func corpus(
        for tools: [any AgentTool],
        catalog: [MiddlewareDescriptor] = MiddlewareCatalog.all,
        toolsetsByTool: [String: String] = [:]
    ) -> [ToolDocument] {
        var byTool: [String: (toolset: MiddlewareDescriptor, tool: ToolDescriptor)] = [:]
        for toolset in catalog {
            for descriptor in toolset.tools { byTool[descriptor.name] = (toolset, descriptor) }
        }
        return tools.map { tool in
            let entry = byTool[tool.name]
            let toolset = entry?.toolset.id ?? toolsetsByTool[tool.name] ?? "other"
            let toolsetDisplay = entry?.toolset.displayName ?? toolsetsByTool[tool.name] ?? "Other"
            // Both voices plus the toolset name, so a query can land through any of them.
            let index = [
                tool.name.replacingOccurrences(of: "_", with: " "),
                entry?.tool.displayName,
                entry?.tool.summary,
                tool.description,
                toolsetDisplay,
                tool.parameters.map(\.name).joined(separator: " ")
            ]
            .compactMap { $0 }
            .filter { !$0.isEmpty }
            .joined(separator: ". ")
            return ToolDocument(
                name: tool.name,
                toolset: toolset,
                toolsetDisplayName: toolsetDisplay,
                signature: signature(of: tool),
                summary: entry?.tool.summary ?? tool.description,
                indexText: index
            )
        }
    }

    /// `name(required: type, optional?: type)` - compact enough that a handful of matches fit inside
    /// the tool-result budget, where raw JSON Schema would not (see
    /// ``ReactAgent/maxToolResultCharacters``).
    static func signature(of tool: any AgentTool) -> String {
        let parameters = tool.parameters.map { parameter in
            "\(parameter.name)\(parameter.isRequired ? "" : "?"): \(typeName(parameter.type))"
        }
        return "\(tool.name)(\(parameters.joined(separator: ", ")))"
    }

    private static func typeName(_ type: ToolParameterType) -> String {
        switch type {
        case .string: "string"
        case .bool: "bool"
        case .int: "int"
        case .double: "number"
        case .data: "base64"
        case .array(let element): "[\(typeName(element))]"
        case .object: "object"
        }
    }
}

/// One retrieval hit: the tool's dispatch name and the retriever's score (higher is better; the
/// scale is retriever-specific and only meaningful for ranking within one search).
public struct ToolMatch: Sendable, Equatable {
    public let name: String
    public let score: Double

    public init(name: String, score: Double) {
        self.name = name
        self.score = score
    }
}

/// Ranks auxiliary tools against a natural-language query.
///
/// Two implementations ship: ``LexicalToolRetriever`` here in the pure framework, and
/// `ColBERTToolRetriever` in `DeepAgentsMLX` (late-interaction MaxSim over LFM2.5-ColBERT-350M).
/// The protocol lives here so `DeepAgents` stays free of MLX.
public protocol ToolRetriever: Sendable {
    /// The best `limit` matches for `query`, best first. Returning fewer is fine; returning nothing
    /// when `corpus` is non-empty is discouraged - a weak match the model can reject beats a dead
    /// end (see the miss policy in ``ToolSearchMiddleware``).
    func search(_ query: String, in corpus: [ToolDocument], limit: Int) async throws -> [ToolMatch]
}

/// Middleware that hides some of the agent's tools from the *rendered* prompt while leaving them
/// executable. `ReactAgent` consults this so its prompt-overhead estimate (which drives the
/// summarization trigger and the context meter) measures the tools the model actually sees, rather
/// than every tool that could be dispatched.
public protocol ToolRenderFiltering {
    /// Dispatch names this middleware strips from `ModelRequest.tools` before the model call.
    var hiddenToolNames: Set<String> { get }
}
