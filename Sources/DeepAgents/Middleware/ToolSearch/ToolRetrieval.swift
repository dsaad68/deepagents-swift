import Foundation

/// One of a tool's parameters, as its **schema** describes it.
///
/// Read from ``AgentTool/toolSchema()`` rather than ``AgentTool/parameters``, because those two are
/// not equivalent: `parameters` is the convenience a built-in tool declares, while `toolSchema()` is
/// the protocol requirement every tool honours - and ``MCPTool`` overrides it to inject the server's
/// JSON Schema verbatim without ever populating `parameters`. Building signatures off `parameters`
/// therefore rendered every MCP tool as taking no arguments at all.
public struct ToolParameterSpec: Sendable, Equatable {
    public let name: String
    /// A short type name for the signature (`string`, `int`, `[string]`, `object`).
    public let type: String
    public let isRequired: Bool
    /// A declared `enum` constraint, rendered inline - a value the model cannot guess.
    public let allowedValues: [String]
    public let description: String

    /// `name!` when required, `name?` when optional. Explicit on both sides: a bare name meaning
    /// "required" is an inference, and one the model does not reliably make.
    public var label: String { "\(name)\(isRequired ? "!" : "?")" }
}

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
    /// The rendered signature, e.g. `grep(pattern!: string, path?: string)`.
    public let signature: String
    /// The one-line description shown under the signature.
    public let summary: String
    /// What the retriever scores against.
    public let indexText: String
    /// The tool's parameters, so the search result can spell out what each required one is for.
    public let parameters: [ToolParameterSpec]

    public init(
        name: String, toolset: String, toolsetDisplayName: String,
        signature: String, summary: String, indexText: String,
        parameters: [ToolParameterSpec] = []
    ) {
        self.name = name
        self.toolset = toolset
        self.toolsetDisplayName = toolsetDisplayName
        self.signature = signature
        self.summary = summary
        self.indexText = indexText
        self.parameters = parameters
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
            let parameters = parameterSpecs(of: tool)
            // Both voices plus the toolset name, so a query can land through any of them. Parameter
            // names come from the schema, so an MCP tool's contribute too.
            let index = [
                tool.name.replacingOccurrences(of: "_", with: " "),
                entry?.tool.displayName,
                entry?.tool.summary,
                tool.description,
                toolsetDisplay,
                parameters.map(\.name).joined(separator: " ")
            ]
            .compactMap { $0 }
            .filter { !$0.isEmpty }
            .joined(separator: ". ")
            return ToolDocument(
                name: tool.name,
                toolset: toolset,
                toolsetDisplayName: toolsetDisplay,
                signature: signature(name: tool.name, parameters: parameters),
                summary: entry?.tool.summary ?? tool.description,
                indexText: index,
                parameters: parameters
            )
        }
    }

    /// `name(required!: type, optional?: type)` - compact enough that a handful of matches fit inside
    /// the tool-result budget, where raw JSON Schema would not (see
    /// ``ReactAgent/maxToolResultCharacters``).
    static func signature(of tool: any AgentTool) -> String {
        signature(name: tool.name, parameters: parameterSpecs(of: tool))
    }

    static func signature(name: String, parameters: [ToolParameterSpec]) -> String {
        let rendered = parameters.map { parameter -> String in
            var text = "\(parameter.label): \(parameter.type)"
            if !parameter.allowedValues.isEmpty {
                text += " (" + parameter.allowedValues.map { "\"\($0)\"" }.joined(separator: "|") + ")"
            }
            return text
        }
        return "\(name)(\(rendered.joined(separator: ", ")))"
    }

    /// A tool's parameters, read out of its rendered JSON Schema.
    ///
    /// Required parameters come first, in the order the schema's `required` array lists them, then the
    /// optional ones alphabetically. Ordering matters twice over: JSON object properties arrive as an
    /// unordered dictionary, so without a rule the signature would differ run to run (and the document
    /// text with it, invalidating the retriever's cache), and putting the mandatory arguments first is
    /// how a signature reads.
    public static func parameterSpecs(of tool: any AgentTool) -> [ToolParameterSpec] {
        guard let function = tool.toolSchema()["function"] as? [String: any Sendable],
              let parameters = function["parameters"] as? [String: any Sendable]
        else { return [] }
        let properties = parameters["properties"] as? [String: any Sendable] ?? [:]
        let required = parameters["required"] as? [String] ?? []
        let requiredSet = Set(required)

        let optional = properties.keys.filter { !requiredSet.contains($0) }.sorted()
        // A name in `required` with no matching property is skipped rather than invented.
        let ordered = required.filter { properties[$0] != nil } + optional
        return ordered.map { name in
            let property = properties[name] as? [String: any Sendable] ?? [:]
            return ToolParameterSpec(
                name: name,
                type: typeName(property),
                isRequired: requiredSet.contains(name),
                allowedValues: allowedValues(property),
                description: property["description"] as? String ?? ""
            )
        }
    }

    /// The schema fragment's type, with an array's element type inlined (`[string]`).
    private static func typeName(_ property: [String: any Sendable]) -> String {
        guard let type = property["type"] as? String else { return "any" }
        switch type {
        case "integer": return "int"
        case "number": return "number"
        case "boolean": return "bool"
        case "array":
            let items = property["items"] as? [String: any Sendable] ?? [:]
            return "[\(typeName(items))]"
        default: return type
        }
    }

    /// A declared `enum`, as strings. Non-string members are described rather than dropped, so a
    /// numeric or mixed enum still reaches the model.
    private static func allowedValues(_ property: [String: any Sendable]) -> [String] {
        if let values = property["enum"] as? [String] { return values }
        guard let values = property["enum"] as? [any Sendable] else { return [] }
        return values.map { String(describing: $0) }
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
