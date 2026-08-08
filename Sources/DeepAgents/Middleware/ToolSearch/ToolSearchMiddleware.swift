import Foundation

/// Keeps auxiliary tools out of the *rendered* prompt while leaving them executable, and gives the
/// agent `search_tools` to find them.
///
/// ## Why this shape
///
/// The rendered tool set is part of the cached prompt prefix: `MlxChatModel` fingerprints
/// `systemPrompt + toolNames` and resets the whole ``PrefixCacheSlot`` when it changes, and the
/// schemas render at the *front* of the prompt, so any change to the tool list invalidates the base
/// and tip snapshots. Growing the tool list mid-run - the obvious way to implement lazy tools - is
/// therefore the one thing that cannot be done cheaply.
///
/// So nothing here ever changes the tool list. The middleware strips auxiliary tools from
/// `ModelRequest.tools` on every round (a constant filter, not a growing one), and the schemas the
/// model needs arrive as the *tool result* of `search_tools` - conversation content, which appends
/// past the cached tip and costs only its own tokens. `ReactAgent` dispatches against its own full
/// tool list rather than the request's, so an auxiliary tool stays callable the whole time:
/// **invisible, but executable**.
///
/// ## Two call shapes, both always present
///
/// `search_tools` returns signatures and the model calls the tool by name (the preferred path). A
/// planner that will not emit a name it cannot see can instead call `run_tool(name:, arguments:)`;
/// ``wrapModelCall`` rewrites that into a direct call before `ReactAgent` sees it, so approvals,
/// human-in-the-loop, and the message log all observe the real tool. Both meta-tools are rendered
/// permanently, which is what makes the fallback free: no switch to flip, and no fingerprint change
/// at the moment the model is already struggling.
public struct ToolSearchMiddleware: AgentMiddleware, ToolRenderFiltering {
    /// Dispatch names of the tools kept out of the prompt.
    public let auxiliaryToolNames: Set<String>
    /// The retrieval corpus, one document per auxiliary tool.
    let documents: [ToolDocument]
    let retriever: any ToolRetriever
    /// Default number of matches a search returns; `search_tools`' `limit` argument overrides it.
    let limit: Int

    /// - Parameters:
    ///   - auxiliaryTools: the tools to hide. Their schemas are never rendered, but they remain in
    ///     the agent's tool list and stay dispatchable.
    ///   - retriever: ranks the corpus against a query. ``LexicalToolRetriever`` needs no model.
    ///   - toolsetsByTool: toolset id per tool for anything outside ``MiddlewareCatalog`` (MCP tools,
    ///     keyed to their server), used for the corpus and the "searched N toolsets" footer.
    ///   - limit: default match count.
    public init(
        auxiliaryTools: [any AgentTool],
        retriever: any ToolRetriever = LexicalToolRetriever(),
        toolsetsByTool: [String: String] = [:],
        limit: Int = 5
    ) {
        auxiliaryToolNames = Set(auxiliaryTools.map(\.name))
        documents = ToolDocument.corpus(for: auxiliaryTools, toolsetsByTool: toolsetsByTool)
        self.retriever = retriever
        self.limit = limit
    }

    public var name: String { "tool_search" }

    public var hiddenToolNames: Set<String> { auxiliaryToolNames }

    public var tools: [any AgentTool] {
        guard !documents.isEmpty else { return [] }
        return [
            SearchToolsTool(documents: documents, retriever: retriever, limit: limit),
            RunTool(auxiliaryToolNames: auxiliaryToolNames)
        ]
    }

    /// Strip the auxiliary tools from what the model sees, add the standing pointer to
    /// `search_tools`, and rewrite any `run_tool` call in the response into a direct call.
    ///
    /// The rewrite happens here rather than in `afterModel` because `ReactAgent` reads the round's
    /// tool calls from the value `wrapModelCall` returns - not from `state.messages` - so this is the
    /// last point at which a rewrite still governs dispatch.
    public func wrapModelCall(
        _ request: ModelRequest,
        _ handler: (ModelRequest) async throws -> ModelResponse
    ) async throws -> ModelResponse {
        guard !documents.isEmpty else { return try await handler(request) }
        let rendered = request.tools.filter { !auxiliaryToolNames.contains($0.name) }
        let composed = [request.systemPrompt, systemPrompt]
            .compactMap { $0 }
            .joined(separator: "\n\n")
        var response = try await handler(request.override(systemPrompt: composed, tools: rendered))
        response.message = Self.rewritingRunToolCalls(
            response.message, auxiliaryToolNames: auxiliaryToolNames
        )
        return response
    }

    /// The standing prompt section: the auxiliary tools' **names**, grouped by toolset, and nothing
    /// else. No schemas, no per-tool prose, and constant for the life of the run - so it sits inside
    /// the cached prefix and never moves the fingerprint.
    ///
    /// Names, not just toolset labels, because the index is what makes routing work. Asked "which
    /// Apple Notes tools do you have?", an agent given only "Apple Notes" as a topic has to guess or
    /// search blindly; given `list_notes, read_note, create_note, update_note` it can answer, and can
    /// call one straight away (dispatch resolves auxiliary names, so a direct call succeeds - it only
    /// needs `search_tools` for the argument shapes). The whole index costs ~150 tokens against the
    /// ~1.5k of per-tool prose it replaces.
    var systemPrompt: String {
        let byToolset = Dictionary(grouping: documents, by: \.toolsetDisplayName)
        let index = byToolset.keys.sorted().map { toolset in
            "  \(toolset): " + byToolset[toolset, default: []].map(\.name).sorted().joined(separator: ", ")
        }
        .joined(separator: "\n")
        return """
        ## Finding more tools

        These tools exist but their schemas are not loaded, so you cannot see their parameters:

        \(index)

        To use one, call `search_tools` with a short description of what you need ("read a note", \
        "check git history"). It returns the exact signatures, and you then call the tool by name \
        like any other. Do not invent a tool that is not listed above or in your own schema. If your \
        format will not let you call a tool that is absent from your schema, call `run_tool` with the \
        name and arguments instead.
        """
    }

    /// Rewrite `run_tool(name:, arguments:)` into a direct call to `name`, preserving the call id so
    /// the eventual tool result still answers the right call. A call naming an unknown tool is left
    /// alone, so `run_tool`'s own `execute` produces the correction instead.
    static func rewritingRunToolCalls(
        _ message: AgentMessage, auxiliaryToolNames: Set<String>
    ) -> AgentMessage {
        guard message.toolCalls.contains(where: { $0.name == RunTool.toolName }) else { return message }
        var rewritten = message
        rewritten.toolCalls = message.toolCalls.map { call in
            guard call.name == RunTool.toolName,
                  let target = ToolArgs.rawString(call.arguments, "name"),
                  auxiliaryToolNames.contains(target)
            else { return call }
            return AgentToolCall(
                id: call.id, name: target, arguments: Self.innerArguments(call.arguments)
            )
        }
        return rewritten
    }

    /// The `arguments` payload of a `run_tool` call, as the target tool's own argument map. Accepts
    /// both the object form and the stringified-JSON form small models tend to emit; an absent or
    /// unparseable payload becomes an empty map, which the target tool then reports as a missing
    /// required parameter.
    static func innerArguments(_ arguments: [String: AgentJSON]) -> [String: AgentJSON] {
        switch arguments["arguments"] {
        case .object(let inner):
            return inner
        case .string(let raw):
            guard let data = raw.data(using: .utf8),
                  let parsed = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
            else { return [:] }
            return parsed.mapValues { AgentJSON.from($0) }
        default:
            return [:]
        }
    }
}

/// `search_tools`: rank the auxiliary tools against a description of what the agent needs and return
/// their signatures.
struct SearchToolsTool: AgentTool {
    let documents: [ToolDocument]
    let retriever: any ToolRetriever
    let limit: Int

    var name: String { "search_tools" }
    var description: String {
        "Find tools that are available but not listed in your prompt. Describe what you need to do "
            + "(e.g. \"read a file\", \"check git history\") and this returns the matching tools' "
            + "names and signatures, which you can then call directly."
    }

    /// Pure - it reads a fixed corpus and writes nothing - so it may run alongside the other
    /// parallel-safe calls in a round.
    var isParallelSafe: Bool { true }

    var parameters: [ToolParameter] {
        [
            .required("query", type: .string, description: "What you need to do, in a few words."),
            .optional("limit", type: .int, description: "How many tools to return. Defaults to \(limit).")
        ]
    }

    /// Keep the rendered result clear of ``ReactAgent/maxToolResultCharacters``, which would
    /// otherwise truncate it mid-signature and leave the model with half a tool name.
    static let resultBudget = 4000

    func execute(_ arguments: [String: AgentJSON], _ context: ToolContext) async throws -> ToolOutput {
        guard let query = ToolArgs.rawString(arguments, "query"), !query.isEmpty else {
            return ToolOutput("Error: `query` is required - describe what you need to do.")
        }
        let wanted = min(max(ToolArgs.int(arguments, "limit") ?? limit, 1), documents.count)
        let matches = try await retriever.search(query, in: documents, limit: wanted)
        let byName = Dictionary(uniqueKeysWithValues: documents.map { ($0.name, $0) })
        let toolsets = Set(documents.map(\.toolsetDisplayName)).sorted().joined(separator: ", ")

        // Miss policy: never answer a search with nothing when tools exist. A weak match the model
        // can reject beats a dead end - and the footer tells it how to widen the search.
        guard !matches.isEmpty else {
            return ToolOutput(
                "No tool matched \"\(query)\". Available toolsets: \(toolsets). "
                    + "Try search_tools again with different words, or answer without a tool."
            )
        }

        // The legend is not decoration: a bare parameter name meaning "required" is an inference, and
        // the whole point of the result is that the model can call the tool correctly first time.
        var lines = ["Found \(matches.count) tool(s) for \"\(query)\". Call one by name, "
            + "or use run_tool if you cannot call a tool that is not in your own schema. "
            + "In the signatures below `arg!` must be passed and `arg?` is optional.", ""]
        var used = lines.joined(separator: "\n").count
        var omitted = 0
        for match in matches {
            guard let document = byName[match.name] else { continue }
            var entry = "  \(document.signature)\n      \(document.summary)"
            // Spell out the mandatory arguments. Optional ones are left to name and type: these are
            // the ones a wrong guess fails on, and the budget is better spent here than everywhere.
            for parameter in document.parameters where parameter.isRequired && !parameter.description.isEmpty {
                entry += "\n      \(parameter.label) - \(parameter.description)"
            }
            if used + entry.count > Self.resultBudget {
                omitted += 1
                continue
            }
            used += entry.count + 1
            lines.append(entry)
        }
        lines.append("")
        if omitted > 0 { lines.append("(\(omitted) further match(es) omitted to stay within the result size.)") }
        lines.append(
            "Searched toolsets: \(toolsets). If none of these fit, call search_tools again with "
                + "different words."
        )
        return ToolOutput(lines.joined(separator: "\n"))
    }
}

/// `run_tool`: the escape hatch for a planner that will not emit a tool name absent from its own
/// schema.
///
/// In the normal case this tool never executes - ``ToolSearchMiddleware/wrapModelCall`` rewrites the
/// call into a direct call to the target, so the target's own approval gating and validation apply.
/// `execute` is therefore only reached when the rewrite could not happen, and its whole job is to say
/// why. (Executing the target here instead would run it inside *this* tool's `wrapToolCall` chain and
/// so bypass the approval gate for the tool that actually runs.)
struct RunTool: AgentTool {
    static let toolName = "run_tool"

    let auxiliaryToolNames: Set<String>

    var name: String { Self.toolName }
    var description: String {
        "Call a tool that search_tools returned but that is not listed in your own schema. Pass the "
            + "tool's exact name and its arguments as an object."
    }

    var parameters: [ToolParameter] {
        [
            .required("name", type: .string, description: "Exact tool name, as returned by search_tools."),
            .required(
                "arguments", type: .object(properties: []),
                description: "The tool's arguments, as an object."
            )
        ]
    }

    func execute(_ arguments: [String: AgentJSON], _ context: ToolContext) async throws -> ToolOutput {
        guard let target = ToolArgs.rawString(arguments, "name"), !target.isEmpty else {
            return ToolOutput(
                "Error: `name` is required - the exact name of the tool to run, as search_tools "
                    + "returned it."
            )
        }
        // The middleware rewrites every call naming a known auxiliary tool, so reaching here means
        // the name is not one of them.
        let known = auxiliaryToolNames.sorted().joined(separator: ", ")
        return ToolOutput(
            "Error: '\(target)' is not a searchable tool. Call search_tools to find the right name, "
                + "or call the tool directly if it is already in your schema. "
                + "Searchable tools: \(known)."
        )
    }
}
