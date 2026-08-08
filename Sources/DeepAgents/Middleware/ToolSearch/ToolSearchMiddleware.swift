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

    /// The standing prompt section: the two tiers, and - the part that actually changes behaviour -
    /// when the model is obliged to search.
    ///
    /// It names **no capabilities at all**: not tool names, not even toolset labels. Auxiliary tools are
    /// discovered only through `search_tools`, which is the whole premise of the tier. Two earlier
    /// versions listed more and both caused harm:
    ///
    /// - **Tool names** (`Apple Notes: create_note, list_notes, …`) made the auxiliary tier
    ///   half-prefilled, cost O(tools) in every prompt - scaling with exactly what lazy loading exists
    ///   to remove - and did the retriever's job for it. If the prompt maps a request onto a tool name,
    ///   ``ColBERTToolRetriever`` is decoration.
    /// - **Toolset labels with counts** (`Apple Notes (4), Clipboard (2)`) were cheaper, O(toolsets),
    ///   but actively caused the failure documented in `STEPS/ISSUES/tool-search-prompt-echo.md`: the
    ///   Title-Cased label was the most copyable noun phrase in the section, so the model lifted
    ///   "Apple Notes" verbatim as its search query and then as a filesystem path
    ///   (`read_file ~/Apple Notes`). Across five runs of one prompt, every run that wrote a lowercase
    ///   query succeeded and every run that copied the label failed.
    ///
    /// So the section is now O(1) - constant no matter how many tools or servers exist - and the only
    /// thing it teaches is the *rule*. The model does not need to know what capabilities exist to know
    /// it must search when its visible tools do not cover a request, and deriving the query from the
    /// user's own words is what the retriever is built to handle.
    ///
    /// Three lines carry the behaviour: the obligation to search before answering; the query-shape
    /// instruction ("the action… not the name of an app, a folder, or a file", which targets the echo
    /// above); and the refusal-block ("never tell the user you are unable…"). The last restores
    /// tier-agnostically something the per-tool prose used to do - `AppleNotesMiddleware` said "never
    /// claim you can't" - which went out with that prose and left nothing in its place.
    ///
    /// Constant for the life of the run, so it sits inside the cached prefix and never moves the
    /// fingerprint.
    var systemPrompt: String {
        """
        ## Your tools: core and auxiliary

        You have two kinds of tools, and you can use all of them. **Core tools** are the ones whose \
        schemas appear in this request - call them directly. You also have **auxiliary tools** that are \
        not listed here and whose schemas are not loaded: you cannot see their names or their parameters \
        until you look them up.

        **If a request needs something your visible tools do not cover, call `search_tools` before you \
        answer.** Describe the action you need in a few lowercase words ("list notes", "check git \
        history") - not the name of an app, a folder, or a file. It returns the matching tools' real \
        names and exact signatures, and you then call one by name like any other tool. One extra step, \
        then it works normally.

        Never tell the user you are unable to do something before you have searched for a tool that \
        does it. Never guess an auxiliary tool's name and never treat one as a file path: \
        `search_tools` is the only way to learn a real name. If your output format will not let you \
        call a tool that is absent from your schema, call `run_tool` with the name and arguments \
        instead.
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
    /// Imperative, like the prompt section: the description is the only thing some models read before
    /// deciding whether a tool applies, so it states the trigger rather than just the capability.
    var description: String {
        "Look up your auxiliary tools - the ones you have but whose schemas are not loaded. Call this "
            + "whenever a request needs something your visible tools do not cover, before telling the "
            + "user anything is impossible. Describe what you need (e.g. \"list notes\", \"check git "
            + "history\") and this returns the matching tools' names and exact signatures, which you "
            + "then call directly."
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
        // can reject beats a dead end - and this is the one place naming the areas helps, because there
        // is no tool to name instead. Lowercased, so a re-query copies a phrase rather than a proper
        // noun that reads like a path.
        guard !matches.isEmpty else {
            return ToolOutput(
                "No tool matched \"\(query)\". Areas that were searched: \(toolsets.lowercased()). "
                    + "Call search_tools again describing the action you need, or answer without a tool."
            )
        }

        // The query is deliberately *not* echoed back. It was ("Found 5 tool(s) for \"Apple Notes\""),
        // and re-injecting the model's own noun phrase is half of the failure recorded in
        // `STEPS/ISSUES/tool-search-prompt-echo.md` - the model went on to read that string as a path.
        //
        // The legend is not decoration either: a bare parameter name meaning "required" is an inference,
        // and the point of this result is that the model can call the tool correctly first time.
        var lines = ["Found \(matches.count) matching tool(s). "
            + "In the signatures below `arg!` must be passed and `arg?` is optional.", ""]
        var used = lines.joined(separator: "\n").count
        var omitted = 0
        var abbreviated = 0

        // Two renderings per tool. The full one explains every parameter - including the optional ones,
        // because that is where a plausible-but-wrong guess lives: `list_notes(folder?)` reads like a
        // disk path to an agent that also has filesystem tools, and only the description says otherwise.
        // The compact one is signature and purpose alone.
        let entries = matches.compactMap { byName[$0.name] }.map { document -> (compact: String, full: String) in
            let compact = "  \(document.signature)\n      \(document.summary)"
            var full = compact
            for parameter in document.parameters where !parameter.description.isEmpty {
                full += "\n      \(parameter.label) - \(parameter.description)"
            }
            return (compact, full)
        }

        // Reserve room for every match's compact form *first*, then spend what is left upgrading the
        // best-ranked ones to full detail. Filling greedily with full entries instead looks equivalent
        // and is not: a few verbose tools consume the whole budget and the tail gets dropped entirely
        // rather than abbreviated - the opposite of the intended degradation, since a tool listed with a
        // bare signature can still be called while an omitted one cannot be found at all.
        var rendered: [String] = []
        for entry in entries {
            guard used + entry.compact.count <= Self.resultBudget else {
                omitted += 1
                continue
            }
            used += entry.compact.count + 1
            rendered.append(entry.compact)
        }
        for index in rendered.indices {
            let delta = entries[index].full.count - entries[index].compact.count
            if used + delta <= Self.resultBudget {
                used += delta
                rendered[index] = entries[index].full
            } else {
                abbreviated += 1
            }
        }
        lines += rendered
        if abbreviated > 0 {
            lines.append("")
            lines.append(
                "(\(abbreviated) of these are listed without parameter detail to stay within the result "
                    + "size - search for one by name to see it in full.)"
            )
        }
        lines.append("")
        if omitted > 0 { lines.append("(\(omitted) further match(es) omitted to stay within the result size.)") }
        // End on the action, naming the best match. What sits last is what the model reaches for, and
        // this used to end with a Title-Cased list of area labels followed by an invitation to search
        // *again* - so the most recent thing it read was a copyable noun phrase and a fallback, not a
        // callable name. See `STEPS/ISSUES/tool-search-prompt-echo.md`.
        if let best = matches.first?.name {
            lines.append(
                "Call `\(best)` now if it does what you need, or one of the others above by name. Only "
                    + "if none of them fit, call search_tools again describing the action differently."
            )
        }
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
