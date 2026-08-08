import Foundation

/// Build a deep agent — Mispher's port of LangChain deepagents' `create_deep_agent`.
///
/// A deep agent is the ReAct core (``createAgent(model:tools:systemPrompt:middleware:memory:maxIterations:messageLog:)``)
/// plus deepagents' built-in middleware pillars: **planning** (`TodoListMiddleware` →
/// `write_todos`), a shared **filesystem** (`FilesystemMiddleware` →
/// `ls`/`read_file`/`write_file`/`edit_file`), and **subagents** (`SubAgentMiddleware` → `task`).
/// Hand it `subagents` to let the main agent delegate isolated subtasks; a general-purpose subagent
/// is available by default.
///
/// One ``FilesystemBackend`` is resolved here (the caller's `backend`, else an in-memory
/// ``StateBackend``) and shared by reference with both the main agent and every subagent, so files
/// written anywhere in the run are visible everywhere.
///
/// Human-in-the-loop (deepagents' `interrupt_on`): pass `interruptOn` plus an `approvalHandler` to
/// gate those tools behind the user's approve / edit / reject decision. The resulting
/// ``HumanInTheLoopMiddleware`` is registered on the main agent AND threaded into every subagent,
/// so a delegated subtask can't bypass the approval gate (in LangGraph, subgraph interrupts
/// propagate the same way).
///
/// - Parameters:
///   - model: the chat model the agent (and, by default, its subagents) runs on.
///   - tools: the agent's own tools; subagents whose `tools` is `nil` inherit these.
///   - systemPrompt: extra instructions, composed after the base deep-agent prompt.
///   - subagents: custom subagents the `task` tool can delegate to.
///   - middleware: extra middleware, appended after the built-in deep-agent stack.
///   - memory: optional thread-scoped short-term memory.
///   - backend: where the filesystem tools store files (default: a fresh in-memory `StateBackend`).
///   - interruptOn: tool name → human-in-the-loop policy; effective only with `approvalHandler`.
///   - approvalHandler: presents an interrupted call to the user and returns their decision.
///   - askUserHandler: presents the agent's `ask_user` questions to the user and returns their
///     answers. When set, an ``AskUserMiddleware`` is registered on the main agent so the model can
///     pause to ask for clarification; without it the `ask_user` tool is absent.
///   - includeFilesystem: register the filesystem pillar (default `true`).
///   - includeGeneralPurpose: register the built-in general-purpose subagent (default `true`).
///   - disabledToolNames: tools to drop from the agent entirely (the user's deactivations).
///   - auxiliaryToolNames: tools to keep out of the *prompt* while leaving them dispatchable, found
///     by the agent through `search_tools` (see ``ToolSearchMiddleware``). Empty disables the
///     feature entirely - no meta-tools, no prompt section, nothing changed.
///   - toolRetriever: ranks the auxiliary tools for `search_tools`. Defaults to
///     ``LexicalToolRetriever``, which needs no model; hosts with MLX can pass a ColBERT retriever.
///   - toolSearchLimit: how many matches a search returns by default.
///   - toolsetsByTool: toolset name per tool for anything outside ``MiddlewareCatalog`` - MCP tools,
///     keyed to the server that contributed them - so the search corpus and result can name them.
public func createDeepAgent(
    model: any ChatModel,
    tools: [any AgentTool] = [],
    systemPrompt: String? = nil,
    subagents: [SubAgent] = [],
    middleware: [any AgentMiddleware] = [],
    memory: (any AgentCheckpointer)? = nil,
    backend: (any FilesystemBackend)? = nil,
    interruptOn: [String: InterruptOnConfig] = [:],
    approvalHandler: ToolApprovalHandler? = nil,
    askUserHandler: AskUserHandler? = nil,
    includeFilesystem: Bool = true,
    includeGeneralPurpose: Bool = true,
    maxIterations: Int = 24,
    disabledToolNames: Set<String> = [],
    messageLog: (any AgentMessageLog)? = nil,
    summarization: SummarizationConfig? = .default,
    auxiliaryToolNames: Set<String> = [],
    toolRetriever: (any ToolRetriever)? = nil,
    toolSearchLimit: Int = 5,
    toolsetsByTool: [String: String] = [:]
) -> ReactAgent {
    let fileBackend: (any FilesystemBackend)? = includeFilesystem ? (backend ?? StateBackend()) : nil

    let humanInTheLoop: HumanInTheLoopMiddleware? = {
        guard let approvalHandler, !interruptOn.isEmpty else { return nil }
        return HumanInTheLoopMiddleware(interruptOn: interruptOn, approvalHandler: approvalHandler)
    }()

    // The base prompt's filesystem bullet names `write_file`, and its own doc comment gives the rule:
    // mentioning a tool the model has no schema for makes it call something that isn't there. Tiering
    // the filesystem auxiliary withholds that schema, so the bullet has to go with it - the tools are
    // still fully present and dispatchable, they are just found through `search_tools` now.
    let filesystemInPrompt = includeFilesystem && !auxiliaryToolNames.contains("write_file")
    let composedPrompt = [DeepAgentPrompt.system(includeFilesystem: filesystemInPrompt), systemPrompt]
        .compactMap { $0 }
        .joined(separator: "\n\n")

    // The built-in deep-agent stack (planning, then filesystem, then subagents), followed by any
    // caller-supplied middleware, with human-in-the-loop last so it gates every tool. Order
    // mirrors deepagents' assembly in `create_deep_agent`. Summarization goes first so its
    // `beforeModel` compacts the history before the other hooks read it; its archive is wired
    // automatically when the caller's `memory` checkpointer also conforms to `CompactionArchive`.
    var stack: [any AgentMiddleware] = []
    if let summarization {
        stack.append(SummarizationMiddleware(
            model: model, archive: memory as? CompactionArchive, config: summarization
        ))
    }
    stack.append(TodoListMiddleware())
    if let fileBackend { stack.append(FilesystemMiddleware(backend: fileBackend)) }
    stack.append(
        SubAgentMiddleware(
            model: model,
            baseTools: tools,
            subagents: subagents,
            backend: fileBackend,
            humanInTheLoop: humanInTheLoop,
            includeGeneralPurpose: includeGeneralPurpose
        )
    )
    stack += middleware
    // Let the agent pause to ask the user for clarification, when the host can present it.
    if let askUserHandler { stack.append(AskUserMiddleware(handler: askUserHandler)) }
    if let humanInTheLoop { stack.append(humanInTheLoop) }

    // Lazy tools. This is the only place that can build it: the auxiliary set is drawn from the
    // *assembled* tool list, which spans the caller's tools, the built-in pillars (a `filesystem`
    // tier means `FilesystemMiddleware`'s tools, constructed above) and the caller's middleware
    // alike. Same union rule as `createAgent`, so a disabled tool is never offered as auxiliary.
    //
    // It goes **first** in the stack, which makes it the OUTERMOST `wrapModelCall` - deliberately.
    // A capability middleware appends prose naming its own tools, and it decides whether to do so by
    // looking at `request.tools` (`contributesRenderedTools(to:)`). That check is only meaningful if
    // the auxiliary schemas are already gone by the time it runs, so the stripping has to happen
    // outside every middleware that might describe a stripped tool. Registered last, the schemas
    // vanished but every word of prose about them stayed - which told the model to call tools it had
    // no way to call. Its `run_tool` rewrite is unaffected by the move: the response still passes back
    // through here before `ReactAgent` normalizes or dispatches it, and it contributes no
    // `wrapToolCall`, so tool-dispatch nesting (and the approval gate's position) is untouched.
    if !auxiliaryToolNames.isEmpty {
        let assembled = (tools + stack.flatMap { $0.tools })
            .filter { !disabledToolNames.contains($0.name) }
        let auxiliary = assembled.filter { auxiliaryToolNames.contains($0.name) }
        if !auxiliary.isEmpty {
            stack.insert(
                ToolSearchMiddleware(
                    auxiliaryTools: auxiliary,
                    retriever: toolRetriever ?? LexicalToolRetriever(),
                    toolsetsByTool: toolsetsByTool,
                    limit: toolSearchLimit
                ),
                at: 0
            )
        }
    }

    return createAgent(
        model: model,
        tools: tools,
        systemPrompt: composedPrompt,
        middleware: stack,
        memory: memory,
        maxIterations: maxIterations,
        disabledToolNames: disabledToolNames,
        messageLog: messageLog
    )
}
