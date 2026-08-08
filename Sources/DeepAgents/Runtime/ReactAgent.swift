import Foundation

/// A compiled agent — Mispher's port of the object returned by LangChain's
/// `create_agent`. It wires middleware, tools, short-term memory, and an underlying
/// `ChatModel` into a ReAct run. Build one with `createAgent(...)`.
///
/// `ReactAgent` owns the ReAct loop: it asks the `ChatModel` for one run-scoped
/// `ModelTurnSession`, then drives it one turn per round — running middleware hooks
/// around each model call, dispatching the tool calls the model emits, and feeding the
/// results back as the next round's input. This mirrors LangChain's graph (`model` ⇄
/// `tools`, looping) and keeps the structured exchange in `AgentState.messages`.
///
/// Each round the model node rebuilds its prompt from `state.messages` against a fresh KV
/// cache (see `RebuildTurnSession`), so middleware may rewrite the conversation between
/// rounds — `ScreenshotMiddleware`, for one, splices a captured image into history for the
/// next round. `wrapModelCall` must still not invoke its handler more than once per round.
public struct ReactAgent: Sendable {
    let model: any ChatModel
    /// The planner's context window in tokens, when the model reports one — for a host's context-usage
    /// meter (the same window summarization's 85% trigger measures against). `nil` when unknown.
    public var contextWindowTokens: Int? { model.contextWindowTokens }
    public let tools: [any AgentTool]
    let systemPrompt: String?
    public let middleware: [any AgentMiddleware]
    let memory: (any AgentCheckpointer)?
    /// Hard cap on model rounds, so a model that loops tool calls forever still
    /// terminates — LangChain's `recursion_limit`. On hitting it the loop runs one final
    /// tool-less turn (see `forceFinalAnswer`) rather than raising.
    let maxIterations: Int
    /// Optional developer sink: every message this run produces is appended to it, in
    /// order (human input, assistant turns with tool calls, tool results).
    let messageLog: (any AgentMessageLog)?

    /// Run the agent over `input` (typically a single human message). Prior turns for
    /// `threadId` are loaded from `memory` (short-term memory) and the updated
    /// conversation is saved back. Progress streams via `onEvent`, in order.
    /// - Returns: `true` on success, `false` if the run failed.
    @discardableResult
    public func run(
        _ input: [AgentMessage],
        threadId: String? = nil,
        onEvent: @Sendable @escaping (AgentEvent) -> Void
    ) async -> Bool {
        do {
            let prior = await loadHistory(threadId)
            var state = AgentState(messages: prior + input)
            // Seed the values summarization reads from state: the thread id (for its archive) and the
            // fixed prompt overhead (system prompt + tool schemas), so its 85% trigger measures the
            // real request size, not just the conversation.
            seedSummarizationState(&state, threadId: threadId)

            // Log this turn's new input (prior history was logged on earlier runs).
            for message in input { await log(message, threadId: threadId, round: nil) }

            for middleware in middleware { await middleware.beforeAgent(&state) }

            // One run-scoped, stateless model node; this loop owns the iteration and hands
            // it the full conversation each round.
            let session = model.makeSession()
            let run = RunContext(session: session, threadId: threadId, onEvent: onEvent)
            var round = 0
            var repeats = RepeatGuard()
            // Set once the loop has already nudged a silent model for its answer, so a model that
            // only ever thinks ends the run instead of being asked again and again.
            var nudgedForAnswer = false

            agentLoop: while true {
                round += 1
                if round > maxIterations {
                    try await forceFinalAnswer(state: &state, round: round, run: run)
                    break agentLoop
                }

                await runBeforeModel(&state, onEvent: onEvent)

                let handler = composedModelHandler(session: session, onEvent: onEvent)
                let request = ModelRequest(
                    messages: state.messages, systemPrompt: systemPrompt, tools: tools
                )
                let modelStarted = Date()
                let response = try await handler(request)
                // Normalize the emitted tool-call names here, at the one place the model's output
                // enters the loop, so exactly one spelling of a tool exists from this line on -
                // in the stored message, the duplicate-round signature, the events, the middleware
                // chain and the approval gate alike. Anything reconciling spellings further down
                // would be a second rule, disagreeing with this one.
                let message = Self.normalizingToolCallNames(response.message, tools: tools)
                state.messages.append(message)
                await recordModelTurn(message, round: round, threadId: threadId, started: modelStarted)

                for middleware in middleware.reversed() { await middleware.afterModel(&state) }

                // Honor a middleware `jump_to` before deciding what to do next.
                switch state.jumpTo {
                case .end: state.jumpTo = nil; break agentLoop
                case .model: state.jumpTo = nil; continue agentLoop
                case .tools, .none: state.jumpTo = nil
                }

                let calls = message.toolCalls
                let malformed = message.malformedToolCallBlocks
                onEvent(.roundCompleted(hadToolCalls: !calls.isEmpty || !malformed.isEmpty))
                if calls.isEmpty, malformed.isEmpty {
                    if !nudgedForAnswer {
                        nudgedForAnswer = try await nudgeIfSilent(
                            message, state: &state, round: round, run: run
                        )
                    }
                    break agentLoop
                }

                // A round whose only tool calls were unparseable is not a final answer:
                // feed the error back so the model can re-emit the call or answer in text.
                if calls.isEmpty {
                    await appendMalformedFeedback(malformed, state: &state, round: round, threadId: threadId)
                    continue agentLoop
                }

                switch repeats.verdict(on: calls) {
                case .stop:
                    try await forceFinalAnswer(state: &state, round: round, run: run)
                    break agentLoop
                case .redirect:
                    await appendDuplicateFeedback(
                        RepeatedRound(calls: calls, previouslyFailed: repeats.previousRoundFailed),
                        state: &state, round: round, threadId: threadId, onEvent: onEvent
                    )
                    continue agentLoop
                case .dispatch:
                    repeats.previousRoundFailed = await dispatchRound(
                        message, state: &state, round: round,
                        threadId: threadId, onEvent: onEvent
                    )
                }
            }

            for middleware in middleware.reversed() { await middleware.afterAgent(&state) }

            await saveHistory(threadId, messages: state.messages)
            onEvent(.completed)
            return true
        } catch {
            onEvent(.failed(Self.describe(error)))
            return false
        }
    }

    /// The per-round model handler: the session turn wrapped in every middleware's
    /// `wrapModelCall`, first-registered middleware outermost. The innermost handler runs
    /// one model turn over the request's messages (honoring middleware history rewrites),
    /// streams visible text via `onEvent(.token(...))`, and returns the assistant message
    /// (text + any tool calls). It does NOT run tools.
    private func composedModelHandler(
        session: any ModelTurnSession,
        onEvent: @Sendable @escaping (AgentEvent) -> Void
    ) -> (ModelRequest) async throws -> ModelResponse {
        let base: (ModelRequest) async throws -> ModelResponse = { request in
            let message = try await session.nextTurn(
                messages: request.messages,
                systemPrompt: request.systemPrompt,
                tools: request.tools,
                onChunk: Self.streamHandler(onEvent)
            )
            return ModelResponse(message: message)
        }
        var handler = base
        for middleware in middleware.reversed() {
            let next = handler
            handler = { request in try await middleware.wrapModelCall(request, next) }
        }
        return handler
    }

    /// Seed the `state.values` entries ``SummarizationMiddleware`` reads: the thread id (so it can
    /// address the archive) and the fixed prompt overhead text (system prompt + tool schemas), so its
    /// trigger counts the whole request, not just the messages. The overhead is only computed when a
    /// summarizer is actually registered.
    private func seedSummarizationState(_ state: inout AgentState, threadId: String?) {
        if let threadId { state.values[SummarizationMiddleware.threadIdStateKey] = threadId }
        guard middleware.contains(where: { $0 is SummarizationMiddleware }) else { return }
        let overhead = SummarizationMiddleware.promptOverheadText(systemPrompt: systemPrompt, tools: tools)
        if !overhead.isEmpty { state.values[SummarizationMiddleware.promptOverheadStateKey] = overhead }
    }

    /// The collaborators every step of one run shares: the model node it drives, the thread it
    /// logs against, and where its events go. Bundled because threading the three of them through
    /// each helper by hand is what pushed those signatures past readable.
    private struct RunContext {
        let session: any ModelTurnSession
        let threadId: String?
        let onEvent: @Sendable (AgentEvent) -> Void
    }

    /// The duplicate-round guard: small models can re-issue the identical tool call(s) round after
    /// round (the convergence bug's signature). It tracks the previous round's call set, because a
    /// set identical to it can't produce new information — anything legitimately re-run (a file
    /// re-read after an edit, a fresh screenshot after a delegation) has a different call in
    /// between, so it is never consecutive-identical.
    private struct RepeatGuard {
        /// Whether every call in the round just dispatched failed, which `appendDuplicateFeedback`
        /// needs to word its redirect truthfully.
        var previousRoundFailed = false
        private var previousSignature: [String]?
        private var repeats = 0

        /// What to do with this round's calls: run them, redirect the model to the result it
        /// already has (the first repeat is never re-executed), or give up on it answering by
        /// itself after `maxRepeatedRounds` and force the final answer.
        enum Verdict { case dispatch, redirect, stop }

        mutating func verdict(on calls: [AgentToolCall]) -> Verdict {
            let signature = calls.map(\.signature).sorted()
            if signature == previousSignature {
                repeats += 1
            } else {
                repeats = 0
                previousSignature = signature
            }
            if repeats >= ReactAgent.maxRepeatedRounds { return .stop }
            return repeats > 0 ? .redirect : .dispatch
        }
    }

    /// A round with no tool calls ordinarily ends the run - its text is the final answer. But "no
    /// tool calls" is not "here is the answer": a reasoning model can spend the whole turn inside
    /// `<think>` and never reach the visible text, and taking that as the answer completes the run
    /// with an empty reply. Observed on-device on a 2.6B: five rounds of real tool work, then 6,788
    /// reasoning chunks, zero answer tokens, and nothing shown to the user.
    ///
    /// So a silent turn is nudged, once, for the answer it never wrote. Returns whether it did -
    /// the caller keeps that, because a model that only ever thinks must end the run rather than be
    /// asked again every round until the iteration cap.
    private func nudgeIfSilent(
        _ message: AgentMessage, state: inout AgentState, round: Int, run: RunContext
    ) async throws -> Bool {
        guard message.text.isBlank else { return false }
        // Drop the silent turn first: an empty assistant message renders as an empty turn in the
        // chat template, which is not a shape the model was trained on.
        let thinking = message.reasoning
        state.messages.removeLast()
        try await forceFinalAnswer(
            state: &state, round: round, run: run, fallbackReasoning: thinking
        )
        return true
    }

    /// Run every middleware's `beforeModel` hook, then emit a `.contextCompacted` event if one of
    /// them (summarization) rewrote the history this round — it leaves the outcome in `state.values`,
    /// mirroring how a tool's `todos` update becomes `.todosUpdated`.
    private func runBeforeModel(
        _ state: inout AgentState, onEvent: @Sendable (AgentEvent) -> Void
    ) async {
        for middleware in middleware { await middleware.beforeModel(&state) }
        guard let outcome = state.values[SummarizationMiddleware.outcomeStateKey] as? CompactionOutcome
        else { return }
        onEvent(.contextCompacted(tokensBefore: outcome.tokensBefore, tokensAfter: outcome.tokensAfter))
        state.values[SummarizationMiddleware.outcomeStateKey] = nil
    }

    /// Append the model's turn to the developer log with its generation time. Split out so the loop
    /// body stays within length.
    private func recordModelTurn(
        _ message: AgentMessage, round: Int, threadId: String?, started: Date
    ) async {
        await messageLog?.append(
            message, threadId: threadId,
            context: AgentLogContext(
                modelID: model.modelID, round: round,
                generationSeconds: Date().timeIntervalSince(started)
            )
        )
    }

    /// Map a session's streamed pieces onto agent events: visible answer `text` to `.token`,
    /// chain-of-thought `reasoning` to `.reasoningToken` (its own channel for the UI's disclosure).
    private static func streamHandler(
        _ onEvent: @escaping @Sendable (AgentEvent) -> Void
    ) -> @Sendable (AgentStreamChunk) -> Void {
        { chunk in
            switch chunk {
            case .text(let text): onEvent(.token(text, isFinal: false))
            case .reasoning(let reasoning): onEvent(.reasoningToken(reasoning))
            }
        }
    }

    /// Dispatch one round's tool calls (from the model's message), appending every result
    /// to the conversation so the model (next round) and any later tool this round see the
    /// full exchange; also feeds back the round's malformed blocks (if any) and surfaces
    /// todo-list updates. Returns whether every call in the round failed, which the
    /// duplicate-round guard uses to word its redirect truthfully.
    ///
    /// A run of consecutive parallel-safe calls (see ``AgentTool/isParallelSafe``) runs as one
    /// concurrent batch; every other call keeps its own place in the serial order and still sees
    /// everything dispatched before it. Whatever the batching, results are appended in the order
    /// the model emitted the calls.
    @discardableResult
    private func dispatchRound(
        _ message: AgentMessage,
        state: inout AgentState,
        round: Int,
        threadId: String?,
        onEvent: @Sendable @escaping (AgentEvent) -> Void
    ) async -> Bool {
        var todosTouched = false
        var failures = 0
        for batch in dispatchBatches(message.toolCalls) {
            // Every call in a concurrent batch is handed the same state snapshot - the one taken
            // before the batch ran. That is exactly what `isParallelSafe` declares: nothing in the
            // batch needs a sibling's result.
            let outcomes: [ToolOutcome]
            if batch.count == 1 {
                onEvent(Self.startEvent(batch[0]))
                outcomes = await [dispatchTool(batch[0], tools: tools, state: state, onEvent: onEvent)]
            } else {
                outcomes = await dispatchConcurrently(batch, state: state, onEvent: onEvent)
            }
            for (result, update, failed) in outcomes {
                if failed { failures += 1 }
                merge(update, into: &state.values)
                if update?.values["todos"] != nil { todosTouched = true }
                state.messages.append(result)
                await log(result, threadId: threadId, round: round)
            }
        }
        if !message.malformedToolCallBlocks.isEmpty {
            await appendMalformedFeedback(
                message.malformedToolCallBlocks, state: &state, round: round, threadId: threadId
            )
        }
        if todosTouched, let todos = state.values["todos"] as? [TodoItem] {
            onEvent(.todosUpdated(todos))
        }
        return !message.toolCalls.isEmpty && failures == message.toolCalls.count
    }

    /// A repeated call set together with how its first execution went - the two facts
    /// `appendDuplicateFeedback` needs to word its redirect.
    private struct RepeatedRound {
        let calls: [AgentToolCall]
        let previouslyFailed: Bool
    }

    /// Feed back a redirect instead of re-executing a consecutive-duplicate call set:
    /// the result is already in the conversation, so re-running burns seconds for
    /// nothing (observed on-device: `read_clipboard` re-run for an identical result at
    /// ~7s a round). Each duplicate call still gets a `tool`-role response — the trained
    /// chat format pairs every emitted call with a result, so skipping silently would be
    /// off-distribution.
    ///
    /// `repeated.previouslyFailed` swaps the wording when the repeated round produced only
    /// errors. Telling the model its result "is in the conversation above" when the conversation
    /// holds an error is a lie it acts on: observed on-device, an unknown-tool failure repeated
    /// once drew the redirect, and the model then answered from a web fetch it never made.
    private func appendDuplicateFeedback(
        _ repeated: RepeatedRound,
        state: inout AgentState,
        round: Int,
        threadId: String?,
        onEvent: @Sendable @escaping (AgentEvent) -> Void
    ) async {
        for call in repeated.calls {
            let text = Self.errorJSON(
                repeated.previouslyFailed
                    ? "You already called \(call.name) with the same arguments and it failed - "
                    + "the error is in the conversation above, and repeating the call unchanged "
                    + "will fail the same way. Fix what the error reports (check the tool name "
                    + "against the tool list, or the arguments), call a different tool, or answer "
                    + "the user now with what you have."
                    : "You already called \(call.name) with the same arguments - its result is "
                    + "in the conversation above. Use that result, call a different tool, "
                    + "or answer the user now."
            )
            onEvent(.toolFailed(name: call.name, error: text, callID: call.id))
            let message = AgentMessage.tool(text, toolCallID: call.id)
            state.messages.append(message)
            await log(message, threadId: threadId, round: round)
        }
    }

    /// Append the parse-error feedback for this round's unparseable tool-call blocks.
    private func appendMalformedFeedback(
        _ blocks: [String], state: inout AgentState, round: Int, threadId: String?
    ) async {
        let feedback = AgentMessage.tool(Self.malformedFeedback(blocks))
        state.messages.append(feedback)
        await log(feedback, threadId: threadId, round: round)
    }

    /// Append to the developer message log. `modelID` records which model *drove* the
    /// turn, so it's attached only to `.ai` messages — human input and tool results
    /// aren't model-generated and would otherwise carry a misleading model id.
    private func log(_ message: AgentMessage, threadId: String?, round: Int?) async {
        await messageLog?.append(
            message, threadId: threadId,
            context: AgentLogContext(modelID: message.role == .ai ? model.modelID : nil, round: round)
        )
    }

    /// Consecutive identical-call-set rounds tolerated before the loop gives up and
    /// forces a final answer. A repeated round is never re-executed (see
    /// `appendDuplicateFeedback`) — this cap ends the run when the model won't move on
    /// even after being redirected to the result it already has.
    static let maxRepeatedRounds = 2

    /// Most tool calls run at once in one concurrent batch. A round is usually two or three
    /// reads, so this only bites on a model that emits a long burst - and a cap keeps that burst
    /// from opening a dozen sockets or subprocesses at once. Calls beyond it run in the next
    /// batch, still in call order.
    static let maxConcurrentToolCalls = 4

    /// Prefixes an answer salvaged from the model's own reasoning, so the user is told they are
    /// reading working-out rather than a considered reply (see ``forceFinalAnswer``).
    static let salvagedAnswerNote =
        "I didn't finish writing an answer. Here is what I was working through:"

    /// The last resort, when the model produced neither an answer nor any reasoning to fall back
    /// on. Names what happened and what to do about it, because a blank reply does neither.
    static let noAnswerNote =
        "I stopped without producing an answer. Ask me again, or rephrase the request."

    /// The turn appended to the end of the conversation when the loop asks for the answer the model
    /// never wrote. It is never stored: the user did not type it, so it must not appear in their
    /// saved thread - only the answer it produces is kept.
    ///
    /// When there is reasoning to hand back, it is quoted, because the model then has to summarise
    /// what it already worked out instead of deriving it again - and deriving it again is what
    /// exhausted the turn in the first place.
    static func answerRequest(reasoning: String?) -> AgentMessage {
        guard let reasoning, !reasoning.isBlank else {
            return .human(
                "You stopped without writing an answer. Give me your final answer now, in plain "
                    + "text, using the conversation above."
            )
        }
        return .human("""
        You were working on this and wrote the reasoning below, but you never wrote the answer \
        itself:

        \(reasoning)

        Give me your final answer now, in plain text. Do not call any tools.
        """)
    }

    /// Longest tool result (in characters) fed back to the model. LFM models have a 32k
    /// context; one oversized `read_file`/`task` result can crowd out the conversation,
    /// so anything longer is cut with a note saying how much was dropped.
    static let maxToolResultCharacters = 6000

    /// One last model turn with NO tools declared (so the chat template omits the tool
    /// list entirely — the strongest "answer now" signal) plus an explicit instruction to
    /// answer in text. Used when the loop is cut short (iteration cap, duplicate-round
    /// guard, a turn that produced only reasoning) so the user still gets an answer instead
    /// of a dangling tool result. Calls the session directly: middleware `wrapModelCall`
    /// guidance describes tools that are deliberately absent here. Any tool calls the model
    /// still emits are dropped.
    ///
    /// This is the last line of defence, so it does not hand back nothing: if even this turn
    /// stays inside `<think>`, the run answers with the thinking - its own first, else
    /// `fallbackReasoning` from the turn that prompted the nudge. A visible train of thought is a
    /// poor answer, but it is an answer; a blank reply tells the user only that something broke.
    private func forceFinalAnswer(
        state: inout AgentState, round: Int, run: RunContext, fallbackReasoning: String? = nil
    ) async throws {
        let onEvent = run.onEvent
        let nudge = """
        Tool calling is now disabled. Using the conversation above, give the user your \
        final answer in plain text. Do not call any tools.
        """
        let prompt = [systemPrompt, nudge].compactMap { $0 }.joined(separator: "\n\n")
        let started = Date()
        // The ask goes at the *end* of the conversation, not only in the system prompt: the prompt
        // sits thousands of tokens from where generation starts, and a model that has just thought
        // itself silent is being handed the identical conversation again. A turn immediately before
        // the generation point is the strongest position, and it can carry the model's own
        // reasoning back so answering is a summary rather than a re-derivation.
        //
        // It is a `.human` turn rather than a mid-conversation `.system` one because system-role
        // messages inside `messages` are model-gated - supported on Opus 4.8 and later, rejected
        // with a 400 on Sonnet 5, Haiku, and older models. This path runs when a turn has already
        // gone wrong; it must not be the thing that fails.
        let messages = state.messages + [Self.answerRequest(reasoning: fallbackReasoning)]
        // Start outside the reasoning channel. On a model whose template opens `<think>` for every
        // turn, an answer only reaches `text` once the model emits `</think>` - so a turn that has
        // already thought itself silent is being asked to answer on a channel that is not open.
        // Closing it up front removes the requirement instead of restating it.
        let message = try await run.session.nextTurn(
            messages: messages, systemPrompt: prompt, tools: [],
            startingOutsideReasoning: true,
            onChunk: Self.streamHandler(onEvent)
        )
        var answer = message.text
        if answer.isBlank {
            // Say whose words these are. Unlabelled, reasoning reads as the answer - "let me try a
            // different approach" looks like the agent is still working, when in fact it has
            // stopped - and the user can't tell a considered reply from a salvaged monologue.
            let thinking = [message.reasoning, fallbackReasoning].compactMap { $0 }.first { !$0.isBlank }
            // With nothing to salvage either, the run still has to say something: a blank reply
            // tells the user only that something broke, and not even that it was the model.
            answer = thinking.map { Self.salvagedAnswerNote + "\n\n" + $0 } ?? Self.noAnswerNote
            // A host builds the visible answer from `.token`, not from the stored message, so the
            // salvaged text has to be streamed as well or the reply is blank on screen.
            onEvent(.token(answer, isFinal: true))
        }
        let final = AgentMessage.ai(answer)
        state.messages.append(final)
        await messageLog?.append(
            final, threadId: run.threadId,
            context: AgentLogContext(
                modelID: model.modelID, round: round,
                generationSeconds: Date().timeIntervalSince(started)
            )
        )
        onEvent(.roundCompleted(hadToolCalls: false))
    }

    /// The `tool`-role error fed back when the model emitted tool-call blocks that could
    /// not be parsed, so it can re-emit them correctly (or answer in text) next round.
    static func malformedFeedback(_ blocks: [String]) -> String {
        let shown = blocks.map { block in
            block.count > 200 ? String(block.prefix(200)) + "…" : block
        }.joined(separator: "\n")
        return errorJSON(
            "Your tool call could not be parsed: \(shown). Re-emit it as "
                + "[tool_name(argument=\"value\")] with every string argument quoted, "
                + "or answer in plain text if you are done."
        )
    }

    /// Check a parsed call against the tool's declared parameters; nil when acceptable,
    /// else a correction message for the model. Deliberately conservative — only the
    /// unambiguous violations (missing required parameter, value outside a declared
    /// `enum`) are rejected, because tools like `write_todos` accept looser shapes than
    /// their schema advertises and coerce them in `execute`.
    static func schemaViolation(_ call: AgentToolCall, tool: any AgentTool) -> String? {
        var problems: [String] = []
        for parameter in tool.parameters {
            let value = call.arguments[parameter.name]
            if parameter.isRequired, value == nil {
                problems.append("missing required parameter `\(parameter.name)`")
            }
            // An empty enum (e.g. the `task` tool with no subagents registered) would
            // reject everything with a blank allowed-list — let the tool itself produce
            // its richer error instead.
            if let allowed = parameter.extraProperties["enum"] as? [String], !allowed.isEmpty,
               case .string(let raw)? = value, !allowed.contains(raw) {
                problems.append(
                    "`\(parameter.name)` must be one of: "
                        + allowed.joined(separator: ", ") + " (got \"\(raw)\")"
                )
            }
        }
        guard !problems.isEmpty else { return nil }
        return "Invalid call to '\(call.name)': " + problems.joined(separator: "; ")
            + ". Fix the arguments and call it again."
    }

    /// Render an error as the JSON object shape (`{"error": "…"}`) LFM tool-use examples
    /// feed back, with proper escaping.
    static func errorJSON(_ text: String) -> String {
        let object = ["error": text]
        guard let data = try? JSONSerialization.data(withJSONObject: object),
              let json = String(data: data, encoding: .utf8)
        else { return "{\"error\": \"unencodable error\"}" }
        return json
    }

    /// Cut an oversized tool result down to `maxToolResultCharacters`, noting the cut so
    /// the model knows it saw a prefix rather than the whole thing.
    static func truncatedToolResult(_ content: String) -> String {
        guard content.count > maxToolResultCharacters else { return content }
        return String(content.prefix(maxToolResultCharacters))
            + "\n[Result truncated: showing the first \(maxToolResultCharacters) of "
            + "\(content.count) characters.]"
    }

    // MARK: - Tool dispatch

    /// `message` with each tool call renamed to the tool it names, under that tool's own spelling.
    ///
    /// This is the ingress half of ``ToolName`` - the egress half is what published the name in the
    /// first place. A call that names nothing is left exactly as the model wrote it, so the
    /// unknown-tool error quotes what was actually emitted.
    static func normalizingToolCallNames(_ message: AgentMessage, tools: [any AgentTool]) -> AgentMessage {
        guard message.toolCalls.contains(where: { call in !tools.contains { $0.name == call.name } })
        else { return message }
        var normalized = message
        normalized.toolCalls = message.toolCalls.map { call in
            guard let tool = ToolName.resolve(call.name, in: tools), tool.name != call.name else { return call }
            return AgentToolCall(id: call.id, name: tool.name, arguments: call.arguments)
        }
        return normalized
    }

    /// What dispatching one call produced: the `tool`-role result message, any state update the
    /// tool returned, and whether the call failed.
    typealias ToolOutcome = (message: AgentMessage, stateUpdate: AgentStateUpdate?, failed: Bool)

    /// The tools whose calls may fan out: those declaring ``AgentTool/isParallelSafe``.
    ///
    /// Being gated is deliberately *not* a disqualification. Every read-only tool defaults to
    /// `.ask` in ``MiddlewareCatalog``, so excluding gated tools would exclude precisely the ones
    /// worth parallelising - and it would do so even when the host answers the gate itself
    /// (ripple's accept-all, an allowlist, a deny rule), where no human is asked anything. The
    /// invariant that actually matters is one approval request at a time, and ``ApprovalRequestQueue``
    /// enforces it where the asking happens rather than by serialising dispatch.
    private var parallelSafeToolNames: Set<String> {
        Set(tools.filter(\.isParallelSafe).map(\.name))
    }

    /// Split a round's calls into dispatch batches, preserving order: each run of consecutive
    /// parallel-safe calls becomes one batch (at most ``maxConcurrentToolCalls`` long) that runs
    /// concurrently, and every other call becomes a batch of one. A call naming no tool of ours
    /// is not parallel-safe by this test, so the unknown-tool error keeps its place in the order.
    private func dispatchBatches(_ calls: [AgentToolCall]) -> [[AgentToolCall]] {
        let parallelSafe = parallelSafeToolNames
        var batches: [[AgentToolCall]] = []
        for call in calls {
            let canJoin = parallelSafe.contains(call.name)
                && (batches.last.map { $0.count < Self.maxConcurrentToolCalls && parallelSafe.contains($0[0].name) } ?? false)
            if canJoin { batches[batches.count - 1].append(call) } else { batches.append([call]) }
        }
        return batches
    }

    /// The `.toolStarted` announcing a call. Separate from ``dispatchTool`` because a concurrent
    /// batch announces every call it holds *before* any of them runs - a host opens all of the
    /// batch's cards at once, which is what actually happens. `batchID` is nil for a call that
    /// runs on its own, and shared by the calls that run together.
    private static func startEvent(_ call: AgentToolCall, batchID: UUID? = nil) -> AgentEvent {
        .toolStarted(
            name: call.name, input: call.describedArguments, callID: call.id, batchID: batchID
        )
    }

    /// Run a batch of parallel-safe calls at once and return their outcomes **in call order**.
    ///
    /// Results come back in the order the model emitted the calls, whatever order they finish in:
    /// the trained chat format pairs each call with its result, in order. Events are the opposite -
    /// they surface the moment they happen, so a batch's cards open together and each fills in as
    /// its call lands. That is only safe because every tool event carries its `callID`; see the
    /// pairing note on ``AgentEvent``.
    ///
    /// The children push their events into a stream this function drains, rather than calling
    /// `onEvent` themselves, so the host's handler is still invoked from one place at a time
    /// instead of from four task-group children at once.
    private func dispatchConcurrently(
        _ calls: [AgentToolCall],
        state: AgentState,
        onEvent: @Sendable @escaping (AgentEvent) -> Void
    ) async -> [ToolOutcome] {
        // One id for the whole batch: it is what lets a host say "these three ran together"
        // rather than showing them as three ordinary calls that happened to be adjacent.
        let batchID = UUID()
        for call in calls { onEvent(Self.startEvent(call, batchID: batchID)) }

        let (events, sink) = AsyncStream<AgentEvent>.makeStream()
        async let dispatched: [ToolOutcome] = withTaskGroup(of: DispatchedCall.self) { group in
            for (index, call) in calls.enumerated() {
                group.addTask {
                    let outcome = await dispatchTool(
                        call, tools: tools, state: state, onEvent: { sink.yield($0) }
                    )
                    return DispatchedCall(index: index, outcome: outcome)
                }
            }
            var byIndex: [Int: ToolOutcome] = [:]
            for await done in group { byIndex[done.index] = done.outcome }
            sink.finish()
            return calls.indices.compactMap { byIndex[$0] }
        }
        for await event in events { onEvent(event) }
        return await dispatched
    }

    /// One finished call of a concurrent batch: what it produced, and where it sat in the round so
    /// the batch can put the results back in call order.
    private struct DispatchedCall: Sendable {
        let index: Int
        let outcome: ToolOutcome
    }

    /// Execute one tool call through the `wrapToolCall` middleware chain. Returns the
    /// `.tool` result message (tagged with the originating call's id), any state update the
    /// tool produced, and whether the call failed. Errors are caught and returned as text so
    /// the model can recover rather than aborting; `failed` lets the duplicate-round guard
    /// tell "you already have this result" from "this call already errored".
    ///
    /// The caller has already emitted this call's `.toolStarted` (see ``startEvent(_:)``); every
    /// event from here on carries `call.id`, including the ones the tool itself emits through its
    /// `ToolContext`.
    private func dispatchTool(
        _ call: AgentToolCall,
        tools: [any AgentTool],
        state: AgentState,
        onEvent: @Sendable @escaping (AgentEvent) -> Void
    ) async -> ToolOutcome {
        // A plain exact match: the loop normalized this name before the call reached here, so a
        // miss means the model named no tool of ours, not that it spelled one differently.
        guard let tool = tools.first(where: { $0.name == call.name }) else {
            let names = tools.map(\.name).joined(separator: ", ")
            let text = Self.errorJSON("Unknown tool '\(call.name)'. Available tools: \(names).")
            onEvent(.toolFailed(name: call.name, error: text, callID: call.id))
            return (.tool(text, toolCallID: call.id), nil, true)
        }

        // Validate the call against the tool's declared schema before executing — the
        // on-device stand-in for Outlines-style schema enforcement (mlx-swift has no
        // constrained decoding). A violation is fed back as an error so the model can fix
        // the arguments next round instead of the tool failing in a less legible way.
        if let violation = Self.schemaViolation(call, tool: tool) {
            let text = Self.errorJSON(violation)
            onEvent(.toolFailed(name: call.name, error: text, callID: call.id))
            return (.tool(text, toolCallID: call.id), nil, true)
        }

        let context = ToolContext(state: state, onEvent: Self.stamping(call.id, onEvent))
        let captured = CapturedUpdate()

        let base: (ToolCallRequest) async throws -> AgentMessage = { request in
            let output = try await tool.execute(request.call.arguments, context)
            captured.value = output.stateUpdate
            captured.failed = output.isFailure
            return .tool(output.content, toolCallID: call.id)
        }

        var handler = base
        for middleware in middleware.reversed() {
            let next = handler
            handler = { request in try await middleware.wrapToolCall(request, next) }
        }

        do {
            var message = try await handler(ToolCallRequest(call: call, state: state))
            message.content = [.text(Self.truncatedToolResult(message.text))]
            // A tool can attach an image (e.g. the screenshot tool) or a line diff (edit_file)
            // via its state update; surface them on the completion event so the UI can show a
            // thumbnail / a diff card.
            let imageURL = (captured.value?.values[ScreenshotState.pendingKey] as? [URL])?.first
            let editDiff = captured.value?.values[EditDiffState.pendingKey] as? FileDiff
            // A tool that reported a failure without throwing is still a failure: report it as one
            // so the transcript doesn't tick a call that did not do what was asked, and so the
            // duplicate-round guard counts it like any other failed call.
            guard !captured.failed else {
                onEvent(.toolFailed(name: call.name, error: message.text, callID: call.id))
                return (message, captured.value, true)
            }
            onEvent(.toolCompleted(
                name: call.name, result: message.text, imageURL: imageURL,
                editDiff: editDiff, callID: call.id
            ))
            return (message, captured.value, false)
        } catch {
            let text = Self.errorJSON(Self.describe(error))
            onEvent(.toolFailed(name: call.name, error: text, callID: call.id))
            return (.tool(text, toolCallID: call.id), nil, true)
        }
    }

    /// Tag the events a tool emits through its `ToolContext` with the call they belong to - the
    /// `task` tool's `.toolProgress`, the shell's streamed output. The tool doesn't know its call
    /// id and shouldn't have to; without the tag a host couldn't route progress to the right card
    /// when two calls to the same tool are open at once.
    private static func stamping(
        _ callID: UUID, _ onEvent: @escaping @Sendable (AgentEvent) -> Void
    ) -> @Sendable (AgentEvent) -> Void {
        { event in
            if case .toolProgress(let name, let subagent, let delta, nil) = event {
                onEvent(.toolProgress(name: name, subagent: subagent, delta: delta, callID: callID))
            } else {
                onEvent(event)
            }
        }
    }

    /// Merge a tool's state update into the agent state — LangChain's `Command(update=…)`
    /// for non-message keys (later writes overwrite earlier ones).
    private func merge(_ update: AgentStateUpdate?, into values: inout [String: any Sendable]) {
        guard let update else { return }
        for (key, value) in update.values { values[key] = value }
    }

    // MARK: - Compaction

    /// Force a summarization pass on a thread's stored history, outside a run — the manual `/compact`
    /// (Ripple) and the Compact action (Mispher app). Loads the thread from `memory`, runs the
    /// ``SummarizationMiddleware`` (if one is registered) with `force: true`, saves the rewritten
    /// `[summary] + tail` back, and returns the outcome. Returns `nil` when there is no summarization
    /// middleware, no memory, or nothing safe to compact.
    @discardableResult
    public func compact(threadId: String?) async -> CompactionOutcome? {
        guard let summarizer = middleware.lazy.compactMap({ $0 as? SummarizationMiddleware }).first,
              threadId != nil
        else { return nil }
        var messages = await loadHistory(threadId)
        // Match the automatic path: count the fixed prompt overhead (system prompt + tool schemas) so
        // the reported before/after sizes reflect the real request, not just the conversation.
        let overheadText = SummarizationMiddleware.promptOverheadText(systemPrompt: systemPrompt, tools: tools)
        let overhead = summarizer.tokenCounter.count(overheadText)
        guard let outcome = await summarizer.compact(
            &messages, threadId: threadId, force: true, overheadTokens: overhead
        ) else { return nil }
        await saveHistory(threadId, messages: messages)
        return outcome
    }

    // MARK: - Memory

    private func loadHistory(_ threadId: String?) async -> [AgentMessage] {
        guard let threadId, let memory else { return [] }
        return await memory.load(threadId)
    }

    private func saveHistory(_ threadId: String?, messages: [AgentMessage]) async {
        guard let threadId, let memory else { return }
        // Persist the full structured conversation — human, assistant (with tool calls),
        // and tool results — so a resumed thread retains what the agent looked up. (A new
        // session only re-templates the text turns into a cold cache; see `MlxTurnSession`.)
        await memory.save(threadId, messages)
    }

    private static func describe(_ error: Error) -> String {
        (error as? LocalizedError)?.errorDescription ?? error.localizedDescription
    }
}

/// A tiny reference box so `dispatchTool`'s inner handler can hand the tool's state
/// update back out of the `wrapToolCall` chain (which only carries the `AgentMessage`).
private final class CapturedUpdate {
    var value: AgentStateUpdate?
    /// Set when the tool reported a failure rather than throwing - see ``ToolOutput/isFailure``.
    var failed = false
}

private extension AgentToolCall {
    /// A deterministic identity for the duplicate-round guard: name plus the key-sorted
    /// argument rendering. Two calls with the same name and arguments compare equal.
    var signature: String { "\(name)(\(describedArguments))" }
}

private extension String {
    /// Empty once whitespace is discounted. A model that answers with a stray newline has said
    /// nothing, and the loop must treat it the same as saying nothing at all.
    var isBlank: Bool { trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }
}
