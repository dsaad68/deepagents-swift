@testable import DeepAgents
import Foundation
import Testing

/// Parallel dispatch within one round: a run of consecutive ``AgentTool/isParallelSafe`` calls
/// executes concurrently, while everything the framework already guaranteed holds unchanged -
/// results in call order, every event naming the call it belongs to, later serial tools still
/// seeing earlier ones' state, and the user still asked about one gated call at a time.
struct ParallelToolCallsTests {
    // MARK: - The calls actually overlap

    @Test func parallelSafeCallsInOneRoundRunConcurrently() async {
        let gate = ConcurrencyGate(expected: 3)
        let calls = (1 ... 3).map { index in
            AgentToolCall(name: "parallel_read", arguments: ["text": .string("f\(index)")])
        }
        let agent = createAgent(
            model: FakeChatModel(answer: "done", toolCalls: calls),
            tools: [ParallelProbeTool(gate: gate)]
        )

        let (ok, events) = await agent.collect([.human("read three files")])

        #expect(ok)
        // All three were in flight at once - the gate only releases once the third arrives.
        #expect(await gate.peakInFlight == 3)
        // Released together, so which completes first is a race; all three land.
        #expect(Set(events.toolCompletedResults.map(\.result)) == ["read: f1", "read: f2", "read: f3"])
    }

    @Test func serialToolsInOneRoundStillRunOneAtATime() async {
        let gate = ConcurrencyGate(expected: 2)
        let calls = (1 ... 2).map { index in
            AgentToolCall(name: "serial_read", arguments: ["text": .string("f\(index)")])
        }
        let agent = createAgent(
            model: FakeChatModel(answer: "done", toolCalls: calls),
            tools: [SerialProbeTool(gate: gate)]
        )

        _ = await agent.collect([.human("read two files")])

        // The same tool without the declaration keeps the old guarantee: one at a time.
        #expect(await gate.peakInFlight == 1)
    }

    @Test func aSerialCallBreaksTheBatchAroundIt() async {
        let gate = ConcurrencyGate(expected: 3)
        let calls = [
            AgentToolCall(name: "parallel_read", arguments: ["text": .string("a")]),
            AgentToolCall(name: "echo", arguments: ["text": .string("b")]),
            AgentToolCall(name: "parallel_read", arguments: ["text": .string("c")])
        ]
        let agent = createAgent(
            model: FakeChatModel(answer: "done", toolCalls: calls),
            tools: [ParallelProbeTool(gate: gate), EchoTool()]
        )

        let (_, events) = await agent.collect([.human("go")])

        // The two parallel-safe calls are not adjacent, so neither can join the other's batch.
        #expect(await gate.peakInFlight == 1)
        #expect(events.toolCompletedResults.map(\.result) == ["read: a", "echo: b", "read: c"])
    }

    @Test func batchesAreCappedAtTheConcurrencyLimit() async {
        // More parallel-safe calls than the cap: the gate is never satisfied, so it releases each
        // batch on its own watchdog and the peak it saw is the cap itself.
        let cap = ReactAgent.maxConcurrentToolCalls
        let gate = ConcurrencyGate(expected: cap + 2)
        let calls = (1 ... (cap + 2)).map { index in
            AgentToolCall(name: "parallel_read", arguments: ["text": .string("f\(index)")])
        }
        let agent = createAgent(
            model: FakeChatModel(answer: "done", toolCalls: calls),
            tools: [ParallelProbeTool(gate: gate)]
        )

        let (ok, events) = await agent.collect([.human("read them all")])

        #expect(ok)
        #expect(await gate.peakInFlight == cap)
        #expect(events.toolCompletedResults.count == cap + 2)
    }

    // MARK: - Order the model and the UI depend on

    @Test func resultsStayInCallOrderWhateverFinishesFirst() async {
        let memory = InMemoryCheckpointer()
        // Slowest first, fastest last: completion order is the reverse of call order.
        let calls = [
            AgentToolCall(name: "parallel_read", arguments: ["text": .string("slow"), "delay_ms": .int(90)]),
            AgentToolCall(name: "parallel_read", arguments: ["text": .string("middle"), "delay_ms": .int(45)]),
            AgentToolCall(name: "parallel_read", arguments: ["text": .string("fast"), "delay_ms": .int(0)])
        ]
        let agent = createAgent(
            model: FakeChatModel(answer: "done", toolCalls: calls),
            tools: [ParallelProbeTool()],
            memory: memory
        )

        let (ok, events) = await agent.collect([.human("go")], threadId: "t")

        #expect(ok)
        // The conversation pairs each call with its result, in the order the model emitted them.
        let saved = await memory.load("t")
        #expect(saved.map(\.role) == [.human, .ai, .tool, .tool, .tool, .ai])
        #expect(saved[1].toolCalls.map(\.id) == calls.map(\.id))
        #expect(saved[2 ... 4].compactMap(\.toolCallID) == calls.map(\.id))
        #expect(saved[2 ... 4].map(\.text) == ["read: slow", "read: middle", "read: fast"])
        // The events do the opposite, and are meant to: each card fills in when its call lands,
        // so the fastest completes first even though its result is appended last.
        #expect(events.toolCompletedResults.map(\.result) == ["read: fast", "read: middle", "read: slow"])
    }

    @Test func everyCardOpensBeforeAnyOfThemFinishes() async {
        let gate = ConcurrencyGate(expected: 3)
        let calls = (1 ... 3).map { index in
            AgentToolCall(name: "parallel_read", arguments: ["text": .string("f\(index)")])
        }
        let agent = createAgent(
            model: FakeChatModel(answer: "done", toolCalls: calls),
            tools: [ParallelProbeTool(gate: gate)]
        )

        let (_, events) = await agent.collect([.human("go")])

        // The batch announces all three calls up front, in call order, so a host shows three
        // running cards rather than three that pop in already finished.
        let lifecycle = events.toolLifecycle
        #expect(lifecycle.map(\.phase) == ["started", "started", "started", "completed", "completed", "completed"])
        #expect(lifecycle.prefix(3).map(\.callID) == calls.map(\.id))
        // Completion order is whatever the batch produces; every one still names its own call.
        #expect(Set(lifecycle.suffix(3).map(\.callID)) == Set(calls.map(\.id)))
    }

    @Test func aSerialCallStillOpensAndClosesItsOwnCard() async {
        let calls = [AgentToolCall(name: "echo", arguments: ["text": .string("x")])]
        let agent = createAgent(
            model: FakeChatModel(answer: "done", toolCalls: calls),
            tools: [EchoTool()]
        )

        let (_, events) = await agent.collect([.human("go")])

        // Hoisting `.toolStarted` out of `dispatchTool` must not change the serial path.
        let lifecycle = events.toolLifecycle
        #expect(lifecycle.map(\.phase) == ["started", "completed"])
        #expect(lifecycle.map(\.callID) == [calls[0].id, calls[0].id])
    }

    @Test func aFailedCallNamesItsCallToo() async {
        let calls = [AgentToolCall(name: "boom", arguments: [:])]
        let agent = createAgent(
            model: FakeChatModel(answer: "done", toolCalls: calls),
            tools: [FailingTool()]
        )

        let (_, events) = await agent.collect([.human("go")])

        let lifecycle = events.toolLifecycle
        #expect(lifecycle.map(\.phase) == ["started", "failed"])
        #expect(lifecycle.map(\.callID) == [calls[0].id, calls[0].id])
    }

    // MARK: - Batch identity, and the events a tool emits itself

    @Test func oneBatchIDIsSharedByItsCallsAndAbsentFromSoloOnes() async {
        let calls = [
            AgentToolCall(name: "parallel_read", arguments: ["text": .string("a")]),
            AgentToolCall(name: "parallel_read", arguments: ["text": .string("b")]),
            AgentToolCall(name: "echo", arguments: ["text": .string("c")])
        ]
        let agent = createAgent(
            model: FakeChatModel(answer: "done", toolCalls: calls),
            tools: [ParallelProbeTool(), EchoTool()]
        )

        let (_, events) = await agent.collect([.human("go")])

        let batches = events.compactMap { event -> UUID?? in
            guard case .toolStarted(_, _, _, let batchID) = event else { return nil }
            return .some(batchID)
        }
        #expect(batches.count == 3)
        #expect(batches[0] != nil)
        #expect(batches[0] == batches[1]) // the two that ran together share one id…
        #expect(batches[2] == .some(nil)) // …and the one that ran alone has none
    }

    /// A tool emitting `.toolProgress` through its `ToolContext` doesn't know its call id, so the
    /// loop stamps it. Without that a host can't route a running tool's output to the right card.
    @Test func progressEmittedByAToolIsStampedWithItsCall() async {
        let calls = (1 ... 2).map { index in
            AgentToolCall(name: "chatty_read", arguments: ["text": .string("f\(index)")])
        }
        let agent = createAgent(
            model: FakeChatModel(answer: "done", toolCalls: calls),
            tools: [ChattyProbeTool()]
        )

        let (_, events) = await agent.collect([.human("go")])

        let progress = events.compactMap { event -> (String, UUID?)? in
            guard case .toolProgress(_, _, let delta, let callID) = event else { return nil }
            return (delta, callID)
        }
        #expect(progress.count == 2)
        // Each call's chunk names that call - the tool passed no id at all.
        #expect(Set(progress.map(\.1)) == Set(calls.map { Optional($0.id) }))
        #expect(progress.contains { $0.0 == "working on f1" && $0.1 == calls[0].id })
        #expect(progress.contains { $0.0 == "working on f2" && $0.1 == calls[1].id })
    }

    // MARK: - State visibility

    @Test func aLaterSerialCallStillSeesAnEarlierCallsStateUpdate() async {
        // Both in ONE round: `set_counter` is not parallel-safe, so `read_counter` runs after it
        // and sees what it wrote - the guarantee `ProbeToolMiddleware` exists to protect.
        let calls = [
            AgentToolCall(name: "set_counter", arguments: [:]),
            AgentToolCall(name: "read_counter", arguments: [:])
        ]
        let agent = createAgent(
            model: FakeChatModel(answer: "done", toolCalls: calls),
            tools: [CounterWriterTool(), CounterReaderTool()]
        )

        let (ok, events) = await agent.collect([.human("go")])

        #expect(ok)
        #expect(events.toolCompletedResults.map(\.result) == ["set", "counter=7"])
    }

    @Test func aParallelCallSeesTheStateFromBeforeItsBatch() async {
        // `set_counter` runs on its own, then the two parallel-safe reads fan out - and both see
        // the value it wrote, because the batch's snapshot is taken after it finished.
        let calls = [
            AgentToolCall(name: "set_counter", arguments: [:]),
            AgentToolCall(name: "parallel_counter", arguments: [:]),
            AgentToolCall(name: "parallel_counter", arguments: [:])
        ]
        let agent = createAgent(
            model: FakeChatModel(answer: "done", toolCalls: calls),
            tools: [CounterWriterTool(), ParallelCounterTool()]
        )

        let (ok, events) = await agent.collect([.human("go")])

        #expect(ok)
        #expect(events.toolCompletedResults.map(\.result) == ["set", "counter=7", "counter=7"])
    }

    // MARK: - The approval gate

    @Test func theApprovalHandlerIsAskedAboutOneCallAtATime() async {
        let spy = ApprovalConcurrencySpy()
        let calls = (1 ... 3).map { index in
            AgentToolCall(name: "parallel_read", arguments: ["text": .string("f\(index)")])
        }
        let agent = createAgent(
            model: FakeChatModel(answer: "done", toolCalls: calls),
            tools: [ParallelProbeTool()],
            middleware: [
                HumanInTheLoopMiddleware(
                    interruptOn: ["parallel_read": InterruptOnConfig()],
                    approvalHandler: spy.handler
                )
            ]
        )

        let (ok, events) = await agent.collect([.human("go")])

        #expect(ok)
        // Neither host can show two approval cards at once, so the handler is asked three times
        // and never twice over - even though the calls were dispatched together.
        #expect(await spy.peakInFlight == 1)
        #expect(await spy.names == ["parallel_read", "parallel_read", "parallel_read"])
        #expect(Set(events.toolCompletedResults.map(\.result)) == ["read: f1", "read: f2", "read: f3"])
    }

    /// The regression this whole design turns on: every read-only tool defaults to `.ask`, so if
    /// being gated disqualified a tool from fanning out, nothing in the shipped tool set ever would
    /// - least of all under a host that answers the gate itself.
    @Test func autoApprovedGatedCallsStillFanOut() async {
        let gate = ConcurrencyGate(expected: 3)
        let calls = (1 ... 3).map { index in
            AgentToolCall(name: "parallel_read", arguments: ["text": .string("f\(index)")])
        }
        let agent = createAgent(
            model: FakeChatModel(answer: "done", toolCalls: calls),
            tools: [ParallelProbeTool(gate: gate)],
            middleware: [
                HumanInTheLoopMiddleware(
                    interruptOn: ["parallel_read": InterruptOnConfig()],
                    approvalHandler: { _ in .approve }
                )
            ]
        )

        let (ok, _) = await agent.collect([.human("go")])

        #expect(ok)
        #expect(await gate.peakInFlight == 3)
    }

    /// The queue holds the *decision*, not the execution. If it wrapped the whole call, gating a
    /// tool would quietly re-serialise the batch - the same mistake as excluding gated tools, one
    /// layer down, and just as invisible.
    @Test func theQueueDoesNotHoldTheGateWhileTheToolRuns() async {
        let gate = ConcurrencyGate(expected: 2)
        let calls = (1 ... 3).map { index in
            AgentToolCall(name: "parallel_read", arguments: ["text": .string("f\(index)")])
        }
        let agent = createAgent(
            model: FakeChatModel(answer: "done", toolCalls: calls),
            tools: [ParallelProbeTool(gate: gate)],
            middleware: [
                HumanInTheLoopMiddleware(
                    interruptOn: ["parallel_read": InterruptOnConfig()],
                    approvalHandler: { _ in
                        // A card a human takes a moment to answer, so the decisions are staggered.
                        try? await Task.sleep(for: .milliseconds(30))
                        return .approve
                    }
                )
            ]
        )

        _ = await agent.collect([.human("go")])

        // An approved call executes while the next card is still up, so two are in flight at once.
        #expect(await gate.peakInFlight >= 2)
    }

    /// A rejected call still short-circuits, and rejecting one of a batch leaves the others alone.
    @Test func rejectingOneCallOfABatchDoesNotTouchTheOthers() async {
        let calls = (1 ... 3).map { index in
            AgentToolCall(name: "parallel_read", arguments: ["text": .string("f\(index)")])
        }
        let rejected = calls[1].id
        let agent = createAgent(
            model: FakeChatModel(answer: "done", toolCalls: calls),
            tools: [ParallelProbeTool()],
            middleware: [
                HumanInTheLoopMiddleware(
                    interruptOn: ["parallel_read": InterruptOnConfig()],
                    approvalHandler: { request in
                        request.id == rejected ? .reject(message: "no") : .approve
                    }
                )
            ]
        )

        let (ok, events) = await agent.collect([.human("go")])

        #expect(ok)
        // A rejection comes back as the call's result (the model is told the user declined), so
        // the two approved calls ran and only the rejected one carries the refusal.
        let results = events.toolCompletedResults.map(\.result)
        #expect(results.filter { $0.hasPrefix("read:") }.sorted() == ["read: f1", "read: f3"])
        #expect(results.filter { $0.contains("rejected") }.count == 1)
        // …and it is the one the handler declined, not whichever finished last.
        let refused = events.compactMap { event -> UUID? in
            guard case .toolCompleted(_, let result, _, _, let id) = event,
                  result.contains("rejected") else { return nil }
            return id
        }
        #expect(refused == [rejected])
    }
}

// MARK: - Test fixtures

private extension [AgentEvent] {
    /// The tool lifecycle events in order, as (phase, call) pairs - what the host is told, and
    /// which card it is told about. Pairing is by `callID`, so these are the two facts that matter.
    var toolLifecycle: [(phase: String, callID: UUID?)] {
        compactMap { event -> (phase: String, callID: UUID?)? in
            switch event {
            case .toolStarted(_, _, let id, _): return ("started", id)
            case .toolCompleted(_, _, _, _, let id): return ("completed", id)
            case .toolFailed(_, _, let id): return ("failed", id)
            default: return nil
            }
        }
    }
}

/// A rendezvous the probe tools park in: every arrival is held until `expected` of them are in
/// flight at once, which is what proves they really do overlap. Serial dispatch would deadlock
/// on it, so a watchdog releases each waiter shortly after it parks - a test then fails on the
/// peak it asserts rather than hanging.
private actor ConcurrencyGate {
    private let expected: Int
    private var inFlight = 0
    private var waiters: [CheckedContinuation<Void, Never>] = []
    private(set) var peakInFlight = 0

    init(expected: Int) { self.expected = expected }

    func arrive() async {
        inFlight += 1
        peakInFlight = max(peakInFlight, inFlight)
        if inFlight >= expected { releaseAll() } else { await park() }
        inFlight -= 1
    }

    private func park() async {
        await withCheckedContinuation { continuation in
            waiters.append(continuation)
            Task { try? await Task.sleep(for: .milliseconds(250)); self.releaseAll() }
        }
    }

    private func releaseAll() {
        let parked = waiters
        waiters = []
        parked.forEach { $0.resume() }
    }
}

/// A parallel-safe stand-in for `read_file`: it waits at the gate (when given one) and can be
/// told to take `delay_ms`, so a test can make completion order differ from call order.
private struct ParallelProbeTool: AgentTool {
    var gate: ConcurrencyGate?
    var name: String { "parallel_read" }
    var description: String { "Read a file (test double)." }
    var isParallelSafe: Bool { true }
    var parameters: [ToolParameter] {
        [
            .required("text", type: .string, description: "What to read."),
            .optional("delay_ms", type: .int, description: "How long to take.")
        ]
    }

    func execute(_ arguments: [String: AgentJSON], _ context: ToolContext) async throws -> ToolOutput {
        guard case .string(let text)? = arguments["text"] else { return ToolOutput("read: <none>") }
        await gate?.arrive()
        if case .int(let delay)? = arguments["delay_ms"], delay > 0 {
            try? await Task.sleep(for: .milliseconds(delay))
        }
        return ToolOutput("read: \(text)")
    }
}

/// A parallel-safe tool that streams progress through its `ToolContext`, carrying no call id of
/// its own - exactly what `task` and `shell` do.
private struct ChattyProbeTool: AgentTool {
    var name: String { "chatty_read" }
    var description: String { "Read something, noisily (test double)." }
    var isParallelSafe: Bool { true }
    var parameters: [ToolParameter] {
        [.required("text", type: .string, description: "What to read.")]
    }

    func execute(_ arguments: [String: AgentJSON], _ context: ToolContext) async throws -> ToolOutput {
        guard case .string(let text)? = arguments["text"] else { return ToolOutput("read: <none>") }
        context.onEvent(.toolProgress(name: name, subagent: nil, delta: "working on \(text)"))
        return ToolOutput("read: \(text)")
    }
}

/// The same tool without the declaration - the control for `serialToolsInOneRoundStillRunOneAtATime`.
private struct SerialProbeTool: AgentTool {
    let gate: ConcurrencyGate
    var name: String { "serial_read" }
    var description: String { "Read a file, serially (test double)." }
    var parameters: [ToolParameter] {
        [.required("text", type: .string, description: "What to read.")]
    }

    func execute(_ arguments: [String: AgentJSON], _ context: ToolContext) async throws -> ToolOutput {
        await gate.arrive()
        return ToolOutput("read")
    }
}

/// Writes a non-"todos" key into agent state (a serial tool - it is what later calls read).
private struct CounterWriterTool: AgentTool {
    var name: String { "set_counter" }
    var description: String { "Set the counter to 7." }

    func execute(_ arguments: [String: AgentJSON], _ context: ToolContext) async throws -> ToolOutput {
        ToolOutput("set", stateUpdate: .set("counter", 7))
    }
}

/// Reads the counter back out of agent state, serially.
private struct CounterReaderTool: AgentTool {
    var name: String { "read_counter" }
    var description: String { "Read the counter from state." }

    func execute(_ arguments: [String: AgentJSON], _ context: ToolContext) async throws -> ToolOutput {
        ToolOutput("counter=\(context.state.values["counter"] as? Int ?? -1)")
    }
}

/// A parallel-safe reader of the agent state, for checking which snapshot a batch is handed.
private struct ParallelCounterTool: AgentTool {
    var name: String { "parallel_counter" }
    var description: String { "Read the counter from state." }
    var isParallelSafe: Bool { true }

    func execute(_ arguments: [String: AgentJSON], _ context: ToolContext) async throws -> ToolOutput {
        ToolOutput("counter=\(context.state.values["counter"] as? Int ?? -1)")
    }
}

/// Records what the approval gate was shown and the most requests it was ever asked to present
/// at once - one card at a time is the invariant a host's UI can actually render.
private actor ApprovalConcurrencySpy {
    private(set) var names: [String] = []
    private(set) var peakInFlight = 0
    private var inFlight = 0

    private func begin(_ request: ToolApprovalRequest) {
        names.append(request.toolName)
        inFlight += 1
        peakInFlight = max(peakInFlight, inFlight)
    }

    private func end() { inFlight -= 1 }

    nonisolated var handler: ToolApprovalHandler {
        { request in
            await self.begin(request)
            // Hold the "card" open long enough that a concurrent second request would overlap it.
            try? await Task.sleep(for: .milliseconds(20))
            await self.end()
            return .approve
        }
    }
}
