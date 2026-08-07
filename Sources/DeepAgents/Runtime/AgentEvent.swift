import Foundation

/// A streamed event from an agent run, consumed by the UI/integration layer. Mirrors
/// the meaningful moments of a ReAct loop: answer tokens, tool activity, plan updates,
/// and terminal status.
public enum AgentEvent: Sendable {
    /// A streamed chunk of the assistant's visible text. `isFinal` is `false` while the
    /// run is still mid-loop (this round's text is interim reasoning that precedes a tool
    /// call) — final-ness is only known at a round's end, so live chunks stream as
    /// `false` and the consumer reclassifies on `roundCompleted`.
    case token(String, isFinal: Bool)
    /// A streamed chunk of the assistant's chain-of-thought reasoning, on its own channel
    /// (separate from the visible answer `token`s). A reasoning model emits these before/around
    /// its answer; the UI shows them in the "thinking…" disclosure.
    case reasoningToken(String)
    /// A ReAct round finished. `hadToolCalls == true` means the text streamed during this
    /// round was interim (tools are about to run); `false` means it was the final answer.
    case roundCompleted(hadToolCalls: Bool)
    /// A tool call is about to run; `input` is a human-readable rendering of its args.
    /// `callID` is the originating ``AgentToolCall/id`` - see the note below.
    case toolStarted(name: String, input: String, callID: UUID? = nil)
    /// A still-running tool streamed incremental output. The `task` tool emits these to stream a
    /// subagent's answer live; `subagent` names which subagent produced the chunk (nil for other
    /// tools). The first one of a run may carry an empty `delta` just to attach the subagent label.
    case toolProgress(name: String, subagent: String?, delta: String, callID: UUID? = nil)
    /// A tool call finished; `result` is what the model sees next. `imageURL`, when present,
    /// is an image the tool produced (e.g. a screenshot) for the UI to show as a thumbnail.
    /// `editDiff`, when present, is a line diff an `edit_file` produced for the UI to render
    /// (the model still sees only the short `result` text).
    case toolCompleted(
        name: String, result: String, imageURL: URL? = nil, editDiff: FileDiff? = nil, callID: UUID? = nil
    )
    /// A tool call failed; the model receives the error and may recover.
    case toolFailed(name: String, error: String, callID: UUID? = nil)
    /// The to-do middleware updated the plan.
    case todosUpdated([TodoItem])
    /// Summarization compacted the conversation this round: the older messages were replaced by a
    /// summary turn. `tokensBefore`/`tokensAfter` are the estimated history sizes around the rewrite,
    /// so the UI can note the drop and refresh its context-usage meter.
    case contextCompacted(tokensBefore: Int, tokensAfter: Int)
    /// The run finished successfully.
    case completed
    /// The run failed.
    case failed(String)
}

// MARK: - Pairing tool events

/// Every tool event carries the `callID` of the call it belongs to. **Pair on it, not on `name`.**
///
/// A round's calls used to run strictly one at a time, so "the most recent unfinished step with
/// this name" identified a call unambiguously. Parallel dispatch ends that: three `read_file`
/// calls can be open at once, and their completions arrive in whatever order they finish. Matching
/// by name would attach the first result to whichever card the host happened to open last.
///
/// `callID` is `nil` only where no call exists to point at - an event a host synthesizes, or a
/// step rebuilt from a persisted transcript. Fall back to name-matching there.
