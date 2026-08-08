import Foundation

/// The base system prompt for a deep agent — the opinionated guidance deepagents prepends to teach
/// the model to plan, delegate, use its filesystem, and verify. `createDeepAgent` composes this
/// with the caller's own system prompt; the planning / filesystem / subagent middleware each add
/// their own tool-specific notes on top.
///
/// Deliberately short and free of caveats: the small LFM models follow a handful of unambiguous
/// rules far better than long prose, and a "skip planning for simple tasks" escape hatch here
/// would contradict agents (like the deep screen agent) that mandate planning — when/whether to
/// plan is the concrete agent's call, stated in its own prompt.
public enum DeepAgentPrompt {
    /// The base prompt, naming only the pillars actually registered: the planning and
    /// subagent pillars are always in the deep-agent stack, but the filesystem one is
    /// optional (`includeFilesystem`) — mentioning `write_file` without the tool would
    /// have the model calling a tool that doesn't exist.
    /// - Parameter workspaceRoot: the folder the file and search tools are rooted at. Stating it is
    ///   not a nicety: with nothing to go on a model invents one, and every invented path is a
    ///   refused call and a wasted round. Measured across 47 on-device runs, 30 of them contained
    ///   `outside the allowed folder` errors - 148 refused calls in total - including a model on
    ///   macOS guessing the Linux path `/home/user`, and one guessing the project's name rather
    ///   than the checkout it was actually running in.
    static func system(includeFilesystem: Bool = true, workspaceRoot: URL? = nil) -> String {
        let filesystemBullet = includeFilesystem
            ? """
            - Use your filesystem for working state. Save notes, drafts, and intermediate results \
            with `write_file` instead of carrying them in the conversation.

            """
            : ""
        let locationBullet = workspaceRoot.map {
            """
            - Your working folder is `\($0.path)`. Paths you pass to tools are resolved inside it, \
            and anything outside it is refused - so prefer relative paths, and never guess an \
            absolute path from a project's name.

            """
        } ?? ""
        return """
        You are a deep agent: you tackle complex, multi-step tasks methodically rather than \
        answering off the cuff.

        - Plan first. For anything non-trivial, use `write_todos` to lay out the steps, then keep \
        the list current as you go.
        - Delegate isolated subtasks. Use the `task` tool to hand a self-contained piece of work \
        to a subagent; this keeps your own context focused. Give the subagent everything it \
        needs — it can't ask follow-ups.
        - Ask for independent lookups together. When you need to read a few files, or search for \
        several things, request them all in one message instead of one at a time.
        \(locationBullet)\(filesystemBullet)- Verify before finishing. Check that you actually did what was asked, then give a clear, \
        complete final answer.
        """
    }
}
