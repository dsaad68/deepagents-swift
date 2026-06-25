# Memory & checkpointing

By default a `ReactAgent` run is stateless: it starts from the messages you pass in and discards everything when it returns. To give an agent **short-term memory across runs** - so a follow-up message picks up where the previous one left off - you supply a checkpointer.

## The `AgentCheckpointer` protocol

```swift
public protocol AgentCheckpointer: Sendable {
    func load(_ threadId: String) async -> [AgentMessage]
    func save(_ threadId: String, _ messages: [AgentMessage]) async
}
```

The contract is intentionally minimal:

- `load(_:)` returns the stored conversation for a thread, or an empty array on a cache miss.
- `save(_:_:)` persists the full updated conversation after every run.

Both methods are `async`, so any backing store - in-memory, disk, SQLite, a remote key-value store - fits the protocol without blocking the agent loop.

All `AgentMessage` types are `Sendable` and `Codable`, which makes serialization straightforward for disk-based implementations.

## Built-in: `InMemoryCheckpointer`

```swift
public actor InMemoryCheckpointer: AgentCheckpointer {
    public init()
    public func load(_ threadId: String) async -> [AgentMessage]
    public func save(_ threadId: String, _ messages: [AgentMessage]) async
    public func clear(_ threadId: String) async
}
```

`InMemoryCheckpointer` is the simplest implementation: it stores thread histories in a Swift `actor`, so reads and writes are safe under concurrent use. Thread histories live for the process lifetime. The `clear(_:)` method lets the host discard a thread when the user starts a fresh conversation.

!!! note "Process lifetime"
    Because `InMemoryCheckpointer` holds messages in RAM, a process restart loses all state. For durable storage across restarts, implement the protocol yourself (see below).

## Wiring a checkpointer

Pass the checkpointer as the `memory:` parameter to either factory:

```swift
import DeepAgents

let memory = InMemoryCheckpointer()

let agent = createDeepAgent(
    model: model,
    memory: memory,
    systemPrompt: "You are a helpful assistant."
)
```

The same pattern works with `createAgent`.

## Thread identity: `threadId`

The checkpointer keys every conversation by a **thread ID** string you supply at run time:

```swift
let ok = await agent.run(
    [.human("What files are in my project?")],
    threadId: "user-session-42"
) { event in
    // handle streamed events
}
```

When `threadId` is non-nil, the agent:

1. Calls `memory.load(threadId)` to retrieve prior messages.
2. Prepends them to the current input to reconstruct the full conversation.
3. Calls `memory.save(threadId, updatedMessages)` after the run, storing the appended history.

When `threadId` is `nil`, the checkpointer is bypassed and the run is stateless regardless of whether one was supplied to the factory.

!!! tip "Thread granularity"
    Use one thread ID per conversation session - not per user or per agent. Multiple agents can share the same `InMemoryCheckpointer` instance safely (it is an `actor`), as long as each conversation has a distinct ID.

## How memory interacts with the agent loop

The reconstructed history is merged with the current input before the first model call. Middleware runs against the full combined history, which means [summarization](summarization.md) can compact prior turns, and [human-in-the-loop](human-in-the-loop.md) can gate tools that appeared in earlier rounds. See [the agent loop](agent-loop.md) for the full round-trip.

## Implementing your own checkpointer

Conform to `AgentCheckpointer` to persist conversations in any backing store:

=== "Disk (JSON)"

    ```swift
    import Foundation
    import DeepAgents

    public actor DiskCheckpointer: AgentCheckpointer {
        let directory: URL

        public init(directory: URL) throws {
            self.directory = directory
            try FileManager.default.createDirectory(
                at: directory, withIntermediateDirectories: true
            )
        }

        public func load(_ threadId: String) async -> [AgentMessage] {
            let url = file(for: threadId)
            guard let data = try? Data(contentsOf: url) else { return [] }
            return (try? JSONDecoder().decode([AgentMessage].self, from: data)) ?? []
        }

        public func save(_ threadId: String, _ messages: [AgentMessage]) async {
            guard let data = try? JSONEncoder().encode(messages) else { return }
            try? data.write(to: file(for: threadId), options: .atomic)
        }

        private func file(for threadId: String) -> URL {
            directory.appendingPathComponent("\(threadId).json")
        }
    }
    ```

=== "Conceptual (SQLite / cloud)"

    ```swift
    // The protocol has no restrictions on async work - you can call
    // any async database or networking API inside load/save.
    public actor MyRemoteCheckpointer: AgentCheckpointer {
        public func load(_ threadId: String) async -> [AgentMessage] {
            let rows = await myDB.fetchMessages(thread: threadId)
            return rows.map { AgentMessage(from: $0) }
        }

        public func save(_ threadId: String, _ messages: [AgentMessage]) async {
            await myDB.upsertMessages(thread: threadId, messages: messages)
        }
    }
    ```

!!! warning "Codability"
    `AgentMessage` is `Codable`. Custom `AgentContentBlock` cases are sealed enums - no extensions needed. Ensure your serialization round-trips the full message array faithfully; a truncated or reordered history will confuse the model.

## What is persisted

The checkpointer stores the raw `[AgentMessage]` array - every role, every content block, every tool call and result, including images (by URL or base64). [SummarizationMiddleware](summarization.md) may rewrite this array before it is saved, replacing evicted older messages with a compact summary turn. The checkpointer itself is unaware of summarization; it simply stores whatever the agent hands it.

## Summary

| Concept | Details |
|---|---|
| Protocol | `AgentCheckpointer` - two async methods, `load` and `save` |
| Built-in | `InMemoryCheckpointer` (actor; process lifetime) |
| Factory param | `memory:` on both `createAgent` and `createDeepAgent` |
| Thread key | `threadId: String?` passed to `agent.run(...)` |
| Custom stores | Conform to `AgentCheckpointer`; disk/SQLite/cloud all work |
