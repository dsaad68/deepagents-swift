import Foundation
import MCP

enum MCPToolError: LocalizedError {
    case toolFailed(name: String, message: String)

    var errorDescription: String? {
        switch self {
        case .toolFailed(let name, let message):
            return "MCP tool '\(name)' failed: \(message)"
        }
    }
}

/// A tool contributed by a named server, which therefore has an owner: the server whose
/// configuration governs it (its approval mode) and under whose heading a UI lists it.
///
/// Attribution goes through this protocol and never through the tool's dispatch name. That name
/// is sanitized, so distinct server names collapse onto one prefix - `parallel-search` and
/// `parallel_search` both yield `parallel_search__` - and matching on it cannot tell two servers'
/// tools apart. The consequences are not cosmetic: one server's tools inherit another's approval
/// mode, so a "Deny" server's tool can execute as "Approve".
public protocol ServerScopedTool: AgentTool {
    /// The contributing server, exactly as configured (unsanitized).
    var serverName: String { get }
    /// The tool's own name on that server, without the namespace prefix.
    var toolName: String { get }
}

/// The tools in `tools` contributed by `serverName` - the one way to attribute a tool to a server.
public func toolsFromServer(_ serverName: String, in tools: [any AgentTool]) -> [any ServerScopedTool] {
    tools.compactMap { $0 as? any ServerScopedTool }.filter { $0.serverName == serverName }
}

/// An `AgentTool` that proxies to a tool on an MCP server — Mispher's analogue of
/// `langchain-mcp-adapters`' `convert_mcp_tool_to_langchain_tool`.
///
/// The exposed `name` is namespaced `server__tool` so tools from different servers never
/// collide; the original (unprefixed) `toolName` is what we send back to the server. The
/// server's own JSON Schema is injected verbatim by overriding ``toolSchema()`` — which is
/// why `AgentTool` declares `toolSchema()` as a requirement, so this override is honored
/// through the `any AgentTool` existential the model layer iterates.
public struct MCPTool: ServerScopedTool {
    /// Logical server name, used as the tool-name prefix.
    public let serverName: String
    /// The tool's name on the server (what `callTool` is invoked with).
    public let toolName: String
    let toolDescription: String
    /// The tool's JSON Schema, as advertised by the server.
    let inputSchema: Value
    let session: any MCPSession
    /// An explicit dispatch name set by the loader to break a collision (e.g. two servers
    /// that sanitize to the same prefix). When `nil`, the derived namespaced name is used.
    var nameOverride: String?

    /// The namespaced name the model and `ReactAgent` dispatch on. Both components are
    /// sanitized to `[A-Za-z0-9_]` because a user-chosen server name (or an unusual
    /// server-side tool name) containing spaces/punctuation may not round-trip through the
    /// chat template and tool-call parser, which would make `ReactAgent`'s exact-match
    /// dispatch fail with "unknown tool". The server is still invoked with the original
    /// `toolName` (see `execute`), so sanitizing the exposed name is safe.
    public var name: String { nameOverride ?? Self.dispatchName(server: serverName, tool: toolName) }
    public var description: String { toolDescription }

    /// The default namespaced dispatch name for a server/tool pair.
    public static func dispatchName(server: String, tool: String) -> String {
        "\(sanitize(server))__\(sanitize(tool))"
    }

    /// The dispatch-name prefix (`sanitize(server)__`) every tool a server contributes shares.
    ///
    /// For building or displaying a name only. Do **not** attribute a tool to a server with it:
    /// sanitizing maps distinct server names onto one prefix, so the match is ambiguous. Use
    /// ``toolsFromServer(_:in:)``, which reads the server off the tool.
    public static func dispatchPrefix(forServer server: String) -> String {
        "\(sanitize(server))__"
    }

    /// A server or tool name in the canonical spelling - see ``ToolName/normalized(_:)``, which is
    /// the same rule applied to the names the model sends back. The server is still invoked with
    /// the original `toolName` (see `execute`), so normalizing the exposed name is safe.
    static func sanitize(_ component: String) -> String { ToolName.normalized(component) }

    /// The first `required` property the call leaves out, phrased so the model can correct it, or
    /// nil when nothing required is missing.
    ///
    /// Deliberately only this one check. Full JSON Schema validation here would reject calls the
    /// server would have accepted - `anyOf`, loose types, server-side coercion - and a tool that
    /// refuses valid work is worse than one that forwards a bad call. A named `required` property
    /// that simply isn't there is unambiguous.
    ///
    /// When the call carries a near-miss of the missing name (`url` for `urls`), it is quoted back:
    /// that is the mistake actually observed, a 2.6B reading `urls: array` as `url: string` and
    /// then repeating it on the next URL.
    static func missingRequiredArgument(
        _ arguments: [String: AgentJSON], schema: Value
    ) -> String? {
        guard let object = schema.objectValue,
              let required = object["required"]?.arrayValue?.compactMap(\.stringValue),
              !required.isEmpty else { return nil }
        let properties = object["properties"]?.objectValue ?? [:]
        guard let missing = required.first(where: { arguments[$0] == nil }) else { return nil }

        let type = properties[missing]?.objectValue?["type"]?.stringValue
        let described = type.map { "`\(missing)` (\($0))" } ?? "`\(missing)`"
        var complaint = "Missing required argument \(described)."
        // Sorted, not `keys.first`: dictionary order is unstable, and a message that changes
        // between identical runs is one a caller cannot rely on or a test pin.
        if let nearMiss = arguments.keys.sorted().first(where: { Self.isNearMiss($0, of: missing) }) {
            complaint += " You passed `\(nearMiss)`, which this tool does not take."
        }
        if type == "array" { complaint += " It takes an array, so pass a list even for a single value." }
        return complaint + " Required: \(required.map { "`\($0)`" }.joined(separator: ", "))."
    }

    /// Whether `supplied` looks like an attempt at `expected`: one a prefix of the other (the
    /// singular/plural slip) or the same name in a different case. Both without pulling in an
    /// edit-distance implementation.
    static func isNearMiss(_ supplied: String, of expected: String) -> Bool {
        guard supplied != expected else { return false }
        let a = supplied.lowercased(), b = expected.lowercased()
        // Case-only differences count: `URLs` for `urls` leaves the argument genuinely missing,
        // and it is the one mistake a caller is least likely to spot unaided.
        return a == b || a.hasPrefix(b) || b.hasPrefix(a)
    }

    /// Inject the server-provided schema directly, rather than rebuilding it from
    /// `parameters` (which an MCP tool doesn't have) — so nested objects/arrays and every
    /// constraint the server declares survive into the chat template.
    public func toolSchema() -> ToolSchema {
        [
            "type": "function",
            "function": [
                "name": name,
                "description": description,
                "parameters": MCPValueBridge.schemaObject(inputSchema)
            ] as [String: any Sendable]
        ]
    }

    public func execute(
        _ arguments: [String: AgentJSON], _ context: ToolContext
    ) async throws -> ToolOutput {
        // Check the call against the server's own schema first. Left to the server, a mistake comes
        // back in whatever dialect it speaks - a Pydantic dump naming `v1_extract_toolArguments`
        // says nothing a model can act on, and a small model repeats the same call verbatim. Named
        // in the tool's own vocabulary, it can fix it on the next round.
        if let complaint = Self.missingRequiredArgument(arguments, schema: inputSchema) {
            throw MCPToolError.toolFailed(name: name, message: complaint)
        }
        let (content, isError) = try await session.callTool(
            name: toolName, arguments: MCPValueBridge.toMCPArguments(arguments)
        )
        let text = MCPValueBridge.text(from: content)
        // An MCP error result is recoverable: thrown here, `ReactAgent.dispatchTool` turns
        // it into an `Error: …` tool message so the model can react rather than aborting.
        if isError == true {
            throw MCPToolError.toolFailed(name: name, message: text)
        }
        return ToolOutput(text)
    }
}
