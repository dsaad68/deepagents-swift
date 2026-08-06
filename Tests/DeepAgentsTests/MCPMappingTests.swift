@testable import DeepAgents
@testable import DeepAgentsMacTools
@testable import DeepAgentsMLX
import Foundation
import Testing

// Tests for attributing namespaced MCP tools (`server__tool`) back to their server: the dispatch
// prefix, the per-server approval-default map, and the display projection that the MCP Servers tab
// uses to reflect the agent's live (warm) tool set.

@Suite("MCP tool ↔ server mapping")
struct MCPMappingTests {
    @Test("dispatchPrefix sanitizes the server name")
    func dispatchPrefix() {
        #expect(MCPTool.dispatchPrefix(forServer: "parallel-search") == "parallel_search__")
        #expect(MCPTool.dispatchPrefix(forServer: "deepwiki") == "deepwiki__")
        #expect(MCPTool.dispatchPrefix(forServer: "my server") == "my_server__") // space → underscore
    }

    @Test("dispatchName exposes no hyphen — models normalize one away when emitting a call")
    func dispatchNameFoldsHyphens() {
        #expect(MCPTool.dispatchName(server: "parallel-search", tool: "web_search")
            == "parallel_search__web_search")
        #expect(MCPTool.dispatchName(server: "deepwiki", tool: "read-wiki-contents")
            == "deepwiki__read_wiki_contents")
    }

    @Test("mcpApprovalDefaults maps each tool to its server's mode; unmatched tools get none")
    func approvalDefaults() {
        let servers = [
            MCPServerConfig(name: "parallel-search", kind: .http, approvalMode: .ask),
            MCPServerConfig(name: "deepwiki", kind: .http, approvalMode: .approve)
        ]
        let tools: [any AgentTool] = [
            OwnedTool(server: "parallel-search", tool: "web_search"),
            OwnedTool(server: "parallel-search", tool: "web_fetch"),
            OwnedTool(server: "deepwiki", tool: "ask"),
            OwnedTool(server: "other", tool: "thing"), // belongs to no configured server
            StubTool("parallel_search__impostor") // names no server; governed by the catalog
        ]
        let defaults = mcpApprovalDefaults(servers: servers, tools: tools)

        #expect(defaults["parallel_search__web_search"] == .ask)
        #expect(defaults["parallel_search__web_fetch"] == .ask)
        #expect(defaults["deepwiki__ask"] == .approve)
        #expect(defaults["other__thing"] == nil)
        #expect(defaults["parallel_search__impostor"] == nil)
    }

    /// Server names that a dispatch prefix cannot tell apart: sanitizing maps each pair onto one
    /// prefix, so anything attributing by prefix mixes their tools up.
    static let collidingNames = [
        ("parallel-search", "parallel_search"), // hyphen vs underscore
        ("my server", "my_server"), // space vs underscore
        ("a.b", "a-b") // punctuation of any kind folds the same way
    ]

    @Test("tools never cross between servers whose names sanitize alike", arguments: collidingNames)
    func attributionSurvivesACollidingPrefix(first: String, second: String) {
        #expect(
            MCPTool.dispatchPrefix(forServer: first) == MCPTool.dispatchPrefix(forServer: second),
            "the case only means something while the two prefixes are identical"
        )
        let servers = [
            MCPServerConfig(name: first, kind: .http, approvalMode: .deny),
            MCPServerConfig(name: second, kind: .http, approvalMode: .approve)
        ]
        let denied = OwnedTool(server: first, tool: "run")
        let permitted = OwnedTool(server: second, tool: "run", dispatchName: "collision_broken_by_the_loader")
        let tools: [any AgentTool] = [denied, permitted]

        // Each tool keeps its own server's mode - the denied one is never relaxed by its neighbour.
        let defaults = mcpApprovalDefaults(servers: servers, tools: tools)
        #expect(defaults[denied.name] == .deny)
        #expect(defaults[permitted.name] == .approve)

        // And the same attribution drives display, so neither server lists the other's tools.
        #expect(mcpToolsForDisplay(server: first, in: tools).map(\.dispatchName) == [denied.name])
        #expect(mcpToolsForDisplay(server: second, in: tools).map(\.dispatchName) == [permitted.name])
    }

    @Test("a tool renamed to break a collision still displays under its name on the server")
    func displayNameComesFromTheServerNotTheDispatchName() {
        let renamed = OwnedTool(server: "srv", tool: "search", dispatchName: "srv__search_2")
        let display = mcpToolsForDisplay(server: "srv", in: [renamed])
        #expect(display.map(\.name) == ["search"])
        #expect(display.map(\.dispatchName) == ["srv__search_2"])
    }

    @Test("mcpToolsForDisplay attributes tools to a server, strips the prefix, keeps the schema")
    func displayProjection() {
        let tools: [any AgentTool] = [
            SchemaTool(server: "parallel-search", tool: "web_search", description: "Search the web"),
            SchemaTool(server: "parallel-search", tool: "web_fetch", description: "Fetch a URL"),
            SchemaTool(server: "deepwiki", tool: "ask", description: "Ask the wiki")
        ]

        let parallel = mcpToolsForDisplay(server: "parallel-search", in: tools)
        #expect(parallel.map(\.name).sorted() == ["web_fetch", "web_search"])
        let search = parallel.first { $0.name == "web_search" }
        #expect(search?.dispatchName == "parallel_search__web_search")
        #expect(search?.description == "Search the web")
        #expect(search?.schema["type"] as? String == "object")

        let deepwiki = mcpToolsForDisplay(server: "deepwiki", in: tools)
        #expect(deepwiki.map(\.dispatchName) == ["deepwiki__ask"])
    }

    @Test("mcpToolsForDisplay returns nothing for a server with no matching tools")
    func displayProjectionEmpty() {
        let tools: [any AgentTool] = [SchemaTool(server: "deepwiki", tool: "ask", description: "Ask")]
        #expect(mcpToolsForDisplay(server: "parallel-search", in: tools).isEmpty)
    }
}

/// A `ServerScopedTool` stub: the tool declares the server that contributed it, exactly as any
/// real MCP tool does, so attribution can be exercised without an MCP session. `dispatchName`
/// stands in for the loader's collision rename.
private struct OwnedTool: ServerScopedTool {
    let serverName: String
    let toolName: String
    private let dispatchName: String?

    init(server: String, tool: String, dispatchName: String? = nil) {
        serverName = server
        toolName = tool
        self.dispatchName = dispatchName
    }

    var name: String { dispatchName ?? MCPTool.dispatchName(server: serverName, tool: toolName) }
    var description: String { "stub" }
    var parameters: [ToolParameter] { [] }
    func execute(_ arguments: [String: AgentJSON], _ context: ToolContext) async throws -> ToolOutput {
        ToolOutput("ok")
    }
}

/// A minimal `AgentTool` with a chosen name (default schema).
private struct StubTool: AgentTool {
    let name: String
    init(_ name: String) { self.name = name }
    var description: String { "stub" }
    var parameters: [ToolParameter] { [] }
    func execute(_ arguments: [String: AgentJSON], _ context: ToolContext) async throws -> ToolOutput {
        ToolOutput("ok")
    }
}

/// An `AgentTool` that injects a known `toolSchema()` schema (the way ``MCPTool`` does), so the
/// display projection's schema extraction can be asserted.
private struct SchemaTool: ServerScopedTool {
    let serverName: String
    let toolName: String
    let desc: String
    init(server: String, tool: String, description: String) {
        serverName = server
        toolName = tool
        desc = description
    }

    var name: String { MCPTool.dispatchName(server: serverName, tool: toolName) }
    var description: String { desc }
    var parameters: [ToolParameter] { [] }
    func execute(_ arguments: [String: AgentJSON], _ context: ToolContext) async throws -> ToolOutput {
        ToolOutput("ok")
    }

    func toolSchema() -> ToolSchema {
        let schema: [String: any Sendable] = ["type": "object", "properties": [String: any Sendable]()]
        return [
            "type": "function",
            "function": ["name": name, "description": desc, "parameters": schema] as [String: any Sendable]
        ]
    }
}
