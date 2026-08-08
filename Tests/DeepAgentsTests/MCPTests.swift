@testable import DeepAgents
@testable import DeepAgentsMacTools
@testable import DeepAgentsMLX
import Foundation
import MCP
import MLXLMCommon
import Testing

// Tests for the MCP client layer. They exercise the value/schema/result conversions and
// the tool-loading logic without a real server, using a `FakeMCPSession` — the same
// dependency-injection approach the agent tests use with `FakeChatModel`.

/// An in-memory `MCPSession`: returns scripted tools and a scripted `callTool` result, and
/// records the last call so tests can assert argument conversion and unprefixed names.
actor FakeMCPSession: MCPSession {
    let scriptedTools: [MCP.Tool]
    let result: (content: [MCP.Tool.Content], isError: Bool?)
    let failConnect: Bool
    private(set) var lastCall: (name: String, arguments: [String: Value]?)?

    init(
        tools: [MCP.Tool],
        result: (content: [MCP.Tool.Content], isError: Bool?) = (
            [.text(text: "ok", annotations: nil, _meta: nil)], nil
        ),
        failConnect: Bool = false
    ) {
        scriptedTools = tools
        self.result = result
        self.failConnect = failConnect
    }

    struct ConnectFailed: Error {}

    func connect() async throws { if failConnect { throw ConnectFailed() } }
    func listTools() async throws -> [MCP.Tool] { scriptedTools }
    func callTool(
        name: String, arguments: [String: Value]?
    ) async throws -> (content: [MCP.Tool.Content], isError: Bool?) {
        lastCall = (name, arguments)
        return result
    }

    func disconnect() async {}
}

private let objectSchema: Value = [
    "type": "object",
    "properties": ["q": ["type": "string", "description": "query"]],
    "required": ["q"]
]

// MARK: - Value bridge

struct MCPValueBridgeTests {
    @Test func convertsEachJSONValueCase() {
        #expect(MCPValueBridge.toMCPValue(.string("a")) == .string("a"))
        #expect(MCPValueBridge.toMCPValue(.int(7)) == .int(7))
        #expect(MCPValueBridge.toMCPValue(.double(1.5)) == .double(1.5))
        #expect(MCPValueBridge.toMCPValue(.bool(true)) == .bool(true))
        #expect(MCPValueBridge.toMCPValue(.null) == .null)
        #expect(MCPValueBridge.toMCPValue(.array([.int(1), .int(2)])) == .array([.int(1), .int(2)]))
        #expect(
            MCPValueBridge.toMCPValue(.object(["k": .string("v")])) == .object(["k": .string("v")])
        )
    }

    @Test func emptyArgumentsBecomeNil() {
        #expect(MCPValueBridge.toMCPArguments([:]) == nil)
        #expect(MCPValueBridge.toMCPArguments(["q": .string("hi")]) == ["q": .string("hi")])
    }

    @Test func schemaObjectPreservesNestedStructure() {
        let object = MCPValueBridge.schemaObject(objectSchema)
        #expect(object["type"] as? String == "object")
        let properties = object["properties"] as? [String: any Sendable]
        let q = properties?["q"] as? [String: any Sendable]
        #expect(q?["type"] as? String == "string")
        #expect((object["required"] as? [any Sendable])?.first as? String == "q")
    }

    @Test func nonObjectSchemaFallsBack() {
        let object = MCPValueBridge.schemaObject(.string("oops"))
        #expect(object["type"] as? String == "object")
    }

    @Test func flattensContentBlocksToText() {
        let content: [MCP.Tool.Content] = [
            .text(text: "hello", annotations: nil, _meta: nil),
            .image(data: "AAAA", mimeType: "image/png", annotations: nil, _meta: nil)
        ]
        #expect(MCPValueBridge.text(from: content) == "hello\n[image: image/png]")
    }
}

// MARK: - MCPTool

struct MCPToolTests {
    private func tool(
        result: (content: [MCP.Tool.Content], isError: Bool?) = (
            [.text(text: "ok", annotations: nil, _meta: nil)], nil
        )
    ) -> (MCPTool, FakeMCPSession) {
        let session = FakeMCPSession(
            tools: [MCP.Tool(name: "search", description: "Search", inputSchema: objectSchema)],
            result: result
        )
        let tool = MCPTool(
            serverName: "srv", toolName: "search", toolDescription: "Search",
            inputSchema: objectSchema, session: session
        )
        return (tool, session)
    }

    @Test func namespacesNameWithServer() {
        let (tool, _) = tool()
        #expect(tool.name == "srv__search")
    }

    @Test func sanitizesNameComponentsForDispatch() {
        let session = FakeMCPSession(
            tools: [MCP.Tool(name: "do thing!", description: "", inputSchema: ["type": "object"])]
        )
        let tool = MCPTool(
            serverName: "my server", toolName: "do thing!", toolDescription: "",
            inputSchema: ["type": "object"], session: session
        )
        #expect(tool.name == "my_server__do_thing_")
    }

    @Test func toolSpecInjectsServerSchemaVerbatim() {
        let (tool, _) = tool()
        let spec = tool.toolSchema()
        let function = spec["function"] as? [String: any Sendable]
        #expect(function?["name"] as? String == "srv__search")
        #expect(function?["description"] as? String == "Search")
        let parameters = function?["parameters"] as? [String: any Sendable]
        #expect(parameters?["type"] as? String == "object")
        let properties = parameters?["properties"] as? [String: any Sendable]
        #expect((properties?["q"] as? [String: any Sendable])?["type"] as? String == "string")
    }

    @Test func executeCallsUnprefixedNameWithConvertedArgs() async throws {
        let (tool, session) = tool()
        let output = try await tool.execute(["q": .string("hi")], ToolContext())
        #expect(output.content == "ok")
        let last = await session.lastCall
        #expect(last?.name == "search")
        #expect(last?.arguments == ["q": .string("hi")])
    }

    @Test func executeThrowsOnIsError() async {
        let (tool, _) = tool(
            result: ([.text(text: "bad input", annotations: nil, _meta: nil)], true)
        )
        await #expect(throws: MCPToolError.self) {
            _ = try await tool.execute(["q": .string("hi")], ToolContext())
        }
    }

    // MARK: - Checking a call against the server's schema

    // Forwarded blind, a wrong argument comes back in the server's own dialect. Observed with a
    // 2.6B on a `web_fetch` whose schema requires `urls` (array): it sent `url`, got back
    // `1 validation error for v1_extract_toolArguments\nurls\n Field required [type=missing…]`,
    // and made the identical call again on the next URL. Nothing in that names the mistake.

    /// A tool whose one required argument is an array, mirroring the `web_fetch` case.
    private func arrayArgTool() -> (MCPTool, FakeMCPSession) {
        let schema: Value = [
            "type": "object",
            "properties": ["urls": ["type": "array", "description": "URLs to fetch"]],
            "required": ["urls"]
        ]
        let session = FakeMCPSession(
            tools: [MCP.Tool(name: "web_fetch", description: "Fetch", inputSchema: schema)]
        )
        return (
            MCPTool(
                serverName: "srv", toolName: "web_fetch", toolDescription: "Fetch",
                inputSchema: schema, session: session
            ),
            session
        )
    }

    @Test func aMissingRequiredArgumentIsNamedAndNeverReachesTheServer() async throws {
        let (tool, session) = arrayArgTool()
        await #expect(throws: MCPToolError.self) {
            _ = try await tool.execute(["url": .string("https://example.com")], ToolContext())
        }
        // Blocked locally: a call the server would only reject costs a round trip either way.
        #expect(await session.lastCall == nil)
    }

    @Test func theComplaintNamesTheArgumentTheTypeAndTheNearMiss() {
        let schema: Value = [
            "type": "object",
            "properties": ["urls": ["type": "array"]],
            "required": ["urls"]
        ]
        let complaint = MCPTool.missingRequiredArgument(
            ["url": .string("https://example.com")], schema: schema
        )
        let text = try? #require(complaint)
        #expect(text?.contains("`urls`") == true)
        #expect(text?.contains("array") == true)
        #expect(text?.contains("`url`") == true) // the near miss it actually sent
    }

    /// The guard that matters most: a valid call must still go through. Rejecting work the server
    /// would have accepted is worse than forwarding a bad call, which is why this checks only
    /// `required` and never types, `anyOf`, or unknown extras.
    @Test func aValidCallIsStillDispatched() async throws {
        let (tool, session) = arrayArgTool()
        _ = try await tool.execute(["urls": .array([.string("https://example.com")])], ToolContext())
        #expect(await session.lastCall?.name == "web_fetch")
    }

    @Test func extraArgumentsAndLooseSchemasAreNeverBlocked() async throws {
        let (tool, session) = tool()
        // An unknown extra alongside everything required: the server may well accept it.
        _ = try await tool.execute(["q": .string("hi"), "extra": .bool(true)], ToolContext())
        #expect(await session.lastCall != nil)

        // A schema that declares nothing required blocks nothing.
        #expect(MCPTool.missingRequiredArgument([:], schema: ["type": "object"]) == nil)
    }
}

/// Everything the schema check must **not** do, plus the shapes of what it does.
///
/// The risk this carries is not that it misses a bad call - that costs one round trip, which is
/// what happened before it existed. The risk is that it refuses a call the server would have
/// accepted, which costs the user a capability with no way to override it. So most of this suite
/// is about calls that have to get through.
struct MCPArgumentCheckTests {
    private func schema(
        properties: [String: Value] = ["urls": ["type": "array"]], required: Value = ["urls"]
    ) -> Value {
        ["type": "object", "properties": .object(properties), "required": required]
    }

    // MARK: - Never block a call the server might accept

    @Test func aSatisfiedCallPasses() {
        #expect(MCPTool.missingRequiredArgument(["urls": .array([.string("u")])], schema: schema()) == nil)
    }

    @Test func unknownExtrasPass() {
        let arguments: [String: AgentJSON] = ["urls": .array([]), "depth": .int(2), "raw": .bool(true)]
        #expect(MCPTool.missingRequiredArgument(arguments, schema: schema()) == nil)
    }

    /// The declared type is never enforced. A server may coerce a string to an array, or accept a
    /// union the schema flattens - refusing here would be us overruling the server about its own
    /// contract.
    @Test func aRequiredArgumentOfTheWrongTypeStillPasses() {
        #expect(MCPTool.missingRequiredArgument(["urls": .string("u")], schema: schema()) == nil)
        #expect(MCPTool.missingRequiredArgument(["urls": .int(1)], schema: schema()) == nil)
        #expect(MCPTool.missingRequiredArgument(["urls": .object([:])], schema: schema()) == nil)
    }

    /// An explicit null is a value the caller chose to send. Some servers treat it as "unset" and
    /// some as a real argument; that is theirs to decide, not ours.
    @Test func anExplicitNullPasses() {
        #expect(MCPTool.missingRequiredArgument(["urls": .null], schema: schema()) == nil)
    }

    @Test func anEmptyOrAbsentRequiredListBlocksNothing() {
        #expect(MCPTool.missingRequiredArgument([:], schema: schema(required: .array([]))) == nil)
        #expect(MCPTool.missingRequiredArgument([:], schema: ["type": "object"]) == nil)
        #expect(MCPTool.missingRequiredArgument([:], schema: ["type": "object", "properties": .object([:])]) == nil)
    }

    /// Malformed and exotic schemas fall open rather than shut. A server that describes itself in a
    /// way we do not parse still gets its calls.
    @Test func schemasWeCannotReadFallOpen() {
        #expect(MCPTool.missingRequiredArgument([:], schema: .string("nonsense")) == nil)
        #expect(MCPTool.missingRequiredArgument([:], schema: .null) == nil)
        #expect(MCPTool.missingRequiredArgument([:], schema: .array([.string("q")])) == nil)
        // `required` present but not a list of names.
        #expect(MCPTool.missingRequiredArgument([:], schema: schema(required: .string("urls"))) == nil)
        #expect(MCPTool.missingRequiredArgument([:], schema: schema(required: .array([.int(1)]))) == nil)
    }

    /// Only top-level `required` is checked. Nested requirements are the server's business, and
    /// walking them would be the kind of full validation that starts refusing valid calls.
    @Test func nestedRequirementsAreNotEnforced() {
        let nested: Value = [
            "type": "object",
            "properties": ["query": [
                "type": "object",
                "properties": ["text": ["type": "string"]],
                "required": ["text"]
            ]],
            "required": ["query"]
        ]
        #expect(MCPTool.missingRequiredArgument(["query": .object([:])], schema: nested) == nil)
    }

    // MARK: - What it catches, and what it says

    @Test func aMissingRequiredArgumentIsCaught() {
        let complaint = MCPTool.missingRequiredArgument(["url": .string("u")], schema: schema())
        #expect(complaint != nil)
    }

    @Test func theComplaintListsEveryRequiredName() throws {
        let two = schema(
            properties: ["urls": ["type": "array"], "objective": ["type": "string"]],
            required: ["urls", "objective"]
        )
        let complaint = try #require(MCPTool.missingRequiredArgument(["urls": .array([])], schema: two))
        #expect(complaint.contains("`objective`")) // the one actually missing
        #expect(complaint.contains("`urls`")) // …and the full requirement list
    }

    @Test func theArrayHintAppearsOnlyForArrays() throws {
        let arrayComplaint = try #require(MCPTool.missingRequiredArgument([:], schema: schema()))
        #expect(arrayComplaint.contains("pass a list"))

        let stringSchema = schema(properties: ["q": ["type": "string"]], required: ["q"])
        let stringComplaint = try #require(MCPTool.missingRequiredArgument([:], schema: stringSchema))
        #expect(!stringComplaint.contains("pass a list"))
        #expect(stringComplaint.contains("string"))
    }

    @Test func aPropertyWithNoDeclaredTypeIsStillNamed() throws {
        let untyped = schema(properties: ["urls": .object([:])], required: ["urls"])
        let complaint = try #require(MCPTool.missingRequiredArgument([:], schema: untyped))
        #expect(complaint.contains("`urls`"))
    }

    // MARK: - The near-miss hint

    @Test func theHintCatchesSingularPluralAndCase() {
        #expect(MCPTool.isNearMiss("url", of: "urls"))
        #expect(MCPTool.isNearMiss("urls", of: "url"))
        #expect(MCPTool.isNearMiss("URLs", of: "urls")) // case-only: the argument is still missing
        #expect(!MCPTool.isNearMiss("urls", of: "urls")) // identical is not a miss
        #expect(!MCPTool.isNearMiss("objective", of: "urls"))
    }

    /// With several plausible near-misses the wording must not depend on dictionary ordering, or
    /// the same call produces different text on different runs.
    @Test func theHintIsStableAcrossRuns() throws {
        let arguments: [String: AgentJSON] = [
            "url": .string("u"), "URLS": .string("u"), "urlsList": .string("u")
        ]
        let first = try #require(MCPTool.missingRequiredArgument(arguments, schema: schema()))
        for _ in 0 ..< 20 {
            #expect(MCPTool.missingRequiredArgument(arguments, schema: schema()) == first)
        }
    }

    @Test func noHintIsInventedWhenNothingResembles() throws {
        let complaint = try #require(
            MCPTool.missingRequiredArgument(["objective": .string("x")], schema: schema())
        )
        #expect(!complaint.contains("You passed"))
    }
}

// MARK: - MultiServerMCPClient

struct MultiServerMCPClientTests {
    private func emptyTool(_ name: String) -> MCP.Tool {
        MCP.Tool(name: name, description: "", inputSchema: ["type": "object"])
    }

    @Test func aggregatesAndPrefixesAcrossServers() async {
        let a = FakeMCPSession(tools: [emptyTool("one")])
        let b = FakeMCPSession(tools: [emptyTool("two")])
        let client = MultiServerMCPClient(
            configs: [
                MCPServerConfig(name: "a", kind: .stdio),
                MCPServerConfig(name: "b", kind: .http, url: "https://example.com/mcp")
            ]
        ) { config in config.name == "a" ? a : b }

        let names = await client.tools().map(\.name)
        #expect(Set(names) == ["a__one", "b__two"])
    }

    @Test func isolatesAFailingServer() async {
        let good = FakeMCPSession(tools: [emptyTool("ok")])
        let bad = FakeMCPSession(tools: [emptyTool("never")], failConnect: true)
        let client = MultiServerMCPClient(
            configs: [
                MCPServerConfig(name: "good", kind: .stdio),
                MCPServerConfig(name: "bad", kind: .stdio)
            ]
        ) { config in config.name == "good" ? good : bad }

        let names = await client.tools().map(\.name)
        #expect(names == ["good__ok"])
    }

    @Test func loadReportsPerServerStatusWithToolCountsAndErrors() async {
        let good = FakeMCPSession(tools: [emptyTool("ok"), emptyTool("ok2")])
        let bad = FakeMCPSession(tools: [emptyTool("never")], failConnect: true)
        let client = MultiServerMCPClient(
            configs: [
                MCPServerConfig(name: "good", kind: .stdio),
                MCPServerConfig(name: "bad", kind: .stdio)
            ]
        ) { config in config.name == "good" ? good : bad }

        let (tools, statuses) = await client.load()
        #expect(tools.map(\.name) == ["good__ok", "good__ok2"]) // the failing server contributes nothing
        let byName = Dictionary(uniqueKeysWithValues: statuses.map { ($0.name, $0) })
        #expect(byName["good"]?.connected == true)
        #expect(byName["good"]?.toolCount == 2)
        #expect(byName["bad"]?.connected == false) // surfaced, not silently dropped
        #expect(byName["bad"]?.error != nil)
        #expect(byName["bad"]?.toolCount == 0)
    }

    @Test func skipsDisabledServers() async {
        let a = FakeMCPSession(tools: [emptyTool("one")])
        let client = MultiServerMCPClient(
            configs: [MCPServerConfig(name: "a", kind: .stdio, isEnabled: false)]
        ) { _ in a }

        let names = await client.tools().map(\.name)
        #expect(names.isEmpty)
    }

    @Test func disambiguatesCollidingDispatchNames() async {
        // "my server" and "my_server" both sanitize to the same dispatch prefix, so their
        // identically-named tools collide; the later one gets a numeric suffix so both stay
        // reachable.
        let a = FakeMCPSession(tools: [emptyTool("t")])
        let b = FakeMCPSession(tools: [emptyTool("t")])
        let client = MultiServerMCPClient(
            configs: [
                MCPServerConfig(name: "my server", kind: .stdio),
                MCPServerConfig(name: "my_server", kind: .stdio)
            ]
        ) { config in config.name == "my server" ? a : b }

        let names = await client.tools().map(\.name)
        #expect(names == ["my_server__t", "my_server__t_2"])
    }
}
