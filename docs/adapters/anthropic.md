# Anthropic & Bedrock

`DeepAgentsAnthropic` provides two `ChatModel` implementations:

- `AnthropicChatModel` - calls `api.anthropic.com` directly (or any Anthropic-compatible endpoint)
  via the Messages API over SSE.
- `BedrockChatModel` - calls AWS Bedrock's `invoke-with-response-stream` endpoint using SigV4
  request signing or a bearer token (Amazon Bedrock API key). Reuses the same Anthropic message
  codec, so behaviour is identical from the agent's perspective.

## AnthropicChatModel

```swift
public struct AnthropicChatModel: ChatModel {
    public init(
        baseURL: URL,
        model: String,
        apiKey: String? = nil,
        ...,
        parameters: AnthropicGenerateParameters = .init(),
        anthropicVersion: String = "2023-06-01",
        betaHeaders: [String] = [],
        transport: (any AnthropicStreamingTransport)? = nil
    )
}
```

| Parameter | Type | Default | Purpose |
|---|---|---|---|
| `baseURL` | `URL` | (required) | Root URL; the adapter POSTs to `{baseURL}/v1/messages` |
| `model` | `String` | (required) | Model identifier, e.g. `"claude-opus-4-8"` |
| `apiKey` | `String?` | `nil` | Sent as `x-api-key`; read from environment when `nil` |
| `parameters` | `AnthropicGenerateParameters` | `.init()` | Sampling parameters (temperature, max tokens, etc.) |
| `anthropicVersion` | `String` | `"2023-06-01"` | Value of the `anthropic-version` header; matches the stable dated API version |
| `betaHeaders` | `[String]` | `[]` | Values appended to `anthropic-beta`; use to opt into preview features |
| `transport` | `(any AnthropicStreamingTransport)?` | `nil` | Inject a custom transport for testing or proxying |

The adapter sends a standard `POST {baseURL}/v1/messages` with:

- `x-api-key: <key>`
- `anthropic-version: <anthropicVersion>`
- `anthropic-beta: <betaHeaders joined by comma>` (omitted when empty)
- `content-type: application/json`
- `accept: text/event-stream`

Responses are consumed as SSE; the `AnthropicDecoder` extracts tool use blocks, reasoning content,
and text content into `AgentMessage` fields.

### Error handling

`AnthropicModelError` is thrown on non-2xx responses or malformed SSE frames.

## BedrockChatModel

`BedrockChatModel` targets the Bedrock `invoke-with-response-stream` endpoint. It authenticates with
either AWS SigV4 request signing (`SigV4Signer`, built on CryptoKit) or an Amazon Bedrock API key
(bearer token), selected via `BedrockAuth`, and decodes the AWS binary event-stream framing around
the same Anthropic SSE payload.

```swift
public struct BedrockChatModel: ChatModel {
    public init(
        region: String,
        model: String,
        auth: BedrockAuth,
        baseURL: String? = nil,   // endpoint base, used verbatim; required for bearer-token auth
        ...
    )
}
```

### BedrockAuth

```swift
public enum BedrockAuth: Sendable, Equatable {
    case sigV4(BedrockCredentials)
    case bearerToken(String)

    public static func resolve(bearerToken: String? = nil) -> BedrockAuth?
}
```

`resolve(bearerToken:)` prefers a bearer token over SigV4: an explicit `bearerToken`, then the
`AWS_BEARER_TOKEN_BEDROCK` environment variable, then `BedrockCredentials.fromEnvironment()`; it
returns `nil` when none are set. Bearer auth sends `Authorization: Bearer <token>` and requires a
`baseURL` (used verbatim); SigV4 derives the endpoint from `region`.

### BedrockCredentials

```swift
public struct BedrockCredentials: Sendable {
    public let accessKey: String
    public let secretKey: String
    public let sessionToken: String?

    public static func fromEnvironment() -> BedrockCredentials?
}
```

`fromEnvironment()` reads `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and optionally
`AWS_SESSION_TOKEN` from `ProcessInfo.processInfo.environment`. Returns `nil` if the required
keys are absent.

!!! tip "Cross-region inference profiles"
    The `model` parameter accepts cross-region inference profile IDs of the form
    `us.anthropic.claude-...`. Use these when your Bedrock quota is spread across regions and you
    want Bedrock to route automatically. Pass the profile ID exactly as shown in the Bedrock
    console.

### Error handling

`BedrockModelError` is thrown on signing failures, HTTP errors, or malformed event-stream frames.

## Examples

=== "Anthropic direct"

    ```swift
    import DeepAgents
    import DeepAgentsAnthropic

    let model = AnthropicChatModel(
        baseURL: URL(string: "https://api.anthropic.com")!,
        model: "claude-opus-4-8",
        apiKey: ProcessInfo.processInfo.environment["ANTHROPIC_API_KEY"]
    )

    let agent = createDeepAgent(
        model: model,
        systemPrompt: "You are a helpful assistant.",
        maxIterations: 24
    )

    let ok = await agent.run([.human("What is 2 + 2?")]) { event in
        // handle AgentEvent
    }
    ```

=== "AWS Bedrock"

    ```swift
    import DeepAgents
    import DeepAgentsAnthropic

    // Prefers a bearer token (AWS_BEARER_TOKEN_BEDROCK) over SigV4 env credentials
    // (AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_SESSION_TOKEN).
    guard let auth = BedrockAuth.resolve() else {
        fatalError("No Bedrock credentials: set AWS_BEARER_TOKEN_BEDROCK or AWS_ACCESS_KEY_ID/SECRET")
    }

    let model = BedrockChatModel(
        region: "us-east-1",
        // Cross-region inference profile - Bedrock routes across us-* regions automatically
        model: "us.anthropic.claude-opus-4-8-20251101-v1:0",
        auth: auth,
        baseURL: "https://bedrock-runtime.us-east-1.amazonaws.com" // required for bearer-token auth
    )

    let agent = createDeepAgent(
        model: model,
        systemPrompt: "You are a helpful assistant.",
        maxIterations: 24
    )
    ```

## Related

- [Adapters overview](index.md)
- [Architecture](../concepts/architecture.md)
