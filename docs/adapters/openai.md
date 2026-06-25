# OpenAI & Azure

`DeepAgentsOpenAI` provides `OpenAIChatModel`, a single struct that speaks the OpenAI
chat-completions streaming API. One implementation covers:

- **OpenAI** - `api.openai.com`, GPT-4o, o-series, etc.
- **OpenRouter** - drop in a different `baseURL` + bearer token; set `reasoning: true` for
  streamed chain-of-thought on supported models.
- **Self-hosted OpenAI-compatible** - vLLM, Ollama, LM Studio, llama.cpp server.
- **Azure OpenAI** - Azure path + `api-version` query parameter via `OpenAIEndpointStyle.azure`.

## OpenAIChatModel

```swift
public struct OpenAIChatModel: ChatModel {
    public init(
        baseURL: URL,
        model: String,
        apiKey: String? = nil,
        ...,
        reasoning: Bool = false,
        auth: OpenAIAuthStyle = .bearer,
        endpointStyle: OpenAIEndpointStyle = .standard,
        transport: (any OpenAIStreamingTransport)? = nil
    )
}
```

| Parameter | Type | Default | Purpose |
|---|---|---|---|
| `baseURL` | `URL` | (required) | Root of the API; the path suffix is appended by the adapter |
| `model` | `String` | (required) | Model identifier passed in the request body |
| `apiKey` | `String?` | `nil` | Passed as the credential; read from the environment when `nil` (see below) |
| `reasoning` | `Bool` | `false` | Requests streamed chain-of-thought; needed for OpenRouter reasoning models and o-series extended thinking |
| `auth` | `OpenAIAuthStyle` | `.bearer` | How the credential is sent |
| `endpointStyle` | `OpenAIEndpointStyle` | `.standard` | Controls URL path construction |
| `transport` | `(any OpenAIStreamingTransport)?` | `nil` | Inject a custom transport for testing or proxying |

### OpenAIAuthStyle

```swift
public enum OpenAIAuthStyle {
    case bearer   // Authorization: Bearer <key>  (OpenAI, OpenRouter, most self-hosted)
    case apiKey   // api-key: <key>  (Azure OpenAI)
}
```

### OpenAIEndpointStyle

```swift
public enum OpenAIEndpointStyle {
    case standard                                        // POST {baseURL}/chat/completions
    case azure(deployment: String, apiVersion: String)  // Azure path + ?api-version=...
}
```

`.standard` appends `/chat/completions` to `baseURL`. `.azure` constructs the Azure-style path:
`{baseURL}/openai/deployments/{deployment}/chat/completions?api-version={apiVersion}`.

### OpenAIGenerateParameters

`OpenAIGenerateParameters` carries sampling parameters (temperature, top-p, max tokens, etc.).
Pass it via the unlabelled `...` parameters in the init. Refer to the API reference for the full
field list.

### Error handling

`OpenAIModelError` is thrown by `OpenAITurnSession` on non-2xx responses or malformed SSE events.

## API key from the environment

When `apiKey` is `nil`, the adapter looks up the key from `ProcessInfo.processInfo.environment`
at call time. Set `OPENAI_API_KEY` (or the provider-specific variable) in your environment rather
than hardcoding keys in source.

## Examples

=== "OpenAI"

    ```swift
    import DeepAgents
    import DeepAgentsOpenAI

    let model = OpenAIChatModel(
        baseURL: URL(string: "https://api.openai.com")!,
        model: "gpt-4o",
        apiKey: ProcessInfo.processInfo.environment["OPENAI_API_KEY"]
    )

    let agent = createDeepAgent(
        model: model,
        systemPrompt: "You are a helpful assistant.",
        maxIterations: 24
    )
    ```

=== "OpenRouter (with reasoning)"

    ```swift
    import DeepAgents
    import DeepAgentsOpenAI

    // OpenRouter uses a bearer token and its own base URL.
    // Set reasoning: true to receive streamed chain-of-thought from supported models.
    let model = OpenAIChatModel(
        baseURL: URL(string: "https://openrouter.ai/api/v1")!,
        model: "anthropic/claude-opus-4",
        apiKey: ProcessInfo.processInfo.environment["OPENROUTER_API_KEY"],
        reasoning: true,
        auth: .bearer
    )
    ```

=== "Self-hosted (OpenAI-compatible)"

    ```swift
    import DeepAgents
    import DeepAgentsOpenAI

    // vLLM, Ollama, LM Studio, llama.cpp - anything that speaks the completions API.
    // No API key needed for a local server; pass nil or an empty string.
    let model = OpenAIChatModel(
        baseURL: URL(string: "http://localhost:11434/v1")!,
        model: "llama3.2",
        apiKey: nil
    )
    ```

=== "Azure OpenAI"

    ```swift
    import DeepAgents
    import DeepAgentsOpenAI

    // Azure uses the api-key header and a deployment-scoped URL path.
    let model = OpenAIChatModel(
        baseURL: URL(string: "https://my-resource.openai.azure.com")!,
        model: "gpt-4o",   // model field still required; Azure uses 'deployment' in the path
        apiKey: ProcessInfo.processInfo.environment["AZURE_OPENAI_API_KEY"],
        auth: .apiKey,
        endpointStyle: .azure(deployment: "my-gpt4o-deployment", apiVersion: "2025-01-01-preview")
    )
    ```

## Related

- [Adapters overview](index.md)
- [Architecture](../concepts/architecture.md)
