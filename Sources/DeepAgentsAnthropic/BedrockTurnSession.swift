import DeepAgents
import Foundation

/// One run's stateless model node for Bedrock's `invoke-with-response-stream`. Each `nextTurn`
/// renders the conversation with ``AnthropicMessageCodec`` (Bedrock flavor: no `model`/`stream`,
/// plus `anthropic_version`), SigV4-signs the POST, then reassembles the AWS event-stream frames
/// and feeds each frame's inner Anthropic event through ``AnthropicDecoder`` - the same decoder the
/// direct Anthropic adapter uses.
public final class BedrockTurnSession: ModelTurnSession {
    private let region: String
    private let model: String
    private let credentials: BedrockCredentials
    private let supportsVision: Bool
    private let parameters: AnthropicGenerateParameters
    private let transport: any BedrockStreamingTransport

    init(
        region: String,
        model: String,
        credentials: BedrockCredentials,
        supportsVision: Bool,
        parameters: AnthropicGenerateParameters,
        transport: any BedrockStreamingTransport
    ) {
        self.region = region
        self.model = model
        self.credentials = credentials
        self.supportsVision = supportsVision
        self.parameters = parameters
        self.transport = transport
    }

    public func nextTurn(
        messages: [AgentMessage],
        systemPrompt: String?,
        tools: [any AgentTool],
        onChunk: @escaping @Sendable (AgentStreamChunk) -> Void
    ) async throws -> AgentMessage {
        guard let url = Self.endpoint(region: region, model: model) else {
            throw BedrockModelError.badModelID(model)
        }
        let (system, rendered) = AnthropicMessageCodec.render(
            systemPrompt: systemPrompt, messages: messages, supportsVision: supportsVision
        )
        let body = AnthropicMessageCodec.requestBody(
            model: model, system: system, messages: rendered,
            tools: AnthropicMessageCodec.tools(tools), parameters: parameters, bedrock: true
        )
        let data = try JSONSerialization.data(withJSONObject: body)

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("application/vnd.amazon.eventstream", forHTTPHeaderField: "Accept")
        request.httpBody = data
        SigV4.sign(&request, body: data, credentials: credentials, region: region)

        let (status, byteStream) = try await transport.send(request)
        guard (200 ..< 300).contains(status) else {
            var errorBody = Data()
            for try await chunk in byteStream { errorBody.append(chunk) }
            throw BedrockModelError.http(
                status: status, body: String(data: errorBody, encoding: .utf8) ?? ""
            )
        }

        let decoder = AnthropicDecoder()
        var parser = BedrockEventStreamParser()
        for try await chunk in byteStream {
            let (events, errors) = parser.ingest(chunk)
            if let message = errors.first { throw BedrockModelError.stream(message) }
            for eventData in events {
                for piece in decoder.ingest(eventData: eventData) { onChunk(piece) }
            }
        }
        // An Anthropic `error` event wrapped in a chunk frame is a failed turn, not a success.
        if let streamError = decoder.streamError { throw BedrockModelError.stream(streamError) }
        let (trailing, message) = decoder.finish()
        for piece in trailing { onChunk(piece) }
        return message
    }

    /// The Bedrock Runtime streaming-invoke URL for `model` in `region`. The model id is path-encoded
    /// (versioned ids contain `:`, which must become `%3A`).
    static func endpoint(region: String, model: String) -> URL? {
        var allowed = CharacterSet.alphanumerics
        allowed.insert(charactersIn: "-._~")
        let encoded = model.addingPercentEncoding(withAllowedCharacters: allowed) ?? model
        return URL(
            string: "https://bedrock-runtime.\(region).amazonaws.com/model/\(encoded)/invoke-with-response-stream"
        )
    }
}
