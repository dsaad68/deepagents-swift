import DeepAgents
import Foundation

/// AWS credentials for signing Bedrock requests. The session token is optional (set for temporary
/// STS/role credentials). ``fromEnvironment()`` reads the standard AWS variables - the only
/// credential source Ripple's Bedrock models use, keeping secrets out of `settings.json`.
public struct BedrockCredentials: Sendable, Equatable {
    public let accessKey: String
    public let secretKey: String
    public let sessionToken: String?

    public init(accessKey: String, secretKey: String, sessionToken: String? = nil) {
        self.accessKey = accessKey
        self.secretKey = secretKey
        self.sessionToken = sessionToken
    }

    /// Build credentials from `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` / optional
    /// `AWS_SESSION_TOKEN`, or nil when the required pair isn't set.
    public static func fromEnvironment() -> BedrockCredentials? {
        let env = ProcessInfo.processInfo.environment
        guard let accessKey = env["AWS_ACCESS_KEY_ID"], !accessKey.isEmpty,
              let secretKey = env["AWS_SECRET_ACCESS_KEY"], !secretKey.isEmpty
        else { return nil }
        let token = env["AWS_SESSION_TOKEN"]
        return BedrockCredentials(
            accessKey: accessKey, secretKey: secretKey,
            sessionToken: (token?.isEmpty == false) ? token : nil
        )
    }
}

/// A `ChatModel` over **Anthropic models on AWS Bedrock**, via the Bedrock Runtime
/// `invoke-with-response-stream` API. Speaks the same Messages wire format as ``AnthropicChatModel``
/// (it reuses ``AnthropicMessageCodec`` and ``AnthropicDecoder``), but signs each request with SigV4
/// and reads the AWS event-stream framing instead of SSE. Pure Foundation + CryptoKit; no AWS SDK.
public struct BedrockChatModel: ChatModel {
    let region: String
    /// The Bedrock model or cross-region inference-profile id (e.g. `us.anthropic.claude-opus-4-8`).
    let model: String
    let credentials: BedrockCredentials
    public let supportsVision: Bool
    public var modelID: String?
    public var contextWindowTokens: Int?
    let parameters: AnthropicGenerateParameters
    let transport: any BedrockStreamingTransport

    public init(
        region: String,
        model: String,
        credentials: BedrockCredentials,
        supportsVision: Bool = false,
        modelID: String? = nil,
        contextWindowTokens: Int? = nil,
        parameters: AnthropicGenerateParameters = .init(),
        transport: (any BedrockStreamingTransport)? = nil
    ) {
        self.region = region
        self.model = model
        self.credentials = credentials
        self.supportsVision = supportsVision
        self.modelID = modelID ?? model
        self.contextWindowTokens = contextWindowTokens
        self.parameters = parameters
        self.transport = transport ?? URLSessionBedrockTransport()
    }

    public func makeSession() -> any ModelTurnSession {
        BedrockTurnSession(
            region: region, model: model, credentials: credentials,
            supportsVision: supportsVision, parameters: parameters, transport: transport
        )
    }
}

/// An error from the Bedrock Runtime endpoint - a non-2xx response body, an in-stream exception
/// frame, or a model id that can't form a valid endpoint URL.
public enum BedrockModelError: Error, CustomStringConvertible {
    case http(status: Int, body: String)
    case stream(String)
    case badModelID(String)

    public var description: String {
        switch self {
        case .http(let status, let body):
            let detail = body.isEmpty ? "" : ": \(body)"
            return "Bedrock request failed (HTTP \(status))\(detail)"
        case .stream(let message):
            return "Bedrock stream error: \(message)"
        case .badModelID(let model):
            return "Invalid Bedrock model id: \(model)"
        }
    }
}
