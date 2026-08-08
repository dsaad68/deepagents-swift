import Foundation

/// Whether a capability's JSON schema is prefilled into every prompt, or discovered on demand.
///
/// The distinction exists for one reason: the rendered tool set is part of the cached prompt prefix
/// (see ``ToolSearchMiddleware``), so every core tool is paid for on every query - and every
/// auxiliary tool is free until the agent searches for it.
public enum ToolTier: String, Codable, Sendable, CaseIterable {
    /// Schema in the prompt from the first token. Always callable, always costs prompt tokens.
    case core
    /// Not in the prompt. Found through `search_tools`, then called like any other tool.
    case auxiliary

    /// A short label for the `/config` editor and the tool browsers.
    public var label: String {
        switch self {
        case .core: "core"
        case .auxiliary: "auxiliary"
        }
    }

    /// The other tier - what a toggle switches to.
    public var toggled: ToolTier { self == .core ? .auxiliary : .core }
}
