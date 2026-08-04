# macOS tools

`DeepAgentsMacTools` is not a model adapter - it is a collection of `AgentMiddleware` conformers
that give an agent native access to the macOS desktop environment. Pair it with any of the three
model adapters.

## What it provides

Each middleware contributes a set of tools. The agent loop merges them with any other tools at
factory time and renders them into the model's system prompt.

| Middleware | Tools | Mechanism | Notes |
|---|---|---|---|
| `MacToolsMiddleware` | `mdfind`, `open`, `open_app`, `download`, `say`, `notify` | Foundation APIs + subprocess | Spotlight search, URL/file opening, app launching, file downloads, TTS, system notifications |
| `ScreenshotMiddleware` | `take_screenshot`, `take_window_screenshots` | ScreenCaptureKit | Full-screen or per-window capture; requires Screen Recording permission |
| `ClipboardMiddleware` | `read_clipboard`, `write_clipboard` | `NSPasteboard` | Reads and writes the general pasteboard |
| `AppleNotesMiddleware` | `list_notes`, `read_note`, `create_note`, `update_note` | `osascript` subprocess | Avoids TUI hang by shelling out rather than using ScriptingBridge; **write tools are human-in-the-loop gated** |

### Tool reference

#### MacToolsMiddleware

| Tool | Description |
|---|---|
| `mdfind` | Run a Spotlight query; returns matching file paths |
| `open` | Open a URL or file path with its default application |
| `open_app` | Launch an application by name |
| `download` | Download a file from a URL to the local filesystem |
| `say` | Speak text aloud via macOS TTS |
| `notify` | Post a user notification via `NSUserNotificationCenter` / `UNUserNotificationCenter` |

#### ScreenshotMiddleware

| Tool | Description |
|---|---|
| `take_screenshot` | Capture the full display; returns an image in the agent's message context |
| `take_window_screenshots` | Capture individual windows; returns one image per window |

Screenshots are returned as `AgentImage` values inside `AgentContentBlock.image` blocks.
If the active model's `supportsVision` is `true` (e.g. an LFM2.5 VL model or a Claude vision
model), the images flow naturally into the next model turn.

#### ClipboardMiddleware

| Tool | Description |
|---|---|
| `read_clipboard` | Read the current plain-text content of the general pasteboard |
| `write_clipboard` | Write a string to the general pasteboard |

#### AppleNotesMiddleware

| Tool | Description |
|---|---|
| `list_notes` | List notes (optionally filtered by title) |
| `read_note` | Return the plain-text body of a named note |
| `create_note` | Create a new note with a given title and body |
| `update_note` | Append to or replace the body of an existing note |

!!! warning "Write tools require approval"
    `create_note` and `update_note` are human-in-the-loop gated. An `approvalHandler` must be
    provided to `createDeepAgent` (or `createAgent`) for these tools to be permitted to execute.
    Without it, write calls are blocked. See [Human in the loop](../concepts/human-in-the-loop.md).

## Permissions

Two macOS permissions must be granted before the relevant middleware can function:

| Permission | Required by | How to grant |
|---|---|---|
| **Screen Recording** | `ScreenshotMiddleware` | System Settings - Privacy & Security - Screen Recording; the app must be listed and enabled |
| **Automation (Notes)** | `AppleNotesMiddleware` | System Settings - Privacy & Security - Automation; grant access to Notes.app for the running process |

Missing Screen Recording permission causes `take_screenshot` and `take_window_screenshots` to
return an error rather than crashing.

## Usage

```swift
import DeepAgents
import DeepAgentsAnthropic   // or DeepAgentsMLX / DeepAgentsOpenAI
import DeepAgentsMacTools

let model = AnthropicChatModel(
    baseURL: URL(string: "https://api.anthropic.com")!,
    model: "claude-opus-4-8",
    apiKey: ProcessInfo.processInfo.environment["ANTHROPIC_API_KEY"]
)

let agent = createDeepAgent(
    model: model,
    systemPrompt: "You are a macOS assistant.",
    middleware: [
        MacToolsMiddleware(),
        ScreenshotMiddleware(),
        ClipboardMiddleware(),
        AppleNotesMiddleware(),
    ],
    approvalHandler: { request in
        // Gate destructive tools - called before create_note / update_note execute
        print("Approve \(request.toolName)?")
        return .approve
    },
    maxIterations: 24
)
```

!!! tip "Mixing with built-in middleware"
    `createDeepAgent`'s `includeGeneralPurpose` flag adds web, search, and text tools
    automatically. `DeepAgentsMacTools` middleware passed in `middleware:` is layered on top, so
    you get both sets without conflict.

## Related

- [Adapters overview](index.md)
- [Middleware](../concepts/middleware.md)
- [Human in the loop](../concepts/human-in-the-loop.md)
