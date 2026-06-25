# Installation

## Requirements

| Requirement | Version |
|---|---|
| macOS | 26+ (Tahoe) |
| Swift | 6.1+ |
| Apple Silicon | Required for MLX on-device inference (`DeepAgentsMLX`) |
| Xcode | 26+ recommended; required to build Ripple (Xcode emits MLX's Metal shader library) |

!!! note "Apple Silicon and MLX"
    `DeepAgentsMLX` uses Apple's MLX framework, which targets arm64 only. If you are building a CI pipeline or a server tool on x86, omit `DeepAgentsMLX` from your target. The core `DeepAgents` framework and all other adapter products have no such restriction.

## Add to Package.swift

Declare the dependency in your `Package.swift`:

=== "Package.swift"

    ```swift
    // swift-tools-version: 6.1
    import PackageDescription

    let package = Package(
        name: "MyApp",
        platforms: [.macOS("26.0")],
        dependencies: [
            .package(
                url: "https://github.com/dsaad68/deepagents-swift",
                from: "0.1.0"
            ),
        ],
        targets: [
            .executableTarget(
                name: "MyApp",
                dependencies: [
                    .product(name: "DeepAgents",          package: "deepagents-swift"),
                    .product(name: "DeepAgentsAnthropic", package: "deepagents-swift"),
                    // add other products as needed
                ]
            ),
        ]
    )
    ```

=== "Xcode - Add Package"

    1. In Xcode, choose **File > Add Package Dependencies...**
    2. Enter the URL: `https://github.com/dsaad68/deepagents-swift`
    3. Set the version rule to **Up to Next Major Version** from `0.1.0`.
    4. Select the products you need from the list and add them to your target.

## The 5 library products

Each product is an independent SwiftPM library. Add only the products your target actually needs - this keeps your binary lean and avoids pulling in heavy frameworks (MLX, AppKit) where they are not required.

| Product | Import statement | When to add |
|---|---|---|
| `DeepAgents` | `import DeepAgents` | Always - core framework, ReAct loop, middleware, MCP client, all toolsets |
| `DeepAgentsMLX` | `import DeepAgentsMLX` | On-device inference with Apple MLX (Apple Silicon only) |
| `DeepAgentsOpenAI` | `import DeepAgentsOpenAI` | OpenAI, Azure OpenAI, OpenRouter, or any OpenAI-compatible endpoint |
| `DeepAgentsAnthropic` | `import DeepAgentsAnthropic` | Anthropic Messages API or AWS Bedrock |
| `DeepAgentsMacTools` | `import DeepAgentsMacTools` | macOS-specific tools: screenshot, clipboard, Apple Notes, mac CLI, container shell |

!!! info "AppKit-free core"
    The `DeepAgents` core target imports only Foundation and the MCP Swift SDK. It carries zero MLX and zero AppKit dependencies. This is enforced by a CI import guard (`Scripts/check-framework-imports.sh`, run via `just guard`) that fails the build if any `import MLX` or `import AppKit` appears under the core target.

## CI import guard

If your CI runs on Linux or x86 hosts that do not have MLX or AppKit available, you can safely import only `DeepAgents` (plus OpenAI or Anthropic adapters) and the build will succeed. Guard MLX-specific code behind a conditional import if you share sources across targets:

```swift
#if canImport(DeepAgentsMLX)
import DeepAgentsMLX
// MLX-specific setup
#endif
```

This pattern keeps your code portable across environments while still letting the app target use the full MLX stack on device.
