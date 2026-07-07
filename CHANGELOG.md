# Changelog

All notable changes to the DeepAgents Swift framework are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-07-08

### Added

- **Codec-family architecture in `DeepAgentsMLX`.** Each `MlxModel` declares a `codecFamily`
  (`lfm2` | `qwen35` | `gemma4`) and the turn session selects the encode/decode codec per turn.
  Reasoning splitting is a shared, tag-parameterizable `ThinkStream`. A new `acceptsImages`
  capability (independent of the `kind`-based `isVision`) lets one repo both plan and see.
- **Ornith-1.0-9B** (`mlx-community/Ornith-1.0-9B-4bit` / `-8bit`) - a qwen3_5 reasoning model
  that surfaces `<think>` reasoning, emits Qwen-XML tool calls (parsed by mlx-swift-lm's
  `XMLFunctionParser`), and loads through the VLM factory so it also backs vision.
- **Qwen3.6** (`mlx-community/Qwen3.6-27B-OptiQ-4bit`, `Qwen3.6-35B-A3B-OptiQ-4bit` MoE) -
  text-only qwen35 planners with card sampling (the 35B adds presence penalty 1.5); OptiQ
  per-layer bit overrides load via mlx-swift-lm.
- **Gemma 4 E4B** (`mlx-community/gemma-4-e4b-it-8bit`, `gemma-4-e4b-it-OptiQ-4bit`) - a new
  `gemma4` codec family: reasoning arrives in Gemma's thought channel (enabled via the template's
  `<|think|>` trigger), tool calls parse via `GemmaFunctionParser`, the encode carries the
  template's extras (tool-call ids, re-rendered reasoning), and MCP tool schemas are normalized
  to always carry a string `type` (swift-jinja renders `type | upper` strictly).
- **Prefix KV caching** (`PrefixCacheSlot`, `MlxChatModel(prefixCache:)`). The computed KV/SSM
  state of the prompt's unchanged token prefix is reused instead of re-prefilling the whole
  system+tools prompt every ReAct round - a multi-round turn's time-to-first-token drops from
  ~30 s to under a second on Ornith-9B. Hybrid attention+Mamba families reuse via `copy()`
  snapshots (a *tip* for rounds, a *base* for new queries); fully-trimmable caches use
  `KVCache.trim`; rotating (sliding-window) caches always take the snapshot path.
- **Disk-persisted prefix KV** (`PrefixKVStore`, `~/.cache/deepagents/prefix-kv/`). The base
  snapshot survives the process: a fresh launch resumes it on its first turn (cold prompt
  processing ~14 s → ~0.2 s measured on Ornith 9B 4-bit). Snapshots are content-addressed
  (store v2): a cold start resumes from the longest stored base that strict-prefixes the prompt -
  whichever configuration wrote it - then deepens to its own stable boundary. Hosts can set
  `PrefixKVStore.isEnabledOverride`; `DEEPAGENTS_PREFIX_KV=0` is the env kill switch.
- **Apple Notes agent tools** (`AppleNotesMiddleware` in `DeepAgentsMacTools`): `list_notes`,
  `read_note`, `create_note`, `update_note` drive Notes.app over AppleScript via an `osascript`
  subprocess (no host-run-loop hang, no script injection; bulk-fetch listing). Writes are
  approval-gated.
- **`MlxModelLoader` diagnostics.** The loader records why a load failed (`lastLoadError`) for
  hosts to surface, and `inFlightDownloadBytes(since:)` reports live download progress by sizing
  the in-flight URLSession temp files (the hub's Xet transport reports no incremental progress).

### Fixed

- **Qwen3.6 OptiQ generated gibberish.** mlx-swift-lm's `loadWeights` reads every
  `*.safetensors` in a snapshot, and the OptiQ sidecars (`mtp.safetensors`,
  `optiq_vision.safetensors`) tripped the qwen3_5 sanitize heuristic that "+1"-shifts norm
  weights, silently corrupting every layer norm. Text-factory loads now go through a symlink
  view of the snapshot that omits any safetensors not listed in the index
  (`~/.cache/deepagents/model-views/`), matching Python mlx-lm.
- **Nested tool arguments arrive typed on the qwen3_5/gemma4 families.** The turn's tool schemas
  are now passed into `generateTask`, so array/object parameters (e.g. `ask_user`'s `questions`)
  parse into typed values instead of raw JSON strings.

### Changed

- **mlx-swift-lm 3.31.3 → 3.31.4** (mlx-swift 0.31.4): the `TokenRing` 2-D-prompt crash fix
  restores sampling penalties for VLM-loaded models, qwen3_5 gains a text-only-inference crash
  fix and fp32 gated-delta state, GDN prefill is pipelined, and LFM2-MoE routing is fixed.

## [0.2.4] - 2026-06-29

### Added

- **AWS Bedrock bearer-token (API key) authentication.** `BedrockChatModel` now authenticates via a
  new `BedrockAuth` enum -- either AWS SigV4 request signing (`.sigV4(BedrockCredentials)`) or an
  Amazon Bedrock API key sent as `Authorization: Bearer <token>` (`.bearerToken(String)`).
  `BedrockAuth.resolve(bearerToken:)` prefers an explicit token, then the `AWS_BEARER_TOKEN_BEDROCK`
  environment variable, then SigV4 environment credentials. `BedrockChatModel` gains an optional
  `baseURL` used verbatim as the endpoint base (required for bearer auth; SigV4 still derives the
  endpoint from `region`).

### Changed

- **`BedrockChatModel.init` takes `auth: BedrockAuth` instead of `credentials: BedrockCredentials`.**
  Migrate `credentials: creds` call sites to `auth: .sigV4(creds)`.

## [0.2.3] - 2026-06-26

### Added

- **`requireOAuth` on `SwiftSDKMCPSession`.** A new initializer flag that force-attaches the SDK's
  OAuth authorizer to an HTTP server even when its config carries no `oauth` key -- used to drive a
  sign-in against a server whose auth requirement was discovered from a `401` rather than declared up
  front.

### Changed

- **HTTP MCP transport attaches the OAuth authorizer more eagerly.** It is now attached when the
  server is declared `oauth`, when `requireOAuth` is set, **or when a Keychain token already exists**
  for the server. This lets a server that was signed in once reconnect silently (no `oauth` key
  needed) and lets a host discover-then-sign-in a plain HTTP server. The authorizer stays lazy -- it
  only runs the browser flow on a `401`, so attaching it never opens a browser while a token is valid.

## [0.2.2]

- Added the `DeepAgentsVersion.current` constant so host front-ends (the Ripple CLI's `--version`,
  the Mispher app's About pane) can report the framework build they were compiled against.

[0.3.0]: https://github.com/dsaad68/deepagents-swift/releases/tag/0.3.0
[0.2.3]: https://github.com/dsaad68/deepagents-swift/releases/tag/0.2.3
[0.2.2]: https://github.com/dsaad68/deepagents-swift/releases/tag/0.2.2
