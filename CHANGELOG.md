# Changelog

All notable changes to the DeepAgents Swift framework are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- **A run no longer ends with an empty answer.** "No tool calls" was treated as "here is the
  answer", so a reasoning model that spent its whole turn inside `<think>` completed the run with
  nothing shown - observed on-device as five rounds of real tool work, 6,788 reasoning chunks and
  zero answer tokens. A silent turn is now dropped and the model asked once for the answer it never
  wrote, with its own reasoning quoted back so answering is a summary rather than a re-derivation.
  If even that turn stays silent the run replies with the reasoning under a label saying whose words
  they are, and with nothing at all to salvage it says so plainly. The same guarantee covers the
  iteration cap and the duplicate-round guard, which force a final turn of their own.

### Added

- **`ModelTurnSession.nextTurn(…, startingOutsideReasoning:)`** - generate a turn that begins in the
  answer channel rather than the reasoning one. A chat template that opens `<think>` for every turn
  means an answer only reaches `text` once the model emits `</think>`, so asking a model that has
  already thought itself silent to "answer in plain text" asks it to answer on a channel that is not
  open; this closes the block up front instead. `ReactAgent`'s forced final turn uses it. The
  requirement ships with a default implementation that ignores the flag, so existing conformers are
  unaffected. In the MLX adapter it appends `</think>` to the prompt (LFM2.5-2.6B, Ornith) or turns
  the thought channel off (Gemma 4).

- **A round's independent tool calls run concurrently.** `AgentTool.isParallelSafe` (default
  `false`) declares that a tool neither writes anything another call in the round could care about
  nor needs an earlier call's result. The loop splits a round into batches: each run of consecutive
  parallel-safe calls executes at once (at most four), everything else keeps its place in the serial
  order and still sees everything dispatched before it. Three `read_file` calls now cost one read
  instead of three. Declared by the read-only toolset: `ls`, `read_file`, `grep`, `glob`, `tree`,
  `head`, `tail`, `diff`, `fetch`, every `git_*` tool, `current_datetime`, `calculator`, `mdfind`,
  `read_clipboard`. Everything that writes keeps the default, and the round-ordering guarantee with
  it - a round of `write_file` then `read_file` still means what it looks like it means.
- **Tool events carry the call they belong to.** `.toolStarted`, `.toolProgress`, `.toolCompleted`
  and `.toolFailed` gained a `callID`, and `.toolStarted` a `batchID` shared by the calls that ran
  together (`nil` when a call ran alone). **Hosts must pair on `callID`, not on the tool name**:
  three `read_file` calls can be open at once and their completions arrive in whatever order they
  finish, so "the most recent unfinished step with this name" attaches results to the wrong one.
  `batchID` is what lets a host mark a group as having run in parallel. Both are `nil` only for an
  event no call produced - one a host synthesized, or a step rebuilt from a stored transcript -
  where name matching remains the right fallback.

### Changed

- **A batch announces every call before any of them runs**, then reports each completion as it
  lands, so a host shows the whole group running rather than entries that appear already finished.
  Tool *results* are still appended in call order, whatever order the calls finish in; the trained
  chat format pairs each call with its result, in order. Events a tool emits through its
  `ToolContext` (the `task` tool's `.toolProgress`, the shell's streamed output) are stamped with
  the call id on the way out, since a tool does not know its own call.
- **Approval requests are serialised, not dispatch.** A gated tool still fans out;
  `HumanInTheLoopMiddleware` presents one request at a time and raises the next when the previous
  decision returns. The queue holds the decision only, never the execution - an approved call runs
  while the next card is up, and a host that answers the gate itself (an allowlist, accept-all, a
  deny rule) never delays a batch. Excluding gated tools instead would have made the feature inert:
  every read-only tool defaults to `ask` in `MiddlewareCatalog`.
- **Terse tool results now say what they are.** A clean `git status` returned a bare `## main` - 22
  characters that read as a markdown heading, not an answer - and a 2.6B planner re-called it five
  rounds running. `git_status` states the branch and whether the tree is clean, spells out the
  two-column porcelain codes (`modified:`, `untracked:`, `added (staged):`; an unrecognized code
  keeps its raw line), and summarises the count before the file list. `git_diff`, `git_log` and
  `git_blame` replace `(no output)` with what the emptiness means. `calculator` echoes the
  expression (`(12 * 8) + 3 = 99`) instead of a bare number, and `shell` reports a silent success
  rather than `(no output)`. See the [custom tool guide](docs/guides/custom-tool.md) for the rule.
- **The read-only `git` tools pass `--no-optional-locks`**, so `status` and `diff` no longer take
  `.git/index.lock` to opportunistically refresh the index - the one write that would make two of
  them in a batch collide.

## [0.5.0] - 2026-08-06

### Added

- **The prefix KV store can be inspected and reclaimed.** `PrefixKVStore.inventory(directory:
  knownModelIDs:)` reports what is on disk grouped by model, with per-model and total byte counts;
  `removeAll(modelID:)` / `removeAll()` delete a model's artifacts or the whole store; `pruneNow()`
  applies the current limits immediately instead of at the next save. All three ignore the enabled
  flag - turning the cache off stops it writing, it does not make the hundreds of MB already
  written someone else's problem. The scan is read-only and reads safetensors headers only.
- **`PrefixKVStore.maxTotalBytes`** (default 4 GB, `0` for no cap) bounds the store by size, and
  `defaultDirectory` is now public so a host can show the path.
- **`ToolName`** - the single spelling rule for a tool name, applied in both directions.
  `ToolName.normalized(_:)` maps a name to `[a-z0-9_]`; it is what publishes a tool's name to the
  model *and* what every emitted name is put through on the way back in.
- **`ServerScopedTool`** - a tool that names the server which contributed it (`serverName`,
  `toolName`), plus **`toolsFromServer(_:in:)`** to attribute tools to a server. `MCPTool` conforms
  and now exposes `serverName`/`toolName` publicly. Hosts contributing their own server-backed
  tools should conform so approval defaults and per-server UI reach them.

### Changed

- **The snapshot limit is per model, not global.** `maxSnapshots = 6` counted every base in the
  directory, so alternating between two models evicted each other's warm base and left both cold.
  `PrefixKVStore.maxSnapshotsPerModel` (default 6, settable) gives each model its own budget. That
  relaxes the old bound - N models can now keep N x 6 snapshots - which is why `maxTotalBytes`
  ships alongside it; pruning applies the count first, then the byte ceiling, and never evicts the
  newest snapshot (a cap below one base size would otherwise write and delete it every turn).
  Ripple inherits both with no code change.
- **Artifacts are attributed to a model by metadata, never by file name.** Bases already carried a
  `model` key in their safetensors header; traces now carry one in their payload. The file name
  flattens `/` to `--` and can leave one model id prefixing another, so it cannot be parsed back
  reliably. Traces written before this still load, and are attributed by longest-prefix match.
- **`saveTrace` no longer prunes snapshots.** It runs after every turn, and the per-model pass has
  to read a header from every base; it now bounds only the traces it writes.
- **Published tool names are lowercased.** `ToolName.normalized` folds case as well as punctuation,
  so an MCP server exposing `getWeather` now dispatches as `server__getweather`. Every built-in
  tool name was already lowercase, so nothing there moves; a host reading `MCPTool.name` for
  display should read `toolName` instead, which is unchanged.
- **`mcpApprovalDefaults(servers:tools:)` only assigns a default to a `ServerScopedTool`.** It
  previously matched on the dispatch-name prefix, so any tool whose name merely started with
  `server__` was picked up. A host passing non-MCP tools that followed the convention by hand no
  longer gets a default for them - they fall to the tool catalog.

### Fixed

- **Tool names are normalized in both directions, so a hyphenated MCP server is reachable.** The
  naming convention was enforced on egress only: `MCPTool` sanitized the name it published and the
  name the model sent back was compared raw. A `parallel-search` server therefore exposed
  `parallel-search__web_search`, the model emitted `parallel_search__web_search` (models normalize
  punctuation out of a function name), and dispatch answered "unknown tool" - the whole server was
  unreachable. The rule now lives in one place, `ToolName.normalized`, and every name the model
  emits goes through it as it enters the loop, renamed to the tool's own spelling. From that line
  on exactly one spelling of a tool exists, so dispatch is an exact match again and the stored
  message, the duplicate-round signature, the events, the middleware chain and the approval gate
  cannot disagree about what was called. Servers are still invoked with their original tool names.
- **An MCP tool could inherit another server's approval mode.** `mcpApprovalDefaults` and
  `mcpToolsForDisplay` attributed a tool to a server by its dispatch-name prefix, which is
  normalized - so `parallel-search` and `parallel_search` both yield `parallel_search__` and every
  tool under it took whichever mode came last, letting a "Deny" server's tool run as "Approve".
  Attribution now goes through the new `ServerScopedTool` protocol, which carries the contributing
  server on the tool itself; `toolsFromServer(_:in:)` is the one way to ask. `MCPTool` conforms and
  exposes `serverName`/`toolName`. `MCPTool.dispatchPrefix` remains for building a name but must
  not be used to attribute one.
- **The duplicate-round guard told the model a failed call had a result.** A call that errored and
  was re-issued unchanged drew "its result is in the conversation above", so the model answered
  from a result that never existed (observed: an unfetched web page summarized as fact). A repeat
  of a round whose calls all failed now gets told the call failed and to fix the name, use another
  tool, or answer with what it has.

## [0.4.0] - 2026-08-05

### Added

- **LFM2.5 2.6B** (`LiquidAI/LFM2.5-2.6B-MLX/mxfp4` ~1.6 GB, `/mxfp8` ~2.8 GB, `/8bit` ~2.9 GB,
  `/bf16` ~5.4 GB) - a general-purpose 2.69B instruct planner on the `lfm2` codec, slotting between
  the 1.2B models and the 8B-A1B MoE. It runs at its full **128k** context window (every other row
  is clamped) and at the card's `repetition_penalty 1.1` rather than the family's 1.05. It reasons
  before answering, so it also gets the thinking models' 8192-token budget.
- **`MlxModel.startsInReasoning`**, threaded through `MlxChatModel` into the LFM2 decoder. LFM2.5
  2.6B's chat template prefills the opening `<think>` into the generation prompt, so its stream
  begins *inside* reasoning and only ever emits the closing tag - the same shape Ornith's qwen3_5
  codec already hard-codes. Without it the entire reasoning pass leaks into the visible answer,
  trailed by a bare `</think>`. Every other LFM2.5 template generates its own opening tag and is
  unaffected.
- **Precision subfolders.** A catalog id may now carry a third path component naming a subfolder of
  its Hugging Face repo (`LiquidAI/LFM2.5-2.6B-MLX/mxfp4`), for conversions that publish every
  precision in one repo instead of one repo per precision. The new `MlxModelID` splits an id into
  its repo and subfolder; `MlxModel` gains `repoId` and `subfolder`. The id itself remains the
  unique, opaque selection key, so nothing downstream changes.

### Changed

- **`MlxModelLoader.downloadSnapshot` scopes its glob patterns to the model's precision subfolder.**
  The hub matches globs with `fnmatch(glob, path, 0)` - no `FNM_PATHNAME` - so `*` crosses `/` and
  the previous unscoped patterns would have fetched *every* precision in a subfolder-packaged repo
  (~25 GB instead of 1.6 GB). Unchanged for one-precision-per-repo models.
- **`MlxModelLoader.removeFromDisk` deletes only the requested precision.** Sibling precisions share
  one cache directory and one content-addressed blob store, so removal now drops the subfolder and
  then prunes exactly the blobs no surviving precision still references - reclaiming the space
  without breaking siblings. The repo directory is removed once the last precision goes. Behaviour
  for a plain repo id is unchanged.
- `MlxModelLoader.isDownloadedOnDisk` is scoped per precision: a downloaded sibling no longer makes
  another precision report as present.

### Fixed

- **The LFM2.5 chat-template repair now runs after the download, not before it.** On a cold first
  load there was no snapshot yet, so the patch silently no-opped and a freshly downloaded model ran
  its entire first session on the broken template, self-healing only on the second load.

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

[0.5.0]: https://github.com/dsaad68/deepagents-swift/releases/tag/0.5.0
[0.4.0]: https://github.com/dsaad68/deepagents-swift/releases/tag/0.4.0
[0.3.0]: https://github.com/dsaad68/deepagents-swift/releases/tag/0.3.0
[0.2.3]: https://github.com/dsaad68/deepagents-swift/releases/tag/0.2.3
[0.2.2]: https://github.com/dsaad68/deepagents-swift/releases/tag/0.2.2
