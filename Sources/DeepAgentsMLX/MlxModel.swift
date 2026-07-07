import DeepAgents
import Foundation
import MLXLMCommon

/// The on-device message codec a model is driven with. A new wire format is a new codec (and a new
/// case here), selected per model in ``RebuildTurnSession`` - see ``MlxModel/codecFamily``.
public enum MlxCodecFamily: Sendable { case lfm2, qwen35, gemma4 }

/// One on-device MLX model the user can run via `mlx-swift-lm`. Identified by its
/// Hugging Face repo id (downloaded on demand by the package). The catalog is the
/// LiquidAI LFM2.5 family — language (instruct) and vision (VL) — plus the qwen3_5-family
/// reasoning VLMs (Ornith, Qwen3.6) and Gemma 4 E4B, and is trivially extensible: add a row
/// here and it shows up in Settings ▸ Local models.
public struct MlxModel: Identifiable, Sendable, Hashable {
    public enum Kind: Sendable { case language, vision }

    /// Hugging Face repo id, e.g. "LiquidAI/LFM2.5-1.2B-Instruct-MLX-8bit".
    public let id: String
    public let displayName: String
    /// Short descriptor for the row, e.g. "Instruct · 8-bit".
    public let detail: String
    public let kind: Kind
    /// Rough on-disk / resident size, for the row's size hint and a "large" warning.
    public let approxGB: Double
    /// Whether this model accepts image input. Normally tracks `kind == .vision`, but a unified VLM
    /// like Ornith (qwen3_5) is cataloged `.language` - so it lists in the planner picker and drives
    /// the ReAct loop with tools - yet still takes images, so it can also back the vision subagent.
    /// Defaults to `kind == .vision`, so the LFM2 rows are unaffected.
    public let acceptsImages: Bool

    public init(
        id: String, displayName: String, detail: String, kind: Kind, approxGB: Double,
        acceptsImages: Bool? = nil
    ) {
        self.id = id
        self.displayName = displayName
        self.detail = detail
        self.kind = kind
        self.approxGB = approxGB
        self.acceptsImages = acceptsImages ?? (kind == .vision)
    }

    /// True when the model is *classified* as vision (drives picker placement, and the
    /// VisionAgent-vs-AskAgent choice for a standalone selection). A unified VLM stays `.language`
    /// here so it runs as a planner; use ``loadsAsVision`` to decide the load factory / image gating.
    public var isVision: Bool { kind == .vision }

    /// True when the model must load through the VLM factory and can be handed images - either a
    /// dedicated vision model or a unified VLM cataloged `.language` with ``acceptsImages``.
    public var loadsAsVision: Bool { isVision || acceptsImages }

    /// Recommended sampling for this model when driving the on-device ReAct agent, per model card /
    /// docs. LFM2.5: 8B-A1B is temperature 0.2 / top-k 80; the 1.2B instruct models are temperature
    /// 0.1 / top-k 50 (more deterministic — they follow tool instructions more reliably); Thinking
    /// additionally wants top-p 0.1; all take Liquid's repetition penalty 1.05. Qwen3.6 cards want
    /// temperature 1.0 / top-p 0.95 / top-k 20 (the 35B-A3B adds presence penalty 1.5). Generous
    /// `maxTokens` so reasoning + the tool loop aren't truncated.
    ///
    /// (Historical: VLM-loaded models used to run without penalties to dodge the `TokenRing`
    /// 2-D-prompt crash; fixed upstream in mlx-swift-lm 3.31.4, #170.)
    public var agentParameters: GenerateParameters {
        if isVision {
            return .init(maxTokens: 4096, temperature: 0.1, topK: 50, repetitionPenalty: 1.05)
        }
        if id.contains("Ornith") {
            // Ornith-1.0-9B (deepreinforce-ai) is a qwen3_5 reasoning model. Its card recommends
            // temperature 0.6 / top-p 0.95 / top-k 20; the generous `maxTokens` matches the Thinking
            // models' budget, since the <think> pass plus the per-round tool loop spend a lot of
            // tokens before the final answer.
            return .init(maxTokens: 8192, temperature: 0.6, topP: 0.95, topK: 20, repetitionPenalty: 1.05)
        }
        if id.contains("Qwen3.6") {
            // Qwen3.6 cards (27B / 35B-A3B): temperature 1.0 / top-p 0.95 / top-k 20; the 35B-A3B
            // additionally recommends presence_penalty 1.5 (27B: 0.0). The cards suggest an 81920
            // output budget for long reasoning - far past the on-device context window - so clamp
            // `maxTokens` to the same reasoning budget the other thinking models use.
            let presence: Float? = id.contains("35B-A3B") ? 1.5 : nil
            return .init(maxTokens: 8192, temperature: 1.0, topP: 0.95, topK: 20, presencePenalty: presence)
        }
        if isGemmaFamily {
            // Gemma 4 E4B card + shipped generation_config.json: temperature 1.0 / top-p 0.95 /
            // top-k 64, no penalties. Thinking is on (the chat template's `enable_thinking`
            // injects `<|think|>`), so give the thought channel + tool loop the same reasoning
            // budget as the other thinking models.
            return .init(maxTokens: 8192, temperature: 1.0, topP: 0.95, topK: 64)
        }
        if id.contains("8B-A1B") {
            return .init(maxTokens: 4096, temperature: 0.2, topK: 80, repetitionPenalty: 1.05)
        }
        if id.contains("Thinking") {
            // Reasoning models spend tokens on a <think> pass before answering, so give the
            // generation + tool loop extra headroom to avoid truncating mid-reasoning.
            // Liquid's Thinking recommendation adds top_p 0.1 on top of the instruct params;
            // without it the reasoning pass meanders and tool calls come out malformed.
            return .init(maxTokens: 8192, temperature: 0.1, topP: 0.1, topK: 50, repetitionPenalty: 1.05)
        }
        return .init(maxTokens: 4096, temperature: 0.1, topK: 50, repetitionPenalty: 1.05)
    }

    /// True for the qwen3_5-family reasoning VLMs (Ornith, Qwen3.6) - they share the qwen wire
    /// codec and the larger context cap. Keyed off the repo id because `config.json`'s `model_type`
    /// isn't read until the container loads, after these choices are made.
    private var isQwenFamily: Bool { id.contains("Ornith") || id.contains("Qwen3.6") }

    /// True for the gemma4-family models (Gemma 4 E4B) - the gemma wire codec (thought channel +
    /// library-parsed `<|tool_call>` calls). Keyed off the repo id like ``isQwenFamily``.
    private var isGemmaFamily: Bool { id.contains("gemma-4") }

    /// The model's context window in tokens — what summarization's 85% trigger and the context
    /// meter measure against. The LFM2.5 family ships a 32k window (matching the on-device budget the
    /// ReAct loop already assumes). The qwen3_5 family (Ornith, Qwen3.6) and Gemma 4 are trained for
    /// far more (262k / 128k), but on-device memory is the real limit, so they get a conservative
    /// 40k - enough headroom for long reasoning passes without letting the context grow past what
    /// the device holds.
    public var contextWindowTokens: Int { isQwenFamily || isGemmaFamily ? 40960 : 32768 }

    /// Which on-device message codec drives this model in ``RebuildTurnSession`` - the LFM2 wire
    /// format (Pythonic `<|tool_call_start|>` tags, parsed in-app), the qwen3_5 format (`<think>` +
    /// library-parsed `<tool_call>` XML) shared by Ornith and Qwen3.6, or the gemma4 format
    /// (`<|channel>thought` + library-parsed `<|tool_call>` calls).
    public var codecFamily: MlxCodecFamily {
        if isGemmaFamily { return .gemma4 }
        return isQwenFamily ? .qwen35 : .lfm2
    }

    public var sizeLabel: String {
        approxGB >= 1 ? String(format: "%.1f GB", approxGB) : String(format: "%.0f MB", approxGB * 1024)
    }

    /// Compact name for tight chrome (header pills), e.g. "1.2B Instruct".
    public var shortName: String {
        displayName
            .replacingOccurrences(of: "LFM2.5 ", with: "")
            .replacingOccurrences(of: "LFM2.5-", with: "")
    }

    public static let catalog: [MlxModel] = [
        MlxModel(id: "LiquidAI/LFM2.5-350M-MLX-8bit",
                 displayName: "LFM2.5 350M", detail: "Instruct · 8-bit", kind: .language, approxGB: 0.4),
        MlxModel(id: "LiquidAI/LFM2.5-1.2B-Instruct-MLX-8bit",
                 displayName: "LFM2.5 1.2B Instruct", detail: "Instruct · 8-bit", kind: .language, approxGB: 1.3),
        MlxModel(id: "LiquidAI/LFM2.5-1.2B-Instruct-MLX-bf16",
                 displayName: "LFM2.5 1.2B Instruct", detail: "Instruct · bf16", kind: .language, approxGB: 2.5),
        MlxModel(id: "LiquidAI/LFM2.5-1.2B-Thinking-MLX-8bit",
                 displayName: "LFM2.5 1.2B Thinking", detail: "Thinking · 8-bit", kind: .language, approxGB: 1.3),
        MlxModel(id: "LiquidAI/LFM2.5-1.2B-Thinking-MLX-bf16",
                 displayName: "LFM2.5 1.2B Thinking", detail: "Thinking · bf16", kind: .language, approxGB: 2.4),
        MlxModel(id: "LiquidAI/LFM2.5-8B-A1B-MLX-8bit",
                 displayName: "LFM2.5 8B-A1B", detail: "MoE · 8-bit · large", kind: .language, approxGB: 9.0),
        // Ornith-1.0-9B (deepreinforce-ai) is a qwen3_5 reasoning VLM. Cataloged `.language` so it
        // drives the planner with tools + <think>, but `acceptsImages` lets it also load through the
        // VLM factory and back the vision subagent - one repo serving both roles.
        MlxModel(id: "mlx-community/Ornith-1.0-9B-4bit",
                 displayName: "Ornith 1.0 9B", detail: "Reasoning + Vision · 4-bit · large",
                 kind: .language, approxGB: 5.2, acceptsImages: true),
        MlxModel(id: "mlx-community/Ornith-1.0-9B-8bit",
                 displayName: "Ornith 1.0 9B", detail: "Reasoning + Vision · 8-bit · large",
                 kind: .language, approxGB: 9.6, acceptsImages: true),
        // Qwen3.6 (qwen3_5 / qwen3_5_moe) reasoning planners. Text-only on purpose, unlike Ornith:
        // the OptiQ conversions ship no processor configs (images couldn't be prepared), and their
        // sidecar weights (`mtp.safetensors`, `optiq_vision.safetensors`) are only dropped by the
        // *LLM* path's sanitize - the VLM path dies on `Unhandled keys ["mtp"]`. So they load
        // through the text factory and never list in the vision picker. OptiQ per-layer bit
        // overrides load via `perLayerQuantization`. The sidecars must also be hidden from the
        // loader entirely (`MlxModelLoader.sidecarFilteredSnapshot`): stray `mtp.` keys trip the
        // qwen3_5 norm-shift heuristic and corrupt every layer norm into gibberish.
        MlxModel(id: "mlx-community/Qwen3.6-27B-OptiQ-4bit",
                 displayName: "Qwen3.6 27B", detail: "Reasoning · OptiQ 4-bit · large",
                 kind: .language, approxGB: 20.0),
        MlxModel(id: "mlx-community/Qwen3.6-35B-A3B-OptiQ-4bit",
                 displayName: "Qwen3.6 35B A3B", detail: "MoE Reasoning · OptiQ 4-bit · large",
                 kind: .language, approxGB: 24.7),
        // Gemma 4 E4B (gemma4): a reasoning VLM, text-only *for now*. The 8-bit conversion ships
        // full processor configs, but mlx-swift-lm 3.31.4's MLXVLM Gemma4 cannot load E-series
        // checkpoints: the backbone builds k/v projections for every layer while its sanitize
        // drops them for the `num_kv_shared_layers` tail, so `verify: .all` always fails
        // (upstream #338, fix pending in #384). The text (LLM) path constructs KV-shared layers
        // correctly, so both rows load through it. Flip this row to `acceptsImages: true` for
        // Ornith-style dual-role duty once the upstream fix ships.
        MlxModel(id: "mlx-community/gemma-4-e4b-it-8bit",
                 displayName: "Gemma 4 E4B", detail: "Reasoning · 8-bit · large",
                 kind: .language, approxGB: 9.0),
        // The OptiQ conversion is text-only *by construction*, like the Qwen3.6 OptiQ rows: it
        // ships no processor configs (images couldn't be prepared) and carries an
        // `optiq_vision.safetensors` sidecar outside the index, so it loads through the text
        // factory behind `MlxModelLoader.sidecarFilteredSnapshot` and never lists in the vision
        // picker - regardless of the upstream VLM-path fix above.
        MlxModel(id: "mlx-community/gemma-4-e4b-it-OptiQ-4bit",
                 displayName: "Gemma 4 E4B", detail: "Reasoning · OptiQ 4-bit · large",
                 kind: .language, approxGB: 7.5),
        MlxModel(id: "LiquidAI/LFM2.5-VL-450M-MLX-8bit",
                 displayName: "LFM2.5-VL 450M", detail: "Vision · 8-bit", kind: .vision, approxGB: 0.6),
        MlxModel(id: "LiquidAI/LFM2.5-VL-450M-MLX-bf16",
                 displayName: "LFM2.5-VL 450M", detail: "Vision · bf16", kind: .vision, approxGB: 1.0),
        MlxModel(id: "mlx-community/LFM2.5-VL-1.6B-8bit",
                 displayName: "LFM2.5-VL 1.6B", detail: "Vision · 8-bit", kind: .vision, approxGB: 2.1)
    ]

    /// Text (instruct) models only — used by the Translate model picker (translation runs
    /// on a language model, never a VLM).
    public static var languageCatalog: [MlxModel] { catalog.filter { !$0.isVision } }
}
