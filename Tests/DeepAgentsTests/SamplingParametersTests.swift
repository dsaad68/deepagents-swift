@testable import DeepAgents
@testable import DeepAgentsMacTools
@testable import DeepAgentsMLX
import MLXLMCommon
import Testing

/// Regression tests pinning each model family's sampling to Liquid's published
/// recommendations — drifting from them is exactly how tool calling degrades (the docs
/// recommend near-greedy decoding for tool use; the Thinking model additionally needs
/// `top_p 0.1` or its reasoning pass meanders into malformed calls).
struct SamplingParametersTests {
    private func model(_ id: String) -> MlxModel? {
        MlxModel.catalog.first { $0.id == id }
    }

    @Test func instructModelsUseLiquidRecommendedNearGreedySampling() throws {
        let instruct = try #require(model("LiquidAI/LFM2.5-1.2B-Instruct-MLX-bf16"))
        let parameters = instruct.agentParameters
        #expect(parameters.temperature == 0.1)
        #expect(parameters.topK == 50)
        #expect(parameters.repetitionPenalty == 1.05)
    }

    @Test func thinkingModelAddsTopP() throws {
        let thinking = try #require(model("LiquidAI/LFM2.5-1.2B-Thinking-MLX-bf16"))
        let parameters = thinking.agentParameters
        #expect(parameters.temperature == 0.1)
        #expect(parameters.topP == 0.1) // the Thinking-specific recommendation
        #expect(parameters.topK == 50)
        #expect(parameters.repetitionPenalty == 1.05)
        // Reasoning + tool loop need extra headroom over the instruct models.
        #expect(parameters.maxTokens == 8192)
    }

    @Test func moeModelUsesItsModelCardSettings() throws {
        let moe = try #require(model("LiquidAI/LFM2.5-8B-A1B-MLX-8bit"))
        let parameters = moe.agentParameters
        // The 8B-A1B model card recommends hotter sampling than the 1.2B family.
        #expect(parameters.temperature == 0.2)
        #expect(parameters.topK == 80)
        #expect(parameters.repetitionPenalty == 1.05)
    }

    @Test func visionModelsKeepTheRepetitionPenalty() throws {
        // VL models used to omit the penalty to dodge the TokenRing 2-D-prompt crash; that was
        // fixed upstream (mlx-swift-lm 3.31.4, #170), so they follow Liquid's 1.05 again.
        let vision = try #require(model("mlx-community/LFM2.5-VL-1.6B-8bit"))
        #expect(vision.agentParameters.repetitionPenalty == 1.05)
    }

    @Test func ornithUsesCardSampling() throws {
        let ornith = try #require(model("mlx-community/Ornith-1.0-9B-8bit"))
        let parameters = ornith.agentParameters
        // Ornith's card recommends temp 0.6 / top-p 0.95 / top-k 20 (hotter than the LFM2 family).
        #expect(parameters.temperature == 0.6)
        #expect(parameters.topP == 0.95)
        #expect(parameters.topK == 20)
        // Restored after the upstream TokenRing 2-D fix (see `visionModelsKeepTheRepetitionPenalty`).
        #expect(parameters.repetitionPenalty == 1.05)
        // Reasoning + tool loop need the larger budget.
        #expect(parameters.maxTokens == 8192)
    }

    @Test func qwen36ModelsUseCardSampling() throws {
        // Both Qwen3.6 cards: temperature 1.0 / top-p 0.95 / top-k 20, no repetition penalty.
        for id in ["mlx-community/Qwen3.6-27B-OptiQ-4bit", "mlx-community/Qwen3.6-35B-A3B-OptiQ-4bit"] {
            let parameters = try #require(model(id)).agentParameters
            #expect(parameters.temperature == 1.0)
            #expect(parameters.topP == 0.95)
            #expect(parameters.topK == 20)
            #expect(parameters.repetitionPenalty == nil)
            #expect(parameters.maxTokens == 8192)
        }
        // Only the 35B-A3B card adds presence_penalty 1.5 (the 27B card says 0.0).
        let moe = try #require(model("mlx-community/Qwen3.6-35B-A3B-OptiQ-4bit"))
        #expect(moe.agentParameters.presencePenalty == 1.5)
        let dense = try #require(model("mlx-community/Qwen3.6-27B-OptiQ-4bit"))
        #expect(dense.agentParameters.presencePenalty == nil)
    }

    @Test func gemma4ModelsUseCardSampling() throws {
        // Gemma 4 E4B card + shipped generation_config.json: temperature 1.0 / top-p 0.95 /
        // top-k 64, no penalties; the thought channel + tool loop get the reasoning budget.
        for id in ["mlx-community/gemma-4-e4b-it-8bit", "mlx-community/gemma-4-e4b-it-OptiQ-4bit"] {
            let parameters = try #require(model(id)).agentParameters
            #expect(parameters.temperature == 1.0)
            #expect(parameters.topP == 0.95)
            #expect(parameters.topK == 64)
            #expect(parameters.repetitionPenalty == nil)
            #expect(parameters.presencePenalty == nil)
            #expect(parameters.maxTokens == 8192)
        }
    }

    @Test func qwenFamilyReportsTheLargerContextWindow() throws {
        for id in [
            "mlx-community/Ornith-1.0-9B-4bit",
            "mlx-community/Qwen3.6-27B-OptiQ-4bit",
            "mlx-community/Qwen3.6-35B-A3B-OptiQ-4bit",
            "mlx-community/gemma-4-e4b-it-8bit",
            "mlx-community/gemma-4-e4b-it-OptiQ-4bit"
        ] {
            #expect(try #require(model(id)).contextWindowTokens == 40960)
        }
        // The LFM2 family is unchanged.
        let lfm = try #require(model("LiquidAI/LFM2.5-1.2B-Instruct-MLX-bf16"))
        #expect(lfm.contextWindowTokens == 32768)
    }
}
