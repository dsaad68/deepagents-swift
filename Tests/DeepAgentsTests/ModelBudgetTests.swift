@testable import DeepAgentsMLX
import Testing

/// The catalog's context windows and generation budgets, pinned to what each model card documents.
///
/// These are matched on repo-id substrings in an ordered if-chain, so a new row silently inherits
/// whatever branch it happens to match - or the fall-through default. That is exactly how
/// LFM2.5-8B-A1B ended up declaring a 32k window and a 4k output budget when its card documents
/// 128k and 8192: it is a reasoning model, but its branch sat above every reasoning branch and
/// nothing checked. Each expectation below quotes the card it came from.
struct ModelBudgetTests {
    /// Every distinct base model in the catalog, with the window its card documents.
    private static let cardWindows: [(id: String, window: Int, card: String)] = [
        ("LiquidAI/LFM2.5-350M-MLX-8bit", 32768, "Context length: 32,768 tokens"),
        ("LiquidAI/LFM2.5-1.2B-Instruct-MLX-bf16", 32768, "Context length: 32,768 tokens"),
        ("LiquidAI/LFM2.5-1.2B-Thinking-MLX-bf16", 32768, "Context length: 32,768 tokens"),
        ("LiquidAI/LFM2.5-2.6B-MLX/8bit", 131_072, "Context length: 131,072 tokens"),
        ("LiquidAI/LFM2.5-8B-A1B-MLX-8bit", 128_000, "Context length: 128,000 / config.json 128000"),
        ("LiquidAI/LFM2.5-VL-450M-MLX-8bit", 32768, "Context length: 32,768 tokens"),
        ("mlx-community/LFM2.5-VL-1.6B-8bit", 32768, "Context length: 32,768 tokens"),
        ("mlx-community/Ornith-1.0-9B-4bit", 262_144, "--max-model-len 262144"),
        ("mlx-community/Qwen3.6-27B-OptiQ-4bit", 262_144, "262,144 natively"),
        ("mlx-community/Qwen3.6-35B-A3B-OptiQ-4bit", 262_144, "262,144 natively"),
        ("mlx-community/gemma-4-e4b-it-8bit", 128_000, "The small models feature a 128K context window")
    ]

    private func model(_ id: String) -> MlxModel? {
        MlxModel.catalog.first { $0.id == id }
    }

    @Test func everyContextWindowMatchesItsModelCard() throws {
        for expected in Self.cardWindows {
            let row = try #require(model(expected.id), "catalog is missing \(expected.id)")
            #expect(
                row.contextWindowTokens == expected.window,
                "\(expected.id) should report \(expected.window) - card says \"\(expected.card)\""
            )
        }
    }

    /// The window is the card's, not a pre-shrunk one. Keeping a session inside what the hardware
    /// can carry is the compaction threshold's job, so a model must never under-report its window -
    /// that made the context meter and the summarization trigger lie about the model.
    @Test func noModelUnderReportsItsWindow() throws {
        for expected in Self.cardWindows {
            let row = try #require(model(expected.id))
            #expect(row.contextWindowTokens >= expected.window)
        }
    }

    /// A reasoning model spends tokens on its `<think>` pass before it reaches the answer, so a
    /// budget sized for a non-reasoning model truncates it mid-thought and the turn arrives with no
    /// answer at all. Every model that reasons gets at least the reasoning budget.
    @Test func everyReasoningModelGetsTheReasoningBudget() throws {
        let reasoning = [
            "LiquidAI/LFM2.5-1.2B-Thinking-MLX-bf16", // "Thinking" variant
            "LiquidAI/LFM2.5-2.6B-MLX/8bit", // "always thinks before it answers"
            "LiquidAI/LFM2.5-8B-A1B-MLX-8bit", // "explicit chain of thought before the final answer"
            "mlx-community/Ornith-1.0-9B-4bit", // "the assistant turn opens with a <think> block"
            "mlx-community/Qwen3.6-27B-OptiQ-4bit", // "thinking mode by default"
            "mlx-community/gemma-4-e4b-it-8bit" // thought channel via enable_thinking
        ]
        for id in reasoning {
            let row = try #require(model(id))
            #expect(row.agentParameters.maxTokens ?? 0 >= 8192, "\(id) reasons and needs the budget")
        }
    }

    /// The regression this file exists for: 8B-A1B is reasoning-tuned with a 128k window, and used
    /// to declare 32k/4096 because its branch sat above every reasoning branch in the chain.
    @Test func theEightBA1BIsTreatedAsTheReasoningModelItIs() throws {
        let row = try #require(model("LiquidAI/LFM2.5-8B-A1B-MLX-8bit"))
        #expect(row.contextWindowTokens == 128_000)
        #expect(row.agentParameters.maxTokens == 8192) // card: `max_new_tokens=8192`
    }

    /// Qwen3.6 asks for 32,768 output tokens "for most queries", reserving 81,920 for competition
    /// math and coding. The budget was argued down from the 81,920 figure, which is the extreme case.
    @Test func qwenGetsTheEverydayOutputBudgetNotTheExtremeOne() throws {
        for id in ["mlx-community/Qwen3.6-27B-OptiQ-4bit", "mlx-community/Qwen3.6-35B-A3B-OptiQ-4bit"] {
            let row = try #require(model(id))
            #expect(row.agentParameters.maxTokens == 32768)
        }
    }

    /// Only the 35B-A3B card recommends presence penalty 1.5 for thinking mode; the 27B's is 0.0.
    @Test func onlyTheMoEQwenCarriesAPresencePenalty() throws {
        let moe = try #require(model("mlx-community/Qwen3.6-35B-A3B-OptiQ-4bit"))
        let dense = try #require(model("mlx-community/Qwen3.6-27B-OptiQ-4bit"))
        #expect(moe.agentParameters.presencePenalty == 1.5)
        #expect(dense.agentParameters.presencePenalty == nil)
    }
}
