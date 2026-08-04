@testable import DeepAgents
@testable import DeepAgentsMacTools
@testable import DeepAgentsMLX
import Testing

/// The gemma4-family rows (Gemma 4 E4B) are text-only planners for now: the OptiQ conversion by
/// construction (no processor configs, an `optiq_vision.safetensors` sidecar outside the index -
/// the Qwen3.6 OptiQ treatment), and the 8-bit because mlx-swift-lm 3.31.4's MLXVLM Gemma4 path
/// cannot load E-series checkpoints (KV-shared layers; upstream #338/#384) - it goes Ornith-style
/// dual-role once that fix ships. These pin the capability/codec wiring.
struct Gemma4CatalogTests {
    private static let gemmaFamilyIDs = [
        "mlx-community/gemma-4-e4b-it-8bit",
        "mlx-community/gemma-4-e4b-it-OptiQ-4bit"
    ]

    private func model(_ id: String) throws -> MlxModel {
        try #require(MlxModel.catalog.first { $0.id == id })
    }

    @Test func gemmaFamilyModelsAreCatalogedAsLanguage() throws {
        // Cataloged `.language` so they list in the planner picker and drive the ReAct loop.
        for id in Self.gemmaFamilyIDs {
            #expect(try model(id).kind == .language)
        }
    }

    @Test func gemmaFamilySelectsTheGemma4Codec() throws {
        for id in Self.gemmaFamilyIDs {
            #expect(try model(id).codecFamily == .gemma4)
        }
        // No other catalog model picks up the gemma4 codec.
        for model in MlxModel.catalog where !Self.gemmaFamilyIDs.contains(model.id) {
            #expect(model.codecFamily != .gemma4)
        }
    }

    @Test func gemmaModelsAreTextOnly() throws {
        // Both must route through the text (LLM) factory: the VLM path can't load E-series
        // checkpoints at mlx-swift-lm 3.31.4 (its backbone builds k/v projections for the
        // KV-shared tail layers that its own sanitize then drops - verify always fails), and the
        // OptiQ conversion additionally ships no processor configs. When the upstream fix lands,
        // the 8-bit row flips to `acceptsImages: true` and this test changes with it.
        for id in Self.gemmaFamilyIDs {
            let model = try model(id)
            #expect(!model.acceptsImages)
            #expect(!model.isVision)
            #expect(!model.loadsAsVision) // text factory, never the VLM path
        }
    }

    @Test func gemmaFamilyPickerPlacement() throws {
        // Planners only - neither lists as a vision-subagent option (yet).
        for id in Self.gemmaFamilyIDs {
            #expect(MlxModel.languageCatalog.contains { $0.id == id })
            #expect(!MlxModel.catalog.filter(\.acceptsImages).contains { $0.id == id })
        }
    }
}
