@testable import DeepAgents
@testable import DeepAgentsMacTools
@testable import DeepAgentsMLX
import Testing

/// The qwen3_5-family models (Ornith, Qwen3.6) are unified VLMs cataloged as language models, so
/// these pin the capability/codec wiring that lets one repo serve both the planner and the vision
/// subagent and select the qwen3_5 codec.
struct OrnithCatalogTests {
    /// Every qwen3_5-family repo in the catalog (Ornith quants + the Qwen3.6 models).
    private static let qwenFamilyIDs = [
        "mlx-community/Ornith-1.0-9B-4bit",
        "mlx-community/Ornith-1.0-9B-8bit",
        "mlx-community/Qwen3.6-27B-OptiQ-4bit",
        "mlx-community/Qwen3.6-35B-A3B-OptiQ-4bit"
    ]

    private func model(_ id: String) throws -> MlxModel {
        try #require(MlxModel.catalog.first { $0.id == id })
    }

    @Test func allQwenFamilyModelsAreCatalogedAsLanguage() throws {
        // Cataloged `.language` so they list in the planner picker and drive the ReAct loop.
        for id in Self.qwenFamilyIDs {
            #expect(try model(id).kind == .language)
        }
    }

    @Test func qwenFamilySelectsTheQwen35Codec() throws {
        for id in Self.qwenFamilyIDs {
            #expect(try model(id).codecFamily == .qwen35)
        }
        // Every other model stays on the LFM2 codec - except the gemma4 family, pinned in
        // `Gemma4CatalogTests`.
        for model in MlxModel.catalog
            where !Self.qwenFamilyIDs.contains(model.id) && !model.id.contains("gemma-4") {
            #expect(model.codecFamily == .lfm2)
        }
    }

    @Test func ornithAcceptsImagesButStaysALanguageModel() throws {
        for id in Self.qwenFamilyIDs where id.contains("Ornith") {
            let model = try model(id)
            #expect(model.acceptsImages) // can back the vision subagent / take screenshots
            #expect(!model.isVision) // but classified language → planner picker, AskAgent
            #expect(model.loadsAsVision) // loads through the VLM factory so images work
        }
    }

    @Test func qwen36ModelsAreTextOnly() throws {
        // The OptiQ conversions ship no processor configs and their sidecar weights (mtp /
        // optiq_vision) only load through the LLM path's sanitize - so they must NOT route through
        // the VLM factory (see `MlxModel.catalog`).
        for id in Self.qwenFamilyIDs where id.contains("Qwen3.6") {
            let model = try model(id)
            #expect(!model.acceptsImages)
            #expect(!model.isVision)
            #expect(!model.loadsAsVision) // loads through the text (LLM) factory
        }
    }

    @Test func qwenFamilyPickerPlacement() throws {
        // All are planners; only Ornith doubles as a vision-subagent option.
        for id in Self.qwenFamilyIDs {
            #expect(MlxModel.languageCatalog.contains { $0.id == id })
            #expect(MlxModel.catalog.filter(\.acceptsImages).contains { $0.id == id } == id.contains("Ornith"))
        }
    }

    @Test func lfm2CapabilityFlagsAreUnchanged() throws {
        // A dedicated VLM still loads as vision; a plain instruct model takes no images.
        let vl = try model("mlx-community/LFM2.5-VL-1.6B-8bit")
        #expect(vl.isVision)
        #expect(vl.acceptsImages)
        #expect(vl.loadsAsVision)

        let instruct = try model("LiquidAI/LFM2.5-1.2B-Instruct-MLX-bf16")
        #expect(!instruct.isVision)
        #expect(!instruct.acceptsImages)
        #expect(!instruct.loadsAsVision)
    }
}
