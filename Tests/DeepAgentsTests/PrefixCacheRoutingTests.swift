@testable import DeepAgents
@testable import DeepAgentsMacTools
@testable import DeepAgentsMLX
import MLXLMCommon
import Testing

/// The prefix cache has two reuse strategies, and picking the wrong one is not an error - it is
/// silent prompt corruption. `supportsTrimReuse` decides: only caches that can *safely* rewind
/// take the trimming path; everything else (recurrent hybrids, sliding-window rotating caches)
/// resumes from `copy()`-snapshots. Cache construction is pure bookkeeping (no Metal), so this
/// is unit-testable even though generation is not.
struct PrefixCacheRoutingTests {
    @Test func rotatingCachesNeverTakeTheTrimmingPath() {
        // Gemma 4's hybrid stack: standard caches for the global layers, rotating for the
        // sliding-window layers. The rotating entries must force the snapshot path.
        let gemmaLike: [KVCache] = [
            KVCacheSimple(), RotatingKVCache(maxSize: 512, keep: 0),
            RotatingKVCache(maxSize: 512, keep: 0), KVCacheSimple()
        ]
        #expect(!RebuildTurnSession.supportsTrimReuse(gemmaLike))

        // A pure standard-attention stack is the one shape that may trim.
        #expect(RebuildTurnSession.supportsTrimReuse([KVCacheSimple(), KVCacheSimple()]))
    }

    /// Pins the upstream behavior that makes the naive `allSatisfy(\.isTrimmable)` probe unsafe:
    /// a *fresh* rotating cache answers trimmable (offset < window), so probing `newCache` would
    /// route Gemma 4 to the trimming path - where post-rotation `trim` falsely reports success.
    /// If a dependency bump changes this answer, revisit `supportsTrimReuse` (upstream may have
    /// made rotating caches honest about trimming).
    @Test func freshRotatingCacheClaimsToBeTrimmable() {
        #expect(RotatingKVCache(maxSize: 512, keep: 0).isTrimmable)
    }
}
