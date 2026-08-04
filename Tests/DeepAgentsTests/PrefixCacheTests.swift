@testable import DeepAgentsMLX
import Testing

/// The prefix KV cache reuses a model's KV up to the longest *token* prefix shared with the previous
/// prompt, then re-prefills only the suffix (`RebuildTurnSession.makeIterator`). The reuse boundary
/// is `commonPrefixLength`; the trim/offset arithmetic and end-to-end correctness are exercised by
/// the multi-round integration runs (they need a resident model).
struct PrefixCacheTests {
    @Test func commonPrefixLengthFindsTheSharedRun() {
        // Diverge partway: system+tools shared, conversation differs.
        #expect(RebuildTurnSession.commonPrefixLength([1, 2, 3, 4], [1, 2, 3, 9]) == 3)
        // Identical prompts (e.g. a retried round with no new tokens).
        #expect(RebuildTurnSession.commonPrefixLength([1, 2, 3], [1, 2, 3]) == 3)
        // One prompt is a strict prefix of the other (appended-only history).
        #expect(RebuildTurnSession.commonPrefixLength([1, 2, 3], [1, 2, 3, 4, 5]) == 3)
        #expect(RebuildTurnSession.commonPrefixLength([1, 2, 3, 4, 5], [1, 2, 3]) == 3)
    }

    @Test func commonPrefixLengthIsZeroWhenNothingShared() {
        #expect(RebuildTurnSession.commonPrefixLength([9, 1], [1, 9]) == 0)
        #expect(RebuildTurnSession.commonPrefixLength([], [1, 2]) == 0)
        #expect(RebuildTurnSession.commonPrefixLength([1, 2], []) == 0)
        #expect(RebuildTurnSession.commonPrefixLength([], []) == 0)
    }

    @Test func strictPrefixRequiresFullMatchAndRoomToGenerate() {
        // The snapshot's tokens must ALL match and the prompt must extend past them (there must be
        // at least one token left to feed the iterator).
        #expect(RebuildTurnSession.isStrictPrefix([1, 2], of: [1, 2, 3]))
        #expect(!RebuildTurnSession.isStrictPrefix([1, 2, 3], of: [1, 2, 3])) // equal: nothing left to feed
        #expect(!RebuildTurnSession.isStrictPrefix([1, 9], of: [1, 2, 3])) // diverges mid-prefix
        #expect(!RebuildTurnSession.isStrictPrefix([1, 2, 3, 4], of: [1, 2, 3])) // longer than prompt
        #expect(!RebuildTurnSession.isStrictPrefix([], of: [1, 2, 3])) // empty snapshot is useless
    }
}
