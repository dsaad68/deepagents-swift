import Foundation

/// Splitting a catalog id into the Hugging Face repo it lives in and, when a conversion ships
/// several precisions as subfolders of one repo (`LiquidAI/LFM2.5-2.6B-MLX/mxfp4`), the subfolder
/// holding this one.
///
/// ``MlxModel/id`` stays the unique, opaque selection key everywhere - it is the SwiftUI row
/// identity in the app's local-models list, the key of every residency dictionary in
/// ``MlxModelLoader`` and the app's `MlxModelManager`, the KV-cache filename stem in
/// ``PrefixKVStore``, and the value persisted as `selectedModel` / passed to `--model`. Only the
/// handful of places that talk to the Hub or the HF cache ever take it apart.
///
/// The split is deliberately strict. Upstream's `Repo.ID(rawValue:)` splits on the *first* slash
/// only, so handing it a three-component id does not fail validation - it silently yields the repo
/// `LiquidAI/LFM2.5-2.6B-MLX/mxfp4`, which 404s at the API and writes a bogus cache directory.
/// Anything that isn't exactly `org/repo/subfolder` with non-empty, non-relative components is
/// therefore treated as a plain repo id rather than guessed at.
public enum MlxModelID {
    /// `id` split into its repo and, if it carries one, its precision subfolder.
    public static func split(_ id: String) -> (repo: String, subfolder: String?) {
        let parts = id.split(separator: "/", omittingEmptySubsequences: false).map(String.init)
        guard parts.count == 3, parts.allSatisfy({ !$0.isEmpty && $0 != "." && $0 != ".." })
        else { return (id, nil) }
        return ("\(parts[0])/\(parts[1])", parts[2])
    }

    /// The Hugging Face repo id to download from - the whole id unless it carries a subfolder.
    public static func repo(_ id: String) -> String { split(id).repo }

    /// The precision subfolder inside ``repo(_:)``, or nil for a one-precision-per-repo model.
    public static func subfolder(_ id: String) -> String? { split(id).subfolder }
}

public extension MlxModel {
    /// The Hugging Face repo this model's files live in (``id`` minus any precision subfolder).
    var repoId: String { MlxModelID.repo(id) }

    /// The subfolder of ``repoId`` holding this precision, or nil when the repo *is* the precision.
    var subfolder: String? { MlxModelID.subfolder(id) }
}
