//! Pure helpers for the embedded per-unit asset archives.
//!
//! `build.rs` writes the `UNITS` table (one archive per unit, ids from `scripts/app/build_app_assets.py`); `lib.rs`
//! unpacks only the stale units and serves the client statically meanwhile. This module stays dependency-free so its
//! unit tests run on the laptop with `rustc --test` (the app crate itself only builds for Android).

pub struct Unit {
    pub name: &'static str,
    pub id: &'static str,
    /// Paths this unit owns, removed before it is unpacked again (a tree, or the files of the FILES unit).
    pub remove: &'static [&'static str],
    pub archive: &'static [u8],
}

/// Units whose on-disk marker is missing or holds another id. `migrated` is the one-time upgrade from the old
/// single archive (`.assets_id`, no `.assets/` markers): every unit is stale once then.
pub fn stale_units<'a>(
    units: &'a [Unit],
    marker: impl Fn(&str) -> Option<String>,
    migrated: bool,
) -> Vec<&'a Unit> {
    units
        .iter()
        .filter(|u| migrated || marker(u.name).as_deref().map(str::trim) != Some(u.id))
        .collect()
}

/// A request path relative to the web root that cannot escape it (the client-only static server).
pub fn safe_rel(rel: &str) -> bool {
    !rel.is_empty() && !rel.split('/').any(|p| p.is_empty() || p == "." || p == "..")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit(name: &'static str, id: &'static str) -> Unit {
        Unit { name, id, remove: &[], archive: &[] }
    }

    fn markers(items: &[(&'static str, &'static str)]) -> impl Fn(&str) -> Option<String> {
        let items: Vec<_> = items.iter().map(|(n, id)| (*n, id.to_string())).collect();
        move |name| items.iter().find(|(n, _)| *n == name).map(|(_, id)| id.clone())
    }

    #[test]
    fn matching_units_are_skipped() {
        let units = [unit("web", "aaa"), unit("onnx", "bbb")];
        let stale = stale_units(&units, markers(&[("web", "aaa"), ("onnx", "bbb")]), false);
        assert!(stale.is_empty());
    }

    #[test]
    fn changed_or_missing_markers_are_stale() {
        let units = [unit("web", "aaa"), unit("onnx", "bbb"), unit("decks-pool", "ccc")];
        let stale = stale_units(&units, markers(&[("web", "old"), ("onnx", "bbb")]), false);
        let names: Vec<_> = stale.iter().map(|u| u.name).collect();
        assert_eq!(names, ["web", "decks-pool"]);
    }

    #[test]
    fn migration_makes_every_unit_stale() {
        let units = [unit("web", "aaa"), unit("onnx", "bbb")];
        let stale = stale_units(&units, markers(&[("web", "aaa"), ("onnx", "bbb")]), true);
        assert_eq!(stale.len(), 2);
    }

    #[test]
    fn marker_values_are_trimmed() {
        let units = [unit("web", "aaa")];
        assert!(stale_units(&units, markers(&[("web", "aaa\n")]), false).is_empty());
    }

    #[test]
    fn static_paths_cannot_escape_the_root() {
        assert!(safe_rel("index.html"));
        assert!(safe_rel("render/board.js"));
        assert!(safe_rel("assets/boot-mark.svg"));
        for bad in ["", "/", "..", "../secret", "web/../../x", "./x", "a//b"] {
            assert!(!safe_rel(bad), "{bad:?} must be rejected");
        }
    }
}
