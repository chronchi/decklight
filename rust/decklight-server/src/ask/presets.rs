//! Provider presets (config/ask/providers.v1.json, compiled in): where each provider's API lives, where its user gets
//! a key, and whether it wants a per-chat session header.

use serde::Deserialize;

pub const PROVIDERS_JSON: &str = include_str!("../../../../config/ask/providers.v1.json");

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct Preset {
    pub id: String,
    pub name: String,
    pub base_url: String,
    pub default_model: String,
    pub keys_url: String,
    pub session_header: bool,
    pub needs_key: bool,
    pub note: String,
}

#[derive(Deserialize)]
struct File {
    presets: Vec<Preset>,
}

pub fn presets() -> Vec<Preset> {
    serde_json::from_str::<File>(PROVIDERS_JSON).expect("config/ask/providers.v1.json is valid").presets
}

pub fn preset(id: &str) -> Option<Preset> {
    presets().into_iter().find(|p| p.id == id)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn presets_load_with_valid_unique_ids() {
        let all = presets();
        let ids: Vec<&str> = all.iter().map(|p| p.id.as_str()).collect();
        assert_eq!(ids, vec!["opencode-go", "opencode-zen", "openrouter", "custom", "device"]);
        assert!(ids.iter().all(|id| super::super::keys::valid_preset(id)));
        assert!(all.iter().filter(|p| p.needs_key).all(|p| p.base_url.starts_with("https://") && !p.keys_url.is_empty()));
        assert!(preset("opencode-go").unwrap().session_header);
    }
}
