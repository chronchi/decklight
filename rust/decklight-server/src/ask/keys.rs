//! API keys for the Ask providers (docs/plans/ask-tab.md §Decisions 2, user: "I shouldnt be able to check it, i dont
//! want to just store as a plain file because thats a liability"). One key per provider preset, kept by the OS
//! credential store: Secret Service on Linux, Keychain on macOS, Credential Manager on Windows, and on Android a
//! non-exportable AES-GCM key in the Android Keystore encrypting the key into an app-private SharedPreferences file
//! (excluded from backup). Write-only from the UI: the server uses a key, the client only ever sees a hint
//! (`••••1a2b`). No credential store (a headless box) → the key is kept for this session only, and the UI says so.
//!
//! The store is opened on first use, not at startup: a locked keychain may prompt, and only Ask needs it.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::{Arc, Mutex, OnceLock};

/// The credential store's service name; each preset id is a user name under it.
pub const SERVICE: &str = "decklight-ask";

/// Where keys live. Errors are messages for the user that say what to do (fail loud with an action).
pub trait SecretStore: Send + Sync + fmt::Debug {
    /// For the settings screen: "Secret Service", "Android Keystore", "this session only".
    fn kind(&self) -> String;
    /// False when keys are lost when the app stops.
    fn persistent(&self) -> bool;
    fn set(&self, preset: &str, key: &str) -> Result<(), String>;
    fn get(&self, preset: &str) -> Result<Option<String>, String>;
    fn delete(&self, preset: &str) -> Result<(), String>;
}

/// Keys in memory: the fallback without a credential store, and the store for tests and the eval runner.
#[derive(Default)]
pub struct SessionOnly {
    keys: Mutex<BTreeMap<String, String>>,
    /// Why there is no credential store (shown with the setting), if that is why this store is used.
    pub reason: Option<String>,
}

// Debug shows which presets have keys, never the keys.
impl fmt::Debug for SessionOnly {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let presets: Vec<String> = self.keys.lock().map(|k| k.keys().cloned().collect()).unwrap_or_default();
        write!(f, "SessionOnly({presets:?}, reason {:?})", self.reason)
    }
}

impl SessionOnly {
    pub fn because(reason: String) -> SessionOnly {
        SessionOnly { keys: Mutex::default(), reason: Some(reason) }
    }
}

impl SecretStore for SessionOnly {
    fn kind(&self) -> String {
        "this session only".into()
    }
    fn persistent(&self) -> bool {
        false
    }
    fn set(&self, preset: &str, key: &str) -> Result<(), String> {
        self.keys.lock().unwrap().insert(preset.to_string(), key.to_string());
        Ok(())
    }
    fn get(&self, preset: &str) -> Result<Option<String>, String> {
        Ok(self.keys.lock().unwrap().get(preset).cloned())
    }
    fn delete(&self, preset: &str) -> Result<(), String> {
        self.keys.lock().unwrap().remove(preset);
        Ok(())
    }
}

/// The OS credential store through keyring-core.
pub struct Keychain {
    store: Arc<keyring_core::CredentialStore>,
    service: String,
}

impl fmt::Debug for Keychain {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Keychain({}, {})", self.store.vendor(), self.service)
    }
}

/// A keyring error as a message with an action. Never includes the secret (keyring errors do not carry it).
fn explain(e: keyring_core::Error, doing: &str) -> String {
    use keyring_core::Error as E;
    match e {
        E::NoStorageAccess(err) => {
            format!("could not {doing}: the system keychain is locked or refused access ({err}); unlock it and try again")
        }
        E::PlatformFailure(err) => format!("could not {doing}: the system keychain failed ({err}); try again"),
        // Android: a key restored from a backup of another device cannot be decrypted here.
        // The error's own text is left out: its Debug form can hold ciphertext bytes (review 2026-11-14).
        E::BadDataFormat(..) | E::BadEncoding(..) if doing.starts_with("read") => {
            "the saved key cannot be read on this device; enter the key again".to_string()
        }
        E::BadDataFormat(..) | E::BadEncoding(..) => format!("could not {doing}: the keychain holds unreadable data"),
        other => format!("could not {doing}: {other}"),
    }
}

impl Keychain {
    fn entry(&self, preset: &str, doing: &str) -> Result<keyring_core::Entry, String> {
        self.store.build(&self.service, preset, None).map_err(|e| explain(e, doing))
    }
}

impl SecretStore for Keychain {
    fn kind(&self) -> String {
        // "Secret Service store, https://crates.io/..." → "Secret Service store".
        self.store.vendor().split(',').next().unwrap_or_default().trim().to_string()
    }
    fn persistent(&self) -> bool {
        true
    }
    fn set(&self, preset: &str, key: &str) -> Result<(), String> {
        self.entry(preset, "save the key")?.set_password(key).map_err(|e| explain(e, "save the key"))
    }
    fn get(&self, preset: &str) -> Result<Option<String>, String> {
        match self.entry(preset, "read the key")?.get_password() {
            Ok(k) => Ok(Some(k)),
            Err(keyring_core::Error::NoEntry) => Ok(None),
            Err(e) => Err(explain(e, "read the key")),
        }
    }
    fn delete(&self, preset: &str) -> Result<(), String> {
        match self.entry(preset, "delete the key")?.delete_credential() {
            Ok(()) | Err(keyring_core::Error::NoEntry) => Ok(()),
            Err(e) => Err(explain(e, "delete the key")),
        }
    }
}

/// This platform's credential store.
fn system_store() -> Result<Arc<keyring_core::CredentialStore>, String> {
    #[cfg(all(unix, not(any(target_os = "macos", target_os = "ios", target_os = "android"))))]
    let store = zbus_secret_service_keyring_store::Store::new().map(|s| s as Arc<keyring_core::CredentialStore>);
    #[cfg(target_os = "macos")]
    let store = apple_native_keyring_store::keychain::Store::new().map(|s| s as Arc<keyring_core::CredentialStore>);
    #[cfg(target_os = "windows")]
    let store = windows_native_keyring_store::Store::new().map(|s| s as Arc<keyring_core::CredentialStore>);
    #[cfg(target_os = "android")]
    let store = {
        // SharedPreferences file `keyring-decklight-ask.xml` and Keystore alias of the same name (backup rules
        // exclude the file: gen/android/app/src/main/res/xml/).
        let config = std::collections::HashMap::from([("name", SERVICE)]);
        android_native_keyring_store::Store::new_with_configuration(&config)
            .map(|s| s as Arc<keyring_core::CredentialStore>)
    };
    #[cfg(target_os = "ios")]
    let store: Result<Arc<keyring_core::CredentialStore>, keyring_core::Error> =
        Err(keyring_core::Error::NotSupportedByStore("iOS".into()));
    store.map_err(|e| e.to_string())
}

/// The credential store, or keys for this session only when there is none (logged once, shown in settings).
pub fn open() -> Arc<dyn SecretStore> {
    match system_store() {
        Ok(store) => Arc::new(Keychain { store, service: SERVICE.into() }),
        Err(e) => {
            eprintln!("ask: no system keychain ({e}); API keys are kept for this session only");
            Arc::new(SessionOnly::because(e))
        }
    }
}

/// A preset id: lower-case letters, digits and dashes.
pub fn valid_preset(id: &str) -> bool {
    (1..=40).contains(&id.len()) && id.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-')
}

/// The key as the user pasted it, trimmed; an error that says what is wrong with it otherwise.
pub fn clean_key(key: &str) -> Result<String, String> {
    let k = key.trim();
    if k.is_empty() {
        return Err("the key is empty; paste the key from your provider".into());
    }
    if k.chars().any(|c| c.is_whitespace() || c.is_control()) {
        return Err("the key has spaces or line breaks in it; paste only the key".into());
    }
    // API keys are printable ASCII; this also refuses zero-width and bidi characters pasted along (review 2026-11-14).
    if !k.chars().all(|c| c.is_ascii_graphic()) {
        return Err("the key has characters no API key uses; copy it again from your provider".into());
    }
    if !(8..=512).contains(&k.chars().count()) {
        return Err("that does not look like an API key (8 to 512 characters); paste the key from your provider".into());
    }
    Ok(k.to_string())
}

/// What the client may see of a key: its last 4 characters.
pub fn hint(key: &str) -> String {
    let tail: String = key.chars().rev().take(4).collect::<Vec<_>>().into_iter().rev().collect();
    format!("••••{tail}")
}

/// Keys are kept only under the presets' ids (a typo such as `opencode-gp` is an error, not "no key").
fn known(preset: &str) -> Result<(), String> {
    if valid_preset(preset) && super::presets::preset(preset).is_some() {
        Ok(())
    } else {
        Err(format!("unknown provider {preset:?}"))
    }
}

/// The app's keys: the store, opened on first use (or given, for tests and the eval).
#[derive(Debug, Default)]
pub struct ApiKeys {
    store: OnceLock<Arc<dyn SecretStore>>,
}

impl ApiKeys {
    pub fn with_store(store: Arc<dyn SecretStore>) -> ApiKeys {
        let keys = ApiKeys::default();
        let _ = keys.store.set(store);
        keys
    }

    pub fn store(&self) -> &Arc<dyn SecretStore> {
        self.store.get_or_init(open)
    }

    pub fn set(&self, preset: &str, key: &str) -> Result<String, String> {
        known(preset)?;
        let key = clean_key(key)?;
        self.store().set(preset, &key)?;
        Ok(hint(&key))
    }

    /// The key itself: for the provider call only, never for a client message, a log line or the chat store.
    pub fn get(&self, preset: &str) -> Result<Option<String>, String> {
        known(preset)?;
        self.store().get(preset)
    }

    /// The hint for a saved key, `None` when there is none.
    pub fn status(&self, preset: &str) -> Result<Option<String>, String> {
        Ok(self.get(preset)?.map(|k| hint(&k)))
    }

    pub fn delete(&self, preset: &str) -> Result<(), String> {
        known(preset)?;
        self.store().delete(preset)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys_are_cleaned_and_only_hinted() {
        let keys = ApiKeys::with_store(Arc::new(SessionOnly::default()));
        assert_eq!(keys.set("opencode-go", "  sk-test-SECRET-1a2b\n").unwrap(), "••••1a2b");
        assert_eq!(keys.get("opencode-go").unwrap().as_deref(), Some("sk-test-SECRET-1a2b"));
        assert_eq!(keys.status("opencode-go").unwrap().as_deref(), Some("••••1a2b"));
        assert_eq!(keys.status("openrouter").unwrap(), None);
        keys.delete("opencode-go").unwrap();
        assert_eq!(keys.get("opencode-go").unwrap(), None);
        keys.delete("opencode-go").unwrap();
        assert!(keys.set("opencode-go", "").unwrap_err().contains("empty"));
        assert!(keys.set("opencode-go", "sk-one two").unwrap_err().contains("spaces"));
        assert!(keys.set("opencode-go", "short").unwrap_err().contains("8 to 512"));
        assert!(keys.set("../etc", "sk-test-SECRET-1a2b").unwrap_err().contains("unknown provider"));
        assert!(!keys.store().persistent());
    }

    #[test]
    fn debug_output_never_shows_a_key() {
        let store = SessionOnly::default();
        store.set("opencode-go", "sk-test-SECRET-1a2b").unwrap();
        let keys = ApiKeys::with_store(Arc::new(store));
        let shown = format!("{:?}", keys);
        assert!(!shown.contains("SECRET") && shown.contains("opencode-go"), "{shown}");
    }

    /// Writes and deletes a test entry in this machine's real keychain: `cargo test -- --ignored keychain`.
    #[test]
    #[ignore]
    fn keychain_round_trip() {
        let store = system_store().expect("a system keychain");
        let kc = Keychain { store, service: "decklight-ask-test".into() };
        kc.set("probe", "sk-test-SECRET-1a2b").unwrap();
        assert_eq!(kc.get("probe").unwrap().as_deref(), Some("sk-test-SECRET-1a2b"));
        kc.delete("probe").unwrap();
        assert_eq!(kc.get("probe").unwrap(), None);
    }
}
