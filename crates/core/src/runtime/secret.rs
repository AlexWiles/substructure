//! Where secrets live: one store, one table, one place to encrypt.
//!
//! Everything the engine must read back and present to a third party — an
//! OAuth grant, a static token — goes through [`SecretStore`], and the tables
//! that need one carry a reference, the way `blob://` references bytes. The
//! rest of the database never holds a secret.
//!
//! One person on one machine holding their own credential in plain text is
//! acceptable; a team server holding the mail credentials of many persons is
//! not. So the key is optional, and the engine refuses to *start* — not to
//! run — when a personal connection is declared and no key is set.
//!
//! AES-256-GCM under one key from `SUBS_SECRET_KEY` (64 hex chars) — the same
//! primitive and key shape as the cloud's `SECRETS_MASTER_KEY`. Each sealed
//! row records a key version so a rotation scheme can slot in later; today
//! there is one version, and rotating means re-sealing under the new key
//! before dropping the old one.

use aes_gcm::aead::Aead;
use aes_gcm::{Aes256Gcm, KeyInit, Nonce};
use async_trait::async_trait;
use base64::Engine as _;

use crate::event_store::StoreError;

pub const KEYS_ENV: &str = "SUBS_SECRET_KEY";

const NONCE_LEN: usize = 12;

/// An opaque reference to one secret — what every other table carries in
/// place of the material, the way `blob://` references bytes.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(transparent)]
pub struct SecretRef(String);

impl SecretRef {
    /// A fresh reference for a secret about to be written.
    pub fn mint() -> Self {
        Self(uuid::Uuid::now_v7().to_string())
    }

    /// A reference as another table stored it.
    pub fn from_stored(id: String) -> Self {
        Self(id)
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for SecretRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Bytes the engine will need back, keyed by a reference another table
/// carries. Writing to a reference it already holds replaces the value, which
/// is what a token rotation is.
#[async_trait]
pub trait SecretStore: Send + Sync {
    async fn put(&self, tenant_id: &str, r: &SecretRef, value: &[u8]) -> Result<(), StoreError>;
    async fn get(&self, tenant_id: &str, r: &SecretRef) -> Result<Option<Vec<u8>>, StoreError>;
    async fn delete(&self, tenant_id: &str, r: &SecretRef) -> Result<bool, StoreError>;
}

pub struct SecretCipher {
    cipher: Aes256Gcm,
}

impl SecretCipher {
    /// The cipher the environment configures, or nothing when the variable is
    /// unset. A variable that is set and wrong is an error, never "no cipher":
    /// silently running unencrypted is the one outcome this must not have.
    pub fn from_env() -> Result<Option<Self>, String> {
        match std::env::var(KEYS_ENV) {
            Ok(v) => Self::parse(&v).map(Some),
            Err(std::env::VarError::NotPresent) => Ok(None),
            Err(e) => Err(format!("${KEYS_ENV}: {e}")),
        }
    }

    /// 64 hex chars — the same key material shape as the cloud's master key.
    pub fn parse(value: &str) -> Result<Self, String> {
        let bytes = hex::decode(value.trim()).map_err(|e| {
            format!("${KEYS_ENV} is not hex: {e} (generate a key with `openssl rand -hex 32`)")
        })?;
        if bytes.len() != 32 {
            return Err(format!(
                "${KEYS_ENV} is {} bytes, not 32 (generate a key with `openssl rand -hex 32`)",
                bytes.len()
            ));
        }
        let cipher = Aes256Gcm::new_from_slice(&bytes).map_err(|e| format!("${KEYS_ENV}: {e}"))?;
        Ok(Self { cipher })
    }

    /// The version stamped on new rows.
    pub fn current_version(&self) -> u32 {
        1
    }

    pub fn encrypt(&self, plaintext: &[u8]) -> (u32, String) {
        let mut nonce = [0u8; NONCE_LEN];
        rand::Rng::fill(&mut rand::rng(), &mut nonce);
        let sealed = self
            .cipher
            .encrypt(Nonce::from_slice(&nonce), plaintext)
            .expect("AES-GCM encryption is infallible for in-memory data");
        let mut framed = nonce.to_vec();
        framed.extend(sealed);
        (
            self.current_version(),
            base64::engine::general_purpose::STANDARD.encode(framed),
        )
    }

    pub fn decrypt(&self, encoded: &str) -> Result<Vec<u8>, String> {
        let framed = base64::engine::general_purpose::STANDARD
            .decode(encoded)
            .map_err(|e| format!("stored ciphertext is not base64: {e}"))?;
        if framed.len() < NONCE_LEN {
            return Err("stored ciphertext is shorter than its nonce".to_string());
        }
        let (nonce, sealed) = framed.split_at(NONCE_LEN);
        self.cipher
            .decrypt(Nonce::from_slice(nonce), sealed)
            .map_err(|_| {
                format!("decryption failed: ${KEYS_ENV} is not the key this was sealed under")
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key() -> String {
        "ab".repeat(32)
    }

    #[test]
    fn a_secret_round_trips_and_never_repeats_a_nonce() {
        let cipher = SecretCipher::parse(&key()).unwrap();
        let (version, a) = cipher.encrypt(b"xoxb-secret");
        let (_, b) = cipher.encrypt(b"xoxb-secret");
        assert_eq!(version, 1);
        assert_ne!(a, b, "a fresh nonce per encryption");
        assert_eq!(cipher.decrypt(&a).unwrap(), b"xoxb-secret");
        assert_eq!(cipher.decrypt(&b).unwrap(), b"xoxb-secret");
    }

    #[test]
    fn the_wrong_key_and_tampering_fail_closed() {
        let cipher = SecretCipher::parse(&key()).unwrap();
        let (_, sealed) = cipher.encrypt(b"secret");

        let other = SecretCipher::parse(&"cd".repeat(32)).unwrap();
        let err = other.decrypt(&sealed).unwrap_err();
        assert!(err.contains(KEYS_ENV), "{err}");

        let mut tampered = sealed.clone();
        tampered.replace_range(0..1, if sealed.starts_with('A') { "B" } else { "A" });
        assert!(cipher.decrypt(&tampered).is_err());
        assert!(cipher.decrypt("AAAA").is_err());
    }

    #[test]
    fn a_set_and_wrong_variable_is_an_error_not_no_cipher() {
        for wrong in ["", "abc", "zz", &"ab".repeat(16)] {
            assert!(SecretCipher::parse(wrong).is_err(), "{wrong}");
        }
    }
}
