use std::collections::HashMap;

use thiserror::Error;

use super::config::{SecretProviderConfig, SystemConfig};

#[derive(Debug, Error)]
pub enum SecretError {
    #[error("secret not found: {provider} var {key} not set")]
    NotFound { provider: String, key: String },
    #[error("unknown secret provider: {0}")]
    UnknownProvider(String),
    #[error("secret resolution failed: {0}")]
    Resolution(String),
}

pub trait SecretProvider: Send + Sync {
    fn resolve(
        &self,
        secret: &serde_json::Map<String, serde_json::Value>,
    ) -> Result<String, SecretError>;
}

pub struct EnvSecretProvider;

impl SecretProvider for EnvSecretProvider {
    fn resolve(
        &self,
        secret: &serde_json::Map<String, serde_json::Value>,
    ) -> Result<String, SecretError> {
        let key = secret.get("key").and_then(|v| v.as_str()).ok_or_else(|| {
            SecretError::Resolution("env provider requires a \"key\" field".into())
        })?;
        std::env::var(key).map_err(|_| SecretError::NotFound {
            provider: "env".into(),
            key: key.into(),
        })
    }
}

fn build_providers(
    configs: &HashMap<String, SecretProviderConfig>,
) -> Result<HashMap<String, Box<dyn SecretProvider>>, SecretError> {
    let mut providers = HashMap::new();
    for (name, config) in configs {
        let provider: Box<dyn SecretProvider> = match config.provider_type.as_str() {
            "env" => Box::new(EnvSecretProvider),
            other => return Err(SecretError::UnknownProvider(other.into())),
        };
        providers.insert(name.clone(), provider);
    }
    Ok(providers)
}

pub fn resolve_secrets(config: SystemConfig) -> Result<SystemConfig, SecretError> {
    let providers = build_providers(&config.secret_providers)?;

    let mut value = serde_json::to_value(&config)
        .map_err(|e| SecretError::Resolution(format!("serialize: {e}")))?;

    resolve_value(&providers, &mut value)?;

    serde_json::from_value(value).map_err(|e| SecretError::Resolution(format!("deserialize: {e}")))
}

fn resolve_value(
    providers: &HashMap<String, Box<dyn SecretProvider>>,
    value: &mut serde_json::Value,
) -> Result<(), SecretError> {
    match value {
        serde_json::Value::Object(map) => {
            // Check if this object is a secret reference
            if let Some(serde_json::Value::String(provider_name)) = map.get("provider") {
                if let Some(provider) = providers.get(provider_name.as_str()) {
                    let resolved = provider.resolve(map)?;
                    *value = serde_json::Value::String(resolved);
                    return Ok(());
                }
            }
            // Otherwise recurse into children
            let keys: Vec<String> = map.keys().cloned().collect();
            for k in keys {
                if let Some(v) = map.get_mut(&k) {
                    resolve_value(providers, v)?;
                }
            }
        }
        serde_json::Value::Array(arr) => {
            for item in arr.iter_mut() {
                resolve_value(providers, item)?;
            }
        }
        _ => {}
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_env_secret() {
        let provider = EnvSecretProvider;
        std::env::set_var("TEST_SECRET_KEY", "resolved-value");
        let secret = serde_json::json!({ "provider": "env", "key": "TEST_SECRET_KEY" });
        let map = secret.as_object().unwrap();
        assert_eq!(provider.resolve(map).unwrap(), "resolved-value");
        std::env::remove_var("TEST_SECRET_KEY");
    }

    #[test]
    fn missing_env_returns_not_found() {
        let provider = EnvSecretProvider;
        let secret = serde_json::json!({ "provider": "env", "key": "DEFINITELY_NOT_SET_12345" });
        let map = secret.as_object().unwrap();
        let err = provider.resolve(map).unwrap_err();
        assert!(matches!(err, SecretError::NotFound { .. }));
    }

    #[test]
    fn missing_key_field_returns_error() {
        let provider = EnvSecretProvider;
        let secret = serde_json::json!({ "provider": "env" });
        let map = secret.as_object().unwrap();
        let err = provider.resolve(map).unwrap_err();
        assert!(matches!(err, SecretError::Resolution(_)));
    }

    #[test]
    fn resolve_value_replaces_secret_ref() {
        let mut providers: HashMap<String, Box<dyn SecretProvider>> = HashMap::new();
        providers.insert("env".into(), Box::new(EnvSecretProvider));

        std::env::set_var("TEST_RESOLVE_VAL", "the-secret");
        let mut value = serde_json::json!({
            "normal": "stays",
            "nested": {
                "secret": { "provider": "env", "key": "TEST_RESOLVE_VAL" }
            }
        });
        resolve_value(&providers, &mut value).unwrap();
        assert_eq!(value["normal"], "stays");
        assert_eq!(value["nested"]["secret"], "the-secret");
        std::env::remove_var("TEST_RESOLVE_VAL");
    }

    #[test]
    fn unknown_provider_in_config_errors() {
        let mut configs = HashMap::new();
        configs.insert(
            "bad".into(),
            SecretProviderConfig {
                provider_type: "nonexistent".into(),
                settings: serde_json::Map::new(),
            },
        );
        match build_providers(&configs) {
            Err(SecretError::UnknownProvider(_)) => {}
            Ok(_) => panic!("expected UnknownProvider error"),
            Err(e) => panic!("expected UnknownProvider, got {e}"),
        }
    }
}
