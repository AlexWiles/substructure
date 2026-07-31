//! The `[llm.*]` blocks an app declares, as the engine reads them.
//!
//! A block's `type` is the execution venue — the engine with a vendor key, or
//! the agent's own worker — so naming a block on a call settles both where the
//! call runs and, for a worker-run one, the wire shape of its `llm.execute`.
//! Declared in exactly one place, referenced identically from the config file
//! and from a worker's `llm.call`.
//!
//! Credentials are deliberately absent: what binds a key lives where the engine
//! runs (see [`crate::runtime::llm::LlmProviderRegistry`]), not in the document
//! a decision is resolved against.

use std::collections::BTreeMap;
use std::sync::Arc;

use crate::protocol::LlmFormat;
use crate::runtime::session::decision::LlmHandler;

/// One declared block: where a call on it runs, and the wire shape of a
/// worker-run one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LlmBlock {
    pub handler: LlmHandler,
    /// Only ever set on a worker-run block; a server call is always neutral.
    pub format: Option<LlmFormat>,
}

impl LlmBlock {
    /// A block the engine executes itself against a vendor.
    pub fn engine() -> Self {
        Self {
            handler: LlmHandler::Server,
            format: None,
        }
    }

    /// A block the agent's worker executes, shaped as `format` on the wire.
    pub fn worker(format: Option<LlmFormat>) -> Self {
        Self {
            handler: LlmHandler::Worker,
            format,
        }
    }
}

/// Every block one app declares, keyed by the name an agent references. Cheap
/// to clone: a decision seam takes one by value.
#[derive(Debug, Clone, Default)]
pub struct LlmBlocks(Arc<BTreeMap<String, LlmBlock>>);

impl LlmBlocks {
    pub fn new(blocks: BTreeMap<String, LlmBlock>) -> Self {
        Self(Arc::new(blocks))
    }

    pub fn get(&self, name: &str) -> Option<LlmBlock> {
        self.0.get(name).copied()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// The declared names, for the error that says what could have been named.
    pub fn declared(&self) -> String {
        match self.0.is_empty() {
            true => "none".to_string(),
            false => self
                .0
                .keys()
                .map(String::as_str)
                .collect::<Vec<_>>()
                .join(", "),
        }
    }
}

impl FromIterator<(String, LlmBlock)> for LlmBlocks {
    fn from_iter<I: IntoIterator<Item = (String, LlmBlock)>>(iter: I) -> Self {
        Self::new(iter.into_iter().collect())
    }
}
