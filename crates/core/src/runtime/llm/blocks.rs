use std::collections::BTreeMap;
use std::sync::Arc;

use crate::protocol::LlmFormat;
use crate::runtime::session::decision::LlmHandler;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LlmBlock {
    pub handler: LlmHandler,

    pub format: Option<LlmFormat>,
}

impl LlmBlock {
    pub fn engine() -> Self {
        Self {
            handler: LlmHandler::Server,
            format: None,
        }
    }

    pub fn worker(format: Option<LlmFormat>) -> Self {
        Self {
            handler: LlmHandler::Worker,
            format,
        }
    }
}

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

    pub fn declared(&self) -> String {
        crate::copy::declared(self.0.keys())
    }
}

impl FromIterator<(String, LlmBlock)> for LlmBlocks {
    fn from_iter<I: IntoIterator<Item = (String, LlmBlock)>>(iter: I) -> Self {
        Self::new(iter.into_iter().collect())
    }
}
