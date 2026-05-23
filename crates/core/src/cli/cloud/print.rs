// Output helpers shared by every cloud command. The contract is: if
// `globals.json`, emit a single JSON document on stdout (pretty-printed,
// trailing newline) and skip the human path. Errors always stay on stderr
// in plain text — matching `gh`'s convention.

use anyhow::Result;
use serde::Serialize;

pub fn json<T: Serialize>(value: &T) -> Result<()> {
    let s = serde_json::to_string_pretty(value)?;
    println!("{s}");
    Ok(())
}
