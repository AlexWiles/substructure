use schemars::JsonSchema;
use serde::Deserialize;

pub(crate) fn text(size: u64) -> String {
    let units = ["KB", "MB", "GB", "TB"];
    let mut value = size as f64 / 1024.0;
    let mut unit = 0;
    while value >= 1024.0 && unit + 1 < units.len() {
        value /= 1024.0;
        unit += 1;
    }
    match size < 1024 {
        true => format!("{size} B"),
        false => format!("{value:.1} {}", units[unit]),
    }
}

pub(crate) fn parse(text: &str) -> Option<u64> {
    let text = text.trim().to_ascii_lowercase();
    let number = text.trim_end_matches(|c: char| c.is_ascii_alphabetic());
    let scale = match text[number.len()..].trim() {
        "" | "b" => 1u64,
        "k" | "kb" => 1 << 10,
        "m" | "mb" => 1 << 20,
        "g" | "gb" => 1 << 30,
        _ => return None,
    };
    let number: f64 = number.trim().parse().ok()?;
    (number >= 0.0).then_some((number * scale as f64) as u64)
}

/// A size on the wire: bytes, or a word a person writes.
#[derive(Deserialize, JsonSchema)]
#[serde(untagged)]
#[schemars(title = "Size")]
pub(crate) enum Wire {
    Bytes(u64),
    Text(String),
}

pub(crate) fn de<'de, D: serde::Deserializer<'de>>(d: D) -> Result<Option<u64>, D::Error> {
    match Option::<Wire>::deserialize(d)? {
        None => Ok(None),
        Some(Wire::Bytes(n)) => Ok(Some(n)),
        Some(Wire::Text(t)) => parse(&t)
            .map(Some)
            .ok_or_else(|| serde::de::Error::custom(format!("`{t}` is not a size"))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes_read_the_way_a_person_writes_them() {
        assert_eq!(text(0), "0 B");
        assert_eq!(text(1023), "1023 B");
        assert_eq!(text(2048), "2.0 KB");
        assert_eq!(text(20 << 20), "20.0 MB");
        assert_eq!(parse("20mb"), Some(20 << 20));
        assert_eq!(parse("20.0 MB"), Some(20 << 20));
        assert_eq!(parse("512"), Some(512));
        assert_eq!(parse("2 kb"), Some(2048));
        assert_eq!(parse("big"), None);
        assert_eq!(parse("20pb"), None);
    }
}
