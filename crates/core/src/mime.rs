use base64::Engine;

pub(crate) fn parts(mime: &str) -> (&str, &str) {
    let base = mime.split(';').next().unwrap_or_default().trim();
    match base.split_once('/') {
        Some((kind, sub)) => (kind, sub),
        None => (base, ""),
    }
}

pub(crate) fn base(mime: &str) -> &str {
    mime.split(';').next().unwrap_or_default().trim()
}

pub(crate) fn essence(mime: &str) -> &str {
    match parts(mime).0 {
        "" => "file",
        kind => kind,
    }
}

pub(crate) fn text_like(mime: &str) -> bool {
    let (kind, _) = parts(mime);
    kind == "text"
        || matches!(
            base(mime),
            "application/json"
                | "application/xml"
                | "application/yaml"
                | "application/x-yaml"
                | "application/toml"
                | "application/csv"
                | "application/x-ndjson"
                | "application/javascript"
                | "application/typescript"
                | "application/x-sh"
                | "application/sql"
        )
}

pub(crate) fn base64(bytes: &[u8]) -> String {
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

pub(crate) fn data_uri(mime: &str, bytes: &[u8]) -> String {
    format!("data:{mime};base64,{}", base64(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_like_covers_the_business_formats() {
        for mime in [
            "text/plain",
            "text/markdown",
            "text/csv; charset=utf-8",
            "application/json",
            "application/x-yaml",
        ] {
            assert!(text_like(mime), "{mime}");
        }
        for mime in ["application/pdf", "image/png", "application/zip"] {
            assert!(!text_like(mime), "{mime}");
        }
    }

    #[test]
    fn essence_is_the_type_or_file() {
        assert_eq!(essence("image/png"), "image");
        assert_eq!(essence("audio/wav; rate=44100"), "audio");
        assert_eq!(essence(""), "file");
        assert_eq!(parts("video/mp4"), ("video", "mp4"));
    }
}
