use anyhow::Result;
use serde::Serialize;

pub fn json<T: Serialize>(value: &T) -> Result<()> {
    let s = serde_json::to_string_pretty(value)?;
    println!("{s}");
    Ok(())
}

/// Format a decimal USD string (e.g. "1234.5") as currency ("$1,234.50").
/// Returns the input unchanged on parse failure.
pub fn fmt_usd(raw: &str) -> String {
    let Ok(n) = raw.trim().parse::<f64>() else {
        return raw.to_string();
    };
    let neg = n < 0.0;
    let cents_total = (n.abs() * 100.0).round() as u64;
    let whole = cents_total / 100;
    let cents = cents_total % 100;

    let whole_str = whole.to_string();
    let mut rev_with_commas = String::new();
    for (i, ch) in whole_str.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            rev_with_commas.push(',');
        }
        rev_with_commas.push(ch);
    }
    let whole_with_commas: String = rev_with_commas.chars().rev().collect();
    format!(
        "{}${whole_with_commas}.{cents:02}",
        if neg { "-" } else { "" }
    )
}

/// True if a decimal USD string parses to exactly zero.
pub fn is_zero_usd(raw: &str) -> bool {
    raw.trim().parse::<f64>().map(|n| n == 0.0).unwrap_or(false)
}

/// Build the web URL where an app's top-up flow lives, given the CLI's
/// configured API URL. In prod we swap `api.` for `app.`; in dev (same-origin)
/// we leave the host alone.
pub fn topup_url(api_url: &str, app_id: &str) -> String {
    let base = if let Some(rest) = api_url.strip_prefix("https://api.") {
        format!("https://app.{rest}")
    } else if let Some(rest) = api_url.strip_prefix("http://api.") {
        format!("http://app.{rest}")
    } else {
        api_url.to_string()
    };
    format!("{}/apps/{app_id}/overview", base.trim_end_matches('/'))
}

/// Print the standard zero-balance warning line for an app. Goes to stderr
/// so it doesn't pollute `--json` stdout.
pub fn warn_zero_balance(app_name: &str, api_url: &str, app_id: &str) {
    eprintln!(
        "⚠ {app_name} has zero balance. Top up: {}",
        topup_url(api_url, app_id)
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fmt_usd_basic() {
        assert_eq!(fmt_usd("0"), "$0.00");
        assert_eq!(fmt_usd("0.00"), "$0.00");
        assert_eq!(fmt_usd("1.5"), "$1.50");
        assert_eq!(fmt_usd("1234.5"), "$1,234.50");
        assert_eq!(fmt_usd("1234567.89"), "$1,234,567.89");
        assert_eq!(fmt_usd("-5.25"), "-$5.25");
        assert_eq!(fmt_usd("not a number"), "not a number");
    }

    #[test]
    fn is_zero_usd_checks() {
        assert!(is_zero_usd("0"));
        assert!(is_zero_usd("0.0"));
        assert!(is_zero_usd("0.00"));
        assert!(is_zero_usd(" 0 "));
        assert!(!is_zero_usd("0.01"));
        assert!(!is_zero_usd("-1"));
        assert!(!is_zero_usd("not a number"));
    }

    #[test]
    fn topup_url_swaps_api_subdomain() {
        assert_eq!(
            topup_url("https://api.substructure.ai", "abc"),
            "https://app.substructure.ai/apps/abc/overview"
        );
        assert_eq!(
            topup_url("http://api.local.test", "abc"),
            "http://app.local.test/apps/abc/overview"
        );
        assert_eq!(
            topup_url("http://localhost:5173", "abc"),
            "http://localhost:5173/apps/abc/overview"
        );
        assert_eq!(
            topup_url("https://api.substructure.ai/", "abc"),
            "https://app.substructure.ai/apps/abc/overview"
        );
    }
}
