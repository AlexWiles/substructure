use anyhow::Result;
use serde::Serialize;

pub fn json<T: Serialize>(value: &T) -> Result<()> {
    let s = serde_json::to_string_pretty(value)?;
    println!("{s}");
    Ok(())
}

#[derive(Clone, Copy)]
pub enum Align {
    Left,
    Right,
}

pub struct Column {
    pub header: &'static str,
    pub align: Align,
}

impl Column {
    pub const fn left(header: &'static str) -> Self {
        Self {
            header,
            align: Align::Left,
        }
    }
    pub const fn right(header: &'static str) -> Self {
        Self {
            header,
            align: Align::Right,
        }
    }
}

/// Render a borderless table that auto-sizes each column from its widest cell
/// and truncates with `…` when the total exceeds the terminal width.
pub fn table(columns: &[Column], rows: &[Vec<String>]) {
    let n = columns.len();
    if n == 0 {
        return;
    }

    let mut widths: Vec<usize> = columns.iter().map(|c| c.header.chars().count()).collect();
    for row in rows {
        for (i, cell) in row.iter().enumerate().take(n) {
            let w = cell.chars().count();
            if w > widths[i] {
                widths[i] = w;
            }
        }
    }

    // Shrink the widest columns one char at a time until we fit. We never
    // shrink below MIN, so a too-narrow terminal will still overflow rather
    // than collapsing every column into ellipses.
    const GAP: usize = 2;
    const MIN: usize = 6;
    let term = term_width();
    let overhead = GAP * n.saturating_sub(1);
    let mut total: usize = widths.iter().sum::<usize>() + overhead;
    while total > term {
        let (idx, max_w) = widths
            .iter()
            .enumerate()
            .max_by_key(|(_, w)| **w)
            .expect("non-empty columns");
        if *max_w <= MIN {
            break;
        }
        widths[idx] -= 1;
        total -= 1;
    }

    let render = |cells: &[String]| {
        let mut line = String::new();
        for (i, w) in widths.iter().enumerate() {
            let raw = cells.get(i).map(String::as_str).unwrap_or("");
            let cell = truncate(raw, *w);
            // Pad by char count, not byte count, so multi-byte content
            // (e.g. unicode names) lines up.
            let pad = w.saturating_sub(cell.chars().count());
            let padding: String = " ".repeat(pad);
            match columns[i].align {
                Align::Left => {
                    line.push_str(&cell);
                    line.push_str(&padding);
                }
                Align::Right => {
                    line.push_str(&padding);
                    line.push_str(&cell);
                }
            }
            if i + 1 < widths.len() {
                line.push_str("  ");
            }
        }
        println!("{}", line.trim_end());
    };

    let headers: Vec<String> = columns.iter().map(|c| c.header.to_string()).collect();
    render(&headers);
    for row in rows {
        render(row);
    }
}

fn truncate(s: &str, width: usize) -> String {
    let len = s.chars().count();
    if len <= width {
        return s.to_string();
    }
    if width == 0 {
        return String::new();
    }
    if width <= 3 {
        // Too narrow for the ellipsis trick; just dot-pad.
        return ".".repeat(width);
    }
    let mut out: String = s.chars().take(width - 1).collect();
    out.push('…');
    out
}

fn term_width() -> usize {
    // Shells (bash, zsh) set COLUMNS automatically; honor it when exported.
    // No libc dep needed and `--no-color`-style pipe usage falls back cleanly.
    if let Ok(s) = std::env::var("COLUMNS") {
        if let Ok(n) = s.parse::<usize>() {
            if n > 0 {
                return n;
            }
        }
    }
    120
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

/// Web origin for the admin UI, given the CLI's configured API URL. In prod
/// we swap `api.` for `app.`; in dev (same-origin) we leave the host alone.
fn web_base(api_url: &str) -> String {
    let base = if let Some(rest) = api_url.strip_prefix("https://api.") {
        format!("https://app.{rest}")
    } else if let Some(rest) = api_url.strip_prefix("http://api.") {
        format!("http://app.{rest}")
    } else {
        api_url.to_string()
    };
    base.trim_end_matches('/').to_string()
}

/// Admin landing page for a project. Doubles as the top-up entry point.
pub fn admin_url(api_url: &str, project_id: &str) -> String {
    format!("{}/projects/{project_id}", web_base(api_url))
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
    fn admin_url_swaps_api_subdomain() {
        assert_eq!(
            admin_url("https://api.substructure.ai", "abc"),
            "https://app.substructure.ai/projects/abc"
        );
        assert_eq!(
            admin_url("http://api.local.test", "abc"),
            "http://app.local.test/projects/abc"
        );
        assert_eq!(
            admin_url("http://localhost:5173", "abc"),
            "http://localhost:5173/projects/abc"
        );
        assert_eq!(
            admin_url("https://api.substructure.ai/", "abc"),
            "https://app.substructure.ai/projects/abc"
        );
    }

    #[test]
    fn truncate_preserves_short_strings() {
        assert_eq!(truncate("hi", 5), "hi");
        assert_eq!(truncate("hello", 5), "hello");
    }

    #[test]
    fn truncate_uses_ellipsis_when_too_long() {
        assert_eq!(truncate("abcdef", 5), "abcd…");
        assert_eq!(truncate("hello world", 8), "hello w…");
    }

    #[test]
    fn truncate_handles_narrow_widths() {
        assert_eq!(truncate("abcdef", 0), "");
        assert_eq!(truncate("abcdef", 1), ".");
        assert_eq!(truncate("abcdef", 3), "...");
    }
}
