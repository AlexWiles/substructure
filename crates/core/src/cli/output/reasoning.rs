//! The thinking block, bounded so it cannot bury the answer after it.

const REASONING_LINES: usize = 24;

#[derive(Default)]
pub struct Reasoning {
    lines: usize,
    held: usize,
}

impl Reasoning {
    pub fn start(&mut self) {
        self.lines = 0;
        self.held = 0;
    }

    pub fn take(&mut self, delta: &str) -> String {
        let mut kept = String::new();
        for piece in delta.split_inclusive('\n') {
            if self.lines < REASONING_LINES {
                kept.push_str(piece);
                if piece.ends_with('\n') {
                    self.lines += 1;
                }
            } else if piece.ends_with('\n') {
                self.held += 1;
            }
        }
        kept
    }

    pub fn held(&self) -> usize {
        self.held
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stream(reasoning: &mut Reasoning, lines: usize) -> String {
        (1..=lines)
            .map(|n| reasoning.take(&format!("thought {n}\n")))
            .collect()
    }

    #[test]
    fn a_short_thought_is_written_whole() {
        let mut reasoning = Reasoning::default();
        let out = stream(&mut reasoning, 5);
        assert_eq!(out.lines().count(), 5);
        assert_eq!(reasoning.held(), 0);
    }

    #[test]
    fn a_long_thought_stops_at_the_cap_and_counts_the_rest() {
        let mut reasoning = Reasoning::default();
        let out = stream(&mut reasoning, 30);
        assert_eq!(out.lines().count(), REASONING_LINES);
        assert!(out.contains("thought 24\n"));
        assert!(!out.contains("thought 25"));
        assert_eq!(reasoning.held(), 6);
    }

    #[test]
    fn a_delta_that_is_not_a_whole_line_still_counts_by_line() {
        let mut reasoning = Reasoning::default();
        assert_eq!(reasoning.take("half"), "half");
        assert_eq!(reasoning.take(" a line\n"), " a line\n");
        assert_eq!(reasoning.held(), 0);
    }

    #[test]
    fn each_block_gets_the_whole_cap() {
        let mut reasoning = Reasoning::default();
        stream(&mut reasoning, 30);
        assert_eq!(reasoning.held(), 6);
        reasoning.start();
        let out = stream(&mut reasoning, 5);
        assert_eq!(out.lines().count(), 5);
        assert_eq!(reasoning.held(), 0);
    }
}
