//! The Slack front end: what the bot shows, and what a click means.
//!
//! The split is front end from plumbing. This decides what a message reads as —
//! the answer, a failure, the card for a tool call, the buttons on a prompt, the
//! thread's status — and what to do when someone presses one of those buttons.
//! [`super::bot`] decides where it goes and that it arrives: threading,
//! idempotency, the stream lifecycle, retries, the stamp that lets a restart
//! recognise its own reply.
//!
//! Both halves work the same way: the controller *decides* and returns a value,
//! the bot *performs* it. Rendering returns blocks; an action returns an
//! [`ActionOutcome`]. Nothing here calls Slack or the engine, so a controller is
//! testable as a function and a slow one cannot stall delivery.
//!
//! Every method has a default, so [`DefaultController`] is an empty impl and a
//! deployment overrides only the part it wants to change — calling
//! `DefaultController` for anything it wants the engine's version of:
//!
//! ```ignore
//! #[async_trait]
//! impl SlackController for Cloud {
//!     fn turn(&self, ctx: &Context, t: &TurnCompleted) -> Rendered {
//!         match &t.error {
//!             Some(e) if e.code == ErrorCode::BudgetExceeded => self.billing_card(ctx, e),
//!             _ => DefaultController.turn(ctx, t),
//!         }
//!     }
//!     fn trailing_blocks(&self, ctx: &Context) -> Vec<Value> {
//!         vec![context_block(&self.admin_link(ctx))]
//!     }
//!     async fn on_action(&self, a: &Action<'_>) -> ActionOutcome {
//!         match a.action_id {
//!             "upgrade" => ActionOutcome::Reply(self.checkout_link(a)),
//!             _ => ActionOutcome::Ignored,
//!         }
//!     }
//! }
//! ```
//!
//! The one thing a controller must not do is drop a [`PromptOption`]'s
//! `action_id` and `value`. Those are the coordinates the bot reads back out of
//! a click to find the interrupt it answers, and a button without them resolves
//! to nothing. [`PromptOption::button`] is the safe path. Buttons a controller
//! invents carry any `action_id` it likes and come back to
//! [`SlackController::on_action`].

use std::time::Duration;

use async_trait::async_trait;
use serde_json::Value;

use super::{clip, MAX_SECTION};
use crate::session::events::TurnCompleted;

/// Keeps a chatty tool from bloating a card; not a Slack limit.
const MAX_TITLE: usize = 60;
/// The engine's own wording for work that ended with no result — a card that
/// never finished, and a run that was cancelled with one open.
const CANCELLED: &str = "Cancelled.";
#[cfg(test)]
pub(super) const MAX_TITLE_FOR_TESTS: usize = MAX_TITLE;

// ── What the controller is given ─────────────────────────────────────────

/// One Slack message: `text` is the notification fallback and what a client
/// without block support shows, `blocks` are the display. Slack wants both.
#[derive(Debug, Clone, Default)]
pub struct Rendered {
    pub text: String,
    pub blocks: Vec<Value>,
}

/// The session a message belongs to. Resolved before rendering — a controller is
/// a pure function of this and the event, so no render can hang, fail, or
/// depend on a lookup that raced it.
#[derive(Debug, Clone)]
pub struct Context<'a> {
    pub tenant_id: &'a str,
    pub session_id: &'a str,
    pub turn_id: &'a str,
    /// Which agent answered, when the bot knows — absent on a status refresh,
    /// which runs off a stream key rather than an event.
    pub agent_id: Option<&'a str>,
    /// How long the turn took, when the bot was streaming and knows.
    pub elapsed: Option<Duration>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StepStatus {
    InProgress,
    Complete,
    Error,
    /// Ended with no result: the engine dropped the call, or the turn it
    /// belonged to ended while it was still running. A fact rather than a
    /// sentence, so a deployment words it — or renders it — its own way.
    Cancelled,
}

impl StepStatus {
    /// Slack knows three states; a card that ended without a result takes the
    /// one that is not "still going" or "worked".
    pub fn wire(self) -> &'static str {
        match self {
            StepStatus::InProgress => "in_progress",
            StepStatus::Complete => "complete",
            StepStatus::Error | StepStatus::Cancelled => "error",
        }
    }
}

/// One unit of visible work — a tool call or a sub-agent run — folded from the
/// event log. The fold is the bot's; the card is the controller's.
#[derive(Debug, Clone)]
pub struct StepView<'a> {
    pub id: &'a str,
    pub name: &'a str,
    pub status: StepStatus,
    /// How long it took, once it settled.
    pub took: Option<&'a str>,
    pub input: Option<&'a str>,
    pub output: Option<&'a str>,
}

impl StepView<'_> {
    /// `name · 1.2s` once it settled, else the bare name.
    pub fn title(&self) -> String {
        match self.took {
            Some(took) => format!("{} · {took}", self.name),
            None => self.name.to_string(),
        }
    }
}

/// One answer a user can click.
///
/// `action_id` and `value` belong to the bot: it reads them back out of the
/// click to find the interrupt and the option. Render them verbatim.
#[derive(Debug, Clone)]
pub struct PromptOption {
    pub label: String,
    /// `primary` or `danger`, when the author asked for one.
    pub style: Option<String>,
    pub action_id: String,
    pub value: String,
}

impl PromptOption {
    /// The standard button, with the coordinates already in place.
    pub fn button(&self) -> Value {
        let mut button = serde_json::json!({
            "type": "button",
            "action_id": self.action_id,
            "text": { "type": "plain_text", "text": self.label },
            "value": self.value,
        });
        if let Some(style @ ("primary" | "danger")) = self.style.as_deref() {
            button["style"] = style.into();
        }
        button
    }
}

/// A question the session is parked on.
#[derive(Debug, Clone)]
pub struct PromptView<'a> {
    pub message: &'a str,
    pub options: &'a [PromptOption],
    /// RFC 3339, as the author wrote it.
    pub expires_at: Option<&'a str>,
}

/// A press on a button this controller rendered. The bot answers its own
/// prompt buttons before this; everything else arrives here.
#[derive(Debug, Clone)]
pub struct Action<'a> {
    pub action_id: &'a str,
    /// Whatever the button carried, verbatim.
    pub value: &'a str,
    pub tenant_id: &'a str,
    pub session_id: &'a str,
    /// The Slack user who pressed it.
    pub user: &'a str,
    pub channel: &'a str,
    /// The message the button is on, for an outcome that replaces it.
    pub message_ts: &'a str,
    pub thread_ts: &'a str,
}

/// What should happen next. The controller decides; the bot performs.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum ActionOutcome {
    /// Not this controller's button.
    Ignored,
    /// Replace the message the button is on — the usual way to retire it.
    Update(Rendered),
    /// Post into the thread.
    Reply(Rendered),
    /// Send text to the agent as the user who clicked, as though they had
    /// typed it.
    Submit(String),
}

// ── The seam ─────────────────────────────────────────────────────────────

#[async_trait]
pub trait SlackController: Send + Sync {
    /// The finished message, whole. `activity` is the turn's visible work,
    /// already rendered by this controller and trimmed to fit Slack's limits;
    /// the default puts the answer after it and the trailing blocks last.
    ///
    /// Override to compose differently — the answer above the work, the cards
    /// dropped entirely. `None` posts nothing at all.
    fn message(
        &self,
        ctx: &Context<'_>,
        turn: &TurnCompleted,
        activity: &[Value],
    ) -> Option<Rendered> {
        let answer = self.turn(ctx, turn);
        let mut blocks = activity.to_vec();
        blocks.extend(answer.blocks);
        blocks.extend(self.trailing_blocks(ctx));
        Some(Rendered {
            text: answer.text,
            blocks,
        })
    }

    /// The message a cancelled run leaves behind, whole. A cancel completes no
    /// turn, so it arrives here rather than at [`Self::message`] — the work it
    /// had done, and a line saying it stopped.
    ///
    /// `None` leaves the message as it was streamed. The stream closes either
    /// way: what it says is this seam's, that it stops is the bot's.
    fn cancelled(&self, ctx: &Context<'_>, activity: &[Value]) -> Option<Rendered> {
        let _ = ctx;
        let mut blocks = activity.to_vec();
        blocks.push(section_block(CANCELLED));
        blocks.extend(self.trailing_blocks(ctx));
        Some(Rendered {
            text: CANCELLED.to_string(),
            blocks,
        })
    }

    /// What a completed turn says. A failed run arrives here too, as
    /// `turn.error` — a deployment that only wants to reword one failure
    /// branches on the code and delegates the rest.
    fn turn(&self, ctx: &Context<'_>, turn: &TurnCompleted) -> Rendered {
        let answer = match (&turn.error, &turn.data) {
            (Some(error), _) => format!("Error: {error}"),
            (None, Value::Null) => "(no result)".to_string(),
            (None, Value::String(s)) => s.clone(),
            (None, other) => other.to_string(),
        };
        let footer = ctx.elapsed.map(format_elapsed);
        let mut blocks = vec![section_block(&answer)];
        if let Some(footer) = &footer {
            blocks.push(context_block(footer));
        }
        Rendered {
            text: super::with_footer(&answer, footer.as_deref()),
            blocks,
        }
    }

    /// Appended under every message the bot posts. The seam for a footer that
    /// belongs to the deployment rather than to the turn — a link into an admin
    /// panel, a plan notice.
    fn trailing_blocks(&self, ctx: &Context<'_>) -> Vec<Value> {
        let _ = ctx;
        Vec::new()
    }

    /// A settled unit of work, in the finished message.
    fn step_block(&self, step: &StepView<'_>) -> Value {
        let mut card = serde_json::json!({
            "type": "task_card",
            "task_id": step.id,
            "title": clip(&step.title(), MAX_TITLE),
            "status": step.status.wire(),
        });
        // A call that ended with no result has nothing to show, and a card
        // that shows nothing reads as one still waiting for it.
        let outcome = step.output.or(match step.status {
            StepStatus::Cancelled => Some(CANCELLED),
            _ => None,
        });
        for (field, heading, text) in [
            ("details", None, step.input),
            ("output", Some("Result:"), outcome),
        ] {
            // Slack rejects an empty text element.
            if let Some(text) = text.filter(|t| !t.trim().is_empty()) {
                card[field] = rich_text(heading, text);
            }
        }
        card
    }

    /// The same unit while the turn is still streaming. Slack caps a
    /// `task_update` chunk at 256 characters and sets the card by `id`.
    fn step_chunk(&self, step: &StepView<'_>) -> Value {
        serde_json::json!({
            "type": "task_update",
            "id": step.id,
            "title": clip(&step.title(), MAX_TITLE),
            "status": step.status.wire(),
        })
    }

    /// What the model said between calls.
    fn said_block(&self, text: &str) -> Value {
        section_block(text)
    }

    /// Stands for the steps dropped to fit Slack's message limits.
    fn elided_block(&self, count: usize) -> Value {
        context_block(&format!("_… {count} earlier steps_"))
    }

    /// A prompt the session is waiting on.
    fn prompt_blocks(&self, prompt: &PromptView<'_>) -> Vec<Value> {
        let mut blocks = vec![section_block(prompt.message)];
        if !prompt.options.is_empty() {
            let buttons: Vec<Value> = prompt.options.iter().map(PromptOption::button).collect();
            blocks.push(serde_json::json!({ "type": "actions", "elements": buttons }));
        }
        if let Some((raw, ts)) = prompt
            .expires_at
            .and_then(|e| Some((e, chrono::DateTime::parse_from_rfc3339(e).ok()?)))
        {
            blocks.push(context_block(&format!(
                "Expires <!date^{}^{{date_short_pretty}} {{time}}|{raw}>",
                ts.timestamp()
            )));
        }
        blocks
    }

    /// The same prompt once it is answered: the buttons are gone and the
    /// outcome is written under it.
    fn settled_prompt_blocks(&self, posted: &[Value], text: &str, resolution: &str) -> Vec<Value> {
        let mut settled: Vec<Value> = posted
            .iter()
            .filter(|b| b["type"] != "actions")
            .cloned()
            .collect();
        if settled.is_empty() {
            settled.push(section_block(text));
        }
        settled.push(context_block(resolution));
        settled
    }

    /// The thread status while a turn runs. `None` clears it.
    fn status(&self, ctx: &Context<'_>) -> Option<String> {
        let _ = ctx;
        Some("is thinking…".to_string())
    }

    /// Someone pressed a button this controller rendered. Return what should
    /// happen; the bot performs it.
    async fn on_action(&self, action: &Action<'_>) -> ActionOutcome {
        let _ = action;
        ActionOutcome::Ignored
    }
}

/// The open-source engine's own voice: every method as it comes, and no
/// buttons of its own to answer.
pub struct DefaultController;

impl SlackController for DefaultController {}

// ── Block helpers ────────────────────────────────────────────────────────

pub fn section_block(text: &str) -> Value {
    serde_json::json!({
        "type": "section",
        "text": { "type": "mrkdwn", "text": clip(text, MAX_SECTION) },
    })
}

pub fn context_block(text: &str) -> Value {
    serde_json::json!({
        "type": "context",
        "elements": [{ "type": "mrkdwn", "text": text }],
    })
}

pub fn rich_text(heading: Option<&str>, text: &str) -> Value {
    let mut elements = Vec::new();
    if let Some(heading) = heading {
        elements.push(serde_json::json!({
            "type": "text",
            "text": format!("{heading}\n"),
            "style": { "bold": true },
        }));
    }
    elements.push(serde_json::json!({ "type": "text", "text": text }));
    serde_json::json!({
        "type": "rich_text",
        "elements": [{ "type": "rich_text_section", "elements": elements }],
    })
}

pub fn format_elapsed(d: Duration) -> String {
    let ms = d.as_millis() as i64;
    if ms < 60_000 {
        return format!("{:.1}s", ms as f64 / 1000.0);
    }
    format!("{}m {:02}s", ms / 60_000, (ms % 60_000) / 1000)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::{ErrorCode, ErrorInfo};

    fn ctx() -> Context<'static> {
        Context {
            tenant_id: "t",
            session_id: "s",
            turn_id: "turn-1",
            agent_id: Some("a"),
            elapsed: Some(Duration::from_millis(1500)),
        }
    }

    fn completed(data: Value) -> TurnCompleted {
        TurnCompleted {
            turn_id: "turn-1".into(),
            data,
            turn_cost: Default::default(),
            turn_token_usage: Default::default(),
            error: None,
        }
    }

    fn step() -> StepView<'static> {
        StepView {
            id: "tc-1",
            name: "search",
            status: StepStatus::Complete,
            took: Some("1.2s"),
            input: Some("{\"q\":\"x\"}"),
            output: Some("found"),
        }
    }

    #[test]
    fn a_turn_renders_its_output_with_the_elapsed_time() {
        let p = DefaultController.turn(&ctx(), &completed(Value::String("done".into())));
        assert_eq!(p.text, "done\n\n_1.5s_");
        assert_eq!(p.blocks.len(), 2, "the answer and its footer");
        assert_eq!(p.blocks[0]["text"]["text"], "done");
        assert_eq!(p.blocks[1]["type"], "context");
    }

    #[test]
    fn a_non_string_result_renders_as_json() {
        let p = DefaultController.turn(&ctx(), &completed(serde_json::json!({"a": 1})));
        assert!(p.text.starts_with(r#"{"a":1}"#), "{}", p.text);
    }

    #[test]
    fn an_error_reads_as_its_sentence_not_its_code() {
        let mut t = completed(Value::Null);
        t.error = Some(
            ErrorInfo::new(ErrorCode::BudgetExceeded, "the monthly budget is spent")
                .with_param("agent.llm"),
        );
        let p = DefaultController.turn(&ctx(), &t);
        assert!(
            p.text.starts_with("Error: the monthly budget is spent"),
            "{}",
            p.text
        );
        assert!(
            !p.text.contains("agent.llm") && !p.text.contains("budget_exceeded"),
            "the machine-readable half stays out of the message; got {}",
            p.text
        );
    }

    #[test]
    fn a_step_card_carries_its_title_and_both_fields() {
        let card = DefaultController.step_block(&step());
        assert_eq!(card["type"], "task_card");
        assert_eq!(card["title"], "search · 1.2s");
        assert_eq!(card["status"], "complete");
        assert!(card["details"]["type"] == "rich_text");
        assert!(card["output"]["type"] == "rich_text");
    }

    #[test]
    fn a_step_with_nothing_to_show_leaves_the_field_out() {
        let mut s = step();
        s.output = Some("   ");
        s.input = None;
        let card = DefaultController.step_block(&s);
        assert!(card.get("details").is_none());
        assert!(card.get("output").is_none(), "blank is not output");
    }

    /// The one thing a controller must not drop: without them the click has no
    /// interrupt to resolve.
    #[test]
    fn a_prompt_button_carries_the_bots_coordinates() {
        let options = vec![PromptOption {
            label: "Approve".into(),
            style: Some("primary".into()),
            action_id: "prompt_option_0".into(),
            value: r#"{"interrupt_id":"i-1","option":0}"#.into(),
        }];
        let blocks = DefaultController.prompt_blocks(&PromptView {
            message: "Run it?",
            options: &options,
            expires_at: None,
        });
        let button = &blocks[1]["elements"][0];
        assert_eq!(button["action_id"], "prompt_option_0");
        assert_eq!(button["value"], r#"{"interrupt_id":"i-1","option":0}"#);
        assert_eq!(button["style"], "primary");
    }

    #[test]
    fn settling_a_prompt_drops_its_buttons_and_says_the_outcome() {
        let posted = vec![
            section_block("Run it?"),
            serde_json::json!({ "type": "actions", "elements": [] }),
        ];
        let settled = DefaultController.settled_prompt_blocks(&posted, "Run it?", "Approved");
        assert!(settled.iter().all(|b| b["type"] != "actions"));
        assert_eq!(settled.last().unwrap()["elements"][0]["text"], "Approved");
    }

    /// An empty impl is the engine's voice, so a deployment overrides only what
    /// it means to change.
    #[test]
    fn a_controller_overrides_one_method_and_inherits_the_rest() {
        struct Cloud;
        impl SlackController for Cloud {
            fn status(&self, _: &Context<'_>) -> Option<String> {
                Some("is working…".into())
            }
            fn trailing_blocks(&self, ctx: &Context<'_>) -> Vec<Value> {
                vec![context_block(&format!(
                    "<https://admin/{}|View>",
                    ctx.session_id
                ))]
            }
        }
        assert_eq!(Cloud.status(&ctx()).as_deref(), Some("is working…"));
        // Inherited, unchanged.
        let p = Cloud.turn(&ctx(), &completed(Value::String("done".into())));
        assert_eq!(p.text, "done\n\n_1.5s_");
        assert_eq!(Cloud.step_block(&step())["title"], "search · 1.2s");
        assert_eq!(Cloud.trailing_blocks(&ctx()).len(), 1);
    }

    /// The whole frontend is reachable: every surface the bot shows a user has
    /// a method, and a deployment can restyle all of them without touching
    /// delivery.
    #[test]
    fn every_visible_surface_goes_through_the_seam() {
        struct Rebrand;
        impl SlackController for Rebrand {
            fn turn(&self, _: &Context<'_>, _: &TurnCompleted) -> Rendered {
                Rendered {
                    text: "answer".into(),
                    blocks: vec![section_block("answer")],
                }
            }
            fn step_block(&self, s: &StepView<'_>) -> Value {
                section_block(&format!("step {}", s.name))
            }
            fn step_chunk(&self, s: &StepView<'_>) -> Value {
                serde_json::json!({ "type": "task_update", "id": s.id, "title": "…" })
            }
            fn said_block(&self, t: &str) -> Value {
                section_block(&format!("said {t}"))
            }
            fn elided_block(&self, n: usize) -> Value {
                context_block(&format!("{n} hidden"))
            }
            fn prompt_blocks(&self, p: &PromptView<'_>) -> Vec<Value> {
                vec![section_block(p.message)]
            }
            fn settled_prompt_blocks(&self, _: &[Value], _: &str, r: &str) -> Vec<Value> {
                vec![context_block(r)]
            }
            fn status(&self, _: &Context<'_>) -> Option<String> {
                None
            }
            fn trailing_blocks(&self, _: &Context<'_>) -> Vec<Value> {
                vec![context_block("brand")]
            }
            fn cancelled(&self, _: &Context<'_>, _: &[Value]) -> Option<Rendered> {
                Some(Rendered {
                    text: "stopped".into(),
                    blocks: vec![section_block("stopped")],
                })
            }
        }

        let p = Rebrand;
        assert_eq!(p.turn(&ctx(), &completed(Value::Null)).text, "answer");
        assert_eq!(
            p.cancelled(&ctx(), &[]).map(|r| r.text).as_deref(),
            Some("stopped")
        );
        assert_eq!(p.step_block(&step())["text"]["text"], "step search");
        assert_eq!(p.step_chunk(&step())["title"], "…");
        assert_eq!(p.said_block("hi")["text"]["text"], "said hi");
        assert_eq!(p.elided_block(3)["elements"][0]["text"], "3 hidden");
        assert_eq!(
            p.prompt_blocks(&PromptView {
                message: "q",
                options: &[],
                expires_at: None
            })
            .len(),
            1
        );
        assert_eq!(
            p.settled_prompt_blocks(&[], "q", "done")[0]["elements"][0]["text"],
            "done"
        );
        assert_eq!(
            p.status(&ctx()),
            None,
            "a deployment can clear the indicator"
        );
        assert_eq!(p.trailing_blocks(&ctx())[0]["elements"][0]["text"], "brand");
    }

    /// A card that ended with no result is a fact the fold records; what it
    /// reads as, and whether it reads as anything, is the controller's.
    #[test]
    fn work_that_ended_with_no_result_is_a_status_not_a_sentence() {
        let ended = StepView {
            status: StepStatus::Cancelled,
            output: None,
            took: Some("2.0s"),
            ..step()
        };
        // Slack knows three states, so the card is not still going.
        let card = DefaultController.step_block(&ended);
        assert_eq!(card["status"], "error");
        assert_eq!(
            card["output"]["elements"][0]["elements"][1]["text"],
            "Cancelled."
        );

        // A deployment reads the status, not the sentence.
        struct Own;
        impl SlackController for Own {
            fn step_block(&self, s: &StepView<'_>) -> Value {
                match s.status {
                    StepStatus::Cancelled => section_block(&format!("{} — abandoned", s.name)),
                    _ => DefaultController.step_block(s),
                }
            }
        }
        assert_eq!(Own.step_block(&ended)["text"]["text"], "search — abandoned");
        assert_eq!(Own.step_block(&step())["status"], "complete");
    }

    /// The controller composes the whole message, so it can reorder or drop
    /// what the bot hands it.
    #[test]
    fn a_controller_owns_the_final_composition() {
        struct AnswerFirst;
        impl SlackController for AnswerFirst {
            fn message(
                &self,
                ctx: &Context<'_>,
                t: &TurnCompleted,
                activity: &[Value],
            ) -> Option<Rendered> {
                let answer = self.turn(ctx, t);
                let mut blocks = answer.blocks;
                blocks.extend(activity.iter().cloned());
                Some(Rendered {
                    text: answer.text,
                    blocks,
                })
            }
        }
        let activity = vec![section_block("a card")];
        let m = AnswerFirst
            .message(&ctx(), &completed(Value::String("done".into())), &activity)
            .expect("posts");
        assert_eq!(m.blocks[0]["text"]["text"], "done", "the answer leads");
        assert_eq!(m.blocks.last().unwrap()["text"]["text"], "a card");

        // The default puts the work first and its own trailing blocks last.
        let m = DefaultController
            .message(&ctx(), &completed(Value::String("done".into())), &activity)
            .expect("posts");
        assert_eq!(m.blocks[0]["text"]["text"], "a card");
        assert_eq!(m.blocks[1]["text"]["text"], "done");
    }

    /// A turn a controller has nothing to say about posts nothing at all.
    #[test]
    fn a_controller_can_stay_quiet() {
        struct Silent;
        impl SlackController for Silent {
            fn message(&self, _: &Context<'_>, _: &TurnCompleted, _: &[Value]) -> Option<Rendered> {
                None
            }
        }
        assert!(Silent
            .message(&ctx(), &completed(Value::Null), &[])
            .is_none());
    }

    /// A button the controller invented comes back to it, and what it returns
    /// is what the bot does.
    #[tokio::test]
    async fn a_controller_answers_its_own_button() {
        struct Billing;
        #[async_trait]
        impl SlackController for Billing {
            async fn on_action(&self, a: &Action<'_>) -> ActionOutcome {
                match a.action_id {
                    "upgrade" => ActionOutcome::Reply(Rendered {
                        text: format!("<@{}> take it up here", a.user),
                        blocks: vec![section_block("checkout")],
                    }),
                    "retry" => ActionOutcome::Submit("try that again".into()),
                    _ => ActionOutcome::Ignored,
                }
            }
        }
        let action = |id| Action {
            action_id: id,
            value: "{}",
            tenant_id: "t",
            session_id: "slack:C1:1.0",
            user: "U9",
            channel: "C1",
            message_ts: "2.0",
            thread_ts: "1.0",
        };
        assert!(matches!(
            Billing.on_action(&action("upgrade")).await,
            ActionOutcome::Reply(r) if r.text.contains("U9")
        ));
        assert!(matches!(
            Billing.on_action(&action("retry")).await,
            ActionOutcome::Submit(t) if t == "try that again"
        ));
        assert!(matches!(
            Billing.on_action(&action("whatever")).await,
            ActionOutcome::Ignored
        ));
        // The engine's own controller has no buttons to answer.
        assert!(matches!(
            DefaultController.on_action(&action("upgrade")).await,
            ActionOutcome::Ignored
        ));
    }

    #[test]
    fn elapsed_crosses_into_minutes() {
        assert_eq!(format_elapsed(Duration::from_millis(950)), "0.9s");
        assert_eq!(format_elapsed(Duration::from_secs(75)), "1m 15s");
    }
}
