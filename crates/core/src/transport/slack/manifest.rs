use serde_json::{json, Value};

use crate::manifest::AgentSlackConfig;

const CHANNEL_SCOPES: [&str; 3] = ["app_mentions:read", "channels:history", "groups:history"];
const DM_SCOPES: [&str; 1] = ["im:history"];
const ALWAYS: [&str; 4] = ["assistant:write", "chat:write", "files:read", "files:write"];

/// How Slack reaches the engine that will answer.
pub enum Delivery<'a> {
    Socket,
    Events {
        events_url: &'a str,
        interactions_url: &'a str,
    },
}

pub fn render(agent_id: &str, declared: &AgentSlackConfig, delivery: Delivery<'_>) -> Value {
    let name = declared.name(agent_id);
    let mut scopes: Vec<&str> = ALWAYS.to_vec();
    let mut bot_events: Vec<&str> = Vec::new();
    if declared.answers.channels() {
        scopes.extend(CHANNEL_SCOPES);
        bot_events.push("app_mention");
    }
    if declared.answers.dms() {
        scopes.extend(DM_SCOPES);
        bot_events.push("message.im");
    }
    scopes.sort_unstable();

    let mut display = json!({ "name": name });
    if let Some(description) = declared
        .description
        .as_deref()
        .map(str::trim)
        .filter(|d| !d.is_empty())
    {
        display["description"] = json!(description);
    }

    let mut features = json!({
        "bot_user": { "display_name": name, "always_online": true },
        "agent_view": {},
    });
    if declared.answers.dms() {
        features["app_home"] = json!({
            "messages_tab_enabled": true,
            "messages_tab_read_only_enabled": false,
        });
    }

    let mut subscriptions = json!({ "bot_events": bot_events });
    let mut interactivity = json!({ "is_enabled": true });
    let socket_mode = match delivery {
        Delivery::Socket => true,
        Delivery::Events {
            events_url,
            interactions_url,
        } => {
            subscriptions["request_url"] = json!(events_url);
            interactivity["request_url"] = json!(interactions_url);
            false
        }
    };

    json!({
        "display_information": display,
        "features": features,
        "oauth_config": { "scopes": { "bot": scopes } },
        "settings": {
            "event_subscriptions": subscriptions,
            "interactivity": interactivity,
            "org_deploy_enabled": false,
            "socket_mode_enabled": socket_mode,
            "token_rotation_enabled": false,
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::SlackAudience;

    fn declared(answers: SlackAudience) -> AgentSlackConfig {
        AgentSlackConfig {
            name: Some("Support".into()),
            description: Some("Answers questions".into()),
            answers,
        }
    }

    fn events() -> Delivery<'static> {
        Delivery::Events {
            events_url: "https://api.test/api/slack/apps/abc123/events",
            interactions_url: "https://api.test/api/slack/apps/abc123/interactions",
        }
    }

    fn scopes(v: &Value) -> Vec<String> {
        v["oauth_config"]["scopes"]["bot"]
            .as_array()
            .unwrap()
            .iter()
            .map(|s| s.as_str().unwrap().to_string())
            .collect()
    }

    fn bot_events(v: &Value) -> Vec<String> {
        v["settings"]["event_subscriptions"]["bot_events"]
            .as_array()
            .unwrap()
            .iter()
            .map(|s| s.as_str().unwrap().to_string())
            .collect()
    }

    #[test]
    fn a_socket_app_carries_no_url_to_deliver_to() {
        let m = render("support", &declared(SlackAudience::Both), Delivery::Socket);
        assert_eq!(m["settings"]["socket_mode_enabled"], json!(true));
        assert!(m["settings"]["event_subscriptions"]["request_url"].is_null());
        assert!(m["settings"]["interactivity"]["request_url"].is_null());
        assert_eq!(m["settings"]["interactivity"]["is_enabled"], json!(true));
    }

    #[test]
    fn a_delivered_app_carries_both_urls_and_no_socket() {
        let m = render("support", &declared(SlackAudience::Both), events());
        assert_eq!(m["settings"]["socket_mode_enabled"], json!(false));
        assert_eq!(
            m["settings"]["event_subscriptions"]["request_url"],
            json!("https://api.test/api/slack/apps/abc123/events")
        );
        assert_eq!(
            m["settings"]["interactivity"]["request_url"],
            json!("https://api.test/api/slack/apps/abc123/interactions")
        );
    }

    #[test]
    fn how_it_is_delivered_does_not_change_what_it_may_do() {
        let socket = render("support", &declared(SlackAudience::Both), Delivery::Socket);
        let delivered = render("support", &declared(SlackAudience::Both), events());
        assert_eq!(scopes(&socket), scopes(&delivered));
        assert_eq!(bot_events(&socket), bot_events(&delivered));
        assert_eq!(socket["features"], delivered["features"]);
    }

    #[test]
    fn a_bot_that_takes_dms_asks_for_the_messages_tab() {
        for delivery in [Delivery::Socket, events()] {
            let m = render("support", &declared(SlackAudience::Dm), delivery);
            assert_eq!(
                m["features"]["app_home"]["messages_tab_enabled"],
                json!(true)
            );
            assert_eq!(
                m["features"]["app_home"]["messages_tab_read_only_enabled"],
                json!(false)
            );
        }
        let m = render(
            "support",
            &declared(SlackAudience::Channels),
            Delivery::Socket,
        );
        assert!(m["features"]["app_home"].is_null());
    }

    #[test]
    fn every_bot_can_stream_its_reply() {
        for answers in SlackAudience::ALL {
            let m = render("support", &declared(answers), Delivery::Socket);
            assert!(
                scopes(&m).contains(&"assistant:write".to_string()),
                "{answers:?}"
            );
            assert_eq!(m["features"]["agent_view"], json!({}), "{answers:?}");
        }
    }

    #[test]
    fn what_it_answers_decides_what_it_asks_for() {
        let dm = render("support", &declared(SlackAudience::Dm), Delivery::Socket);
        assert_eq!(bot_events(&dm), ["message.im"]);
        assert!(scopes(&dm).contains(&"im:history".to_string()));
        assert!(!scopes(&dm).contains(&"app_mentions:read".to_string()));

        let channels = render(
            "support",
            &declared(SlackAudience::Channels),
            Delivery::Socket,
        );
        assert_eq!(bot_events(&channels), ["app_mention"]);
        assert!(!scopes(&channels).contains(&"im:history".to_string()));
        assert!(scopes(&channels).contains(&"app_mentions:read".to_string()));

        let both = render("support", &declared(SlackAudience::Both), Delivery::Socket);
        assert_eq!(bot_events(&both), ["app_mention", "message.im"]);
    }

    #[test]
    fn a_private_channel_is_readable_wherever_it_runs() {
        for delivery in [Delivery::Socket, events()] {
            let m = render("support", &declared(SlackAudience::Both), delivery);
            assert!(scopes(&m).contains(&"groups:history".to_string()));
        }
    }

    #[test]
    fn an_undeclared_name_is_the_agents_own() {
        let m = render("support", &AgentSlackConfig::default(), Delivery::Socket);
        assert_eq!(m["display_information"]["name"], json!("support"));
        assert_eq!(m["features"]["bot_user"]["display_name"], json!("support"));
        assert!(m["display_information"]["description"].is_null());
    }
}
