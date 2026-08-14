// Skills as Slack buttons: every turn message ends with a row of skills, and a
// click turns one on for the thread and answers again with it.
import { serve } from "@hono/node-server";
import { Hono } from "hono";
import { readdirSync, readFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import matter from "gray-matter";

const dir = join(dirname(fileURLToPath(import.meta.url)), "skills");

// Each skill is a folder: SKILL.md holds its instructions, tools.mjs its functions.
const SKILLS = {};
for (const name of readdirSync(dir)) {
    const { data, content } = matter(readFileSync(join(dir, name, "SKILL.md"), "utf8"));
    const tools = await import(pathToFileURL(join(dir, name, "tools.mjs"))).then((m) => m.default).catch(() => []);
    SKILLS[data.name] = { description: data.description, body: content.trim(), tools };
}

const toolsOf = (active) => active.flatMap((name) => SKILLS[name].tools);

const agent = (active) => ({
    llm: "openrouter",
    model: "deepseek/deepseek-v4-flash-0731",
    system: "You are an assistant in Slack. Keep answers short.",
    tools: toolsOf(active)
});

// A click comes back as a `client.action` named after `action_id`, so the id
// carries which skill it is.
const button = (name, on) => ({
    type: "button",
    action_id: `skill:${name}`,
    text: { type: "plain_text", text: on ? `✓ ${name}` : name },
    value: name,
    style: on ? "primary" : undefined
});

// The row replaces the one a message already has, so it never grows two.
const withRow = (blocks, active) => [
    ...blocks.filter((b) => b.type !== "actions"),
    { type: "actions", elements: Object.keys(SKILLS).map((name) => button(name, active.includes(name))) }
];

// The clicked message, redrawn where it sits.
const redraw = (click, active) => ({
    channel: click.channel,
    ts: click.message_ts,
    text: click.message_text,
    blocks: withRow(click.message_blocks ?? [], active)
});

function decide({ trigger, proposed, state, messages }) {
    const active = state?.active ?? [];

    if (trigger.type === "session.start") {
        return { agent: agent(active), state: { active } };
    }

    // Slack proposes the finished message; append the row to it instead of
    // building a message of our own.
    if (trigger.type === "turn.finished") {
        const view = proposed.channels?.slack?.view;
        if (view) view.blocks = withRow(view.blocks ?? [], active);
        return proposed;
    }

    if (trigger.type === "client.action" && trigger.name.startsWith("skill:")) {
        const name = trigger.name.slice("skill:".length);
        const next = active.includes(name) ? active : [...active, name];
        // `messages` is the whole branch, so the skill goes on the end of the
        // thread. Sending it alone starts the turn on a branch of its own.
        const note = {
            role: "user",
            content: `Use this skill and answer again.\n\n${SKILLS[name].body}`
        };
        return {
            state: { active: next },
            agent: agent(next),
            messages: [...messages, note],
            actions: [{ type: "llm.call" }],
            channels: { slack: { update: redraw(trigger.args, next) } }
        };
    }

    if (trigger.type === "tool.execute") {
        const tool = toolsOf(active).find((t) => t.name === trigger.name);
        return { actions: [{ type: "tool.result", result: { content: [{ type: "text", text: tool.exec(trigger.input.value) }] } }] };
    }

    return proposed;
}

const app = new Hono();
app.post("/", async (c) => c.json(decide(await c.req.json())));

serve({ fetch: app.fetch, port: 4444 }, () =>
    console.log("worker listening on http://localhost:4444"));
