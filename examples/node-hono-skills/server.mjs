// Agent Skills: load a skill's instructions (and tools) on demand, not up front.
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

const catalog = Object.entries(SKILLS).map(([name, s]) => `- ${name}: ${s.description}`).join("\n");

const load_skill = {
    name: "load_skill",
    description: "Load a skill's instructions before doing the task it covers.",
    input: { type: "object", properties: { name: { type: "string" } }, required: ["name"] }
};

// A loaded skill adds its instructions (as the tool result) and unlocks its tools.
// `llm` and `model` name what substructure.toml declares: a config built
// outside a decision cannot inherit them from the proposal.
const agent = (loaded) => ({
    llm: "claude",
    model: "claude-haiku-4-5-20251001",
    stream: true,
    system: `You have skills you can load when a task matches one. Load it before starting.\n\n${catalog}`,
    tools: [load_skill, ...loaded.flatMap((name) => SKILLS[name].tools)]
});

function decide({ trigger, proposed, state }) {
    state = state ?? { loaded: [] };

    if (trigger.type === "session.start") {
        return { agent: agent(state.loaded), state };
    }

    if (trigger.type === "tool.execute") {
        if (trigger.name === "load_skill") {
            const { name } = trigger.input.value;
            if (!state.loaded.includes(name)) state.loaded.push(name);

            return {
                state,
                agent: agent(state.loaded),
                actions: [{ type: "tool.result", result: SKILLS[name].body }]
            };
        }
        const tool = state.loaded
            .flatMap((name) => SKILLS[name].tools)
            .find((t) => t.name === trigger.name);
        return { actions: [{ type: "tool.result", result: tool.exec(trigger.input.value) }] };
    }

    return proposed;
}

const app = new Hono();
app.post("/", async (c) => c.json(decide(await c.req.json())));

serve({ fetch: app.fetch, port: 4444 }, () =>
    console.log("worker listening on http://localhost:4444"));
