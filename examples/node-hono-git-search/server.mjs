// A chat agent with tools that explore one git repo, served with Hono.
import { serve } from "@hono/node-server";
import { Hono } from "hono";
import { execFile } from "node:child_process";
import { existsSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { promisify } from "node:util";

const run = promisify(execFile);

// Edit these to point the tools at a different repo.
const REPO = "substructureai/substructure";
const CLONE_URL = `https://github.com/${REPO}.git`;
const DIR = join(tmpdir(), `subs-git-search-${REPO.replace(/\W+/g, "-")}`);
const MAX_MATCHES = 50; // grep lines returned
const MAX_FILES = 100; // list_files paths returned
const MAX_LINES = 200; // read_file window
const BUF = { maxBuffer: 1 << 24 };

// Shallow-clone the repo once at startup; reuse the clone if it's already there.
if (existsSync(DIR)) {
    console.log(`using existing clone of ${REPO} at ${DIR}`);
} else {
    await run("git", ["clone", "--depth", "1", CLONE_URL, DIR]);
    console.log(`cloned ${REPO} to ${DIR}`);
}

const git = (...args) => run("git", ["-C", DIR, ...args], BUF);

function cap(lines, max) {
    if (lines.length <= max) return { lines };
    return { lines: lines.slice(0, max), truncated: `showing first ${max} of ${lines.length}` };
}

const tools = [
    {
        name: "list_files",
        description: `List tracked files in the ${REPO} git repository, optionally filtered by a glob (e.g. "*.rs").`,
        input: {
            type: "object",
            properties: { pattern: { type: "string", description: 'optional pathspec/glob, e.g. "src/**/*.ts"' } }
        },
        async exec({ pattern }) {
            const { stdout } = await git("ls-files", ...(pattern ? ["--", pattern] : []));
            const { lines, truncated } = cap(stdout.split("\n").filter(Boolean), MAX_FILES);
            return { files: lines, ...(truncated && { truncated }) };
        }
    },
    {
        name: "search_code",
        description: `Search the source code of the ${REPO} git repository.`,
        input: {
            type: "object",
            properties: { query: { type: "string", description: "text or regex to search for" } },
            required: ["query"]
        },
        async exec({ query }) {
            try {
                const { stdout } = await git("grep", "-n", "-I", "--no-color", "-e", query);
                const { lines, truncated } = cap(stdout.split("\n").filter(Boolean), MAX_MATCHES);
                const matches = lines.map((line) => {
                    const [path, lineNo, ...rest] = line.split(":");
                    return { path, line: Number(lineNo), text: rest.join(":") };
                });
                return { matches, ...(truncated && { truncated }) };
            } catch (err) {
                if (err.code === 1 && !err.stdout) return { matches: [] }; // no matches
                throw err;
            }
        }
    },
    {
        name: "read_file",
        description: `Read a file from the ${REPO} git repository. Reads at most ${MAX_LINES} lines; pass start_line to page through large files.`,
        input: {
            type: "object",
            properties: {
                path: { type: "string", description: "repo-relative file path" },
                start_line: { type: "number", description: "1-based first line to read (default 1)" }
            },
            required: ["path"]
        },
        async exec({ path, start_line = 1 }) {
            const { stdout } = await git("show", `HEAD:${path}`);
            const all = stdout.split("\n");
            const start = Math.max(1, Math.floor(start_line));
            const end = Math.min(all.length, start + MAX_LINES - 1);
            const content = all
                .slice(start - 1, end)
                .map((text, i) => `${start + i}\t${text}`)
                .join("\n");
            return {
                path,
                total_lines: all.length,
                content,
                ...(end < all.length && { more: `stopped at line ${end}; pass start_line ${end + 1} for more` })
            };
        }
    }
];

async function decide({ trigger, proposed }) {
    if (trigger.type === "session.start") {
        return {
            agent: {
                ...proposed.agent,
                tools: tools.map(({ name, description, input }) => ({ name, description, input }))
            }
        };
    }

    // The engine only asks us to run tools we declared, so a match is guaranteed.
    if (trigger.type === "tool.execute") {
        const tool = tools.find((t) => t.name === trigger.name);
        try {
            return { actions: [{ type: "tool.result", result: { content: [{ type: "text", text: await tool.exec(trigger.input.value) }] } }] };
        } catch (err) {
            return { actions: [{ type: "tool.error", error: err.message }] };
        }
    }

    return proposed;
}


const app = new Hono();
app.post("/", async (c) => c.json(await decide(await c.req.json())));

serve({ fetch: app.fetch, port: 4444 }, () =>
    console.log("worker listening on http://localhost:4444"));
