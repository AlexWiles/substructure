import Substructure from "@substructure.ai/sdk";
import { readFileSync, writeFileSync, readdirSync, statSync, existsSync } from "fs";
import { resolve, relative } from "path";
import { execSync } from "child_process";

const sub = new Substructure();
const { agent } = sub;

// ── Tools ────────────────────────────────────────────────────────────────────

const WORKING_DIR = process.cwd();

const retry = {
    timeout_secs: 120,
    max_retries: 3,
    backoff_base_secs: 1,
    backoff_max_secs: 10,
};

const readFile = agent.tool({
    description: "Read the contents of a file. Returns the file content as a string.",
    parameters: {
        type: "object",
        properties: {
            path: { type: "string", description: "Relative or absolute file path to read" },
        },
        required: ["path"],
    },
    execute: (args: string) => {
        const { path } = JSON.parse(args);
        const resolved = resolve(WORKING_DIR, path);
        if (!existsSync(resolved)) {
            return { error: `File not found: ${path}` };
        }
        const content = readFileSync(resolved, "utf-8");
        return { path: relative(WORKING_DIR, resolved), content, size: content.length };
    },
    retry,
});

const writeFile = agent.tool({
    description: "Write content to a file. Creates the file if it doesn't exist, overwrites if it does.",
    parameters: {
        type: "object",
        properties: {
            path: { type: "string", description: "Relative or absolute file path to write" },
            content: { type: "string", description: "The content to write to the file" },
        },
        required: ["path", "content"],
    },
    execute: (args: string) => {
        const { path, content } = JSON.parse(args);
        const resolved = resolve(WORKING_DIR, path);
        writeFileSync(resolved, content, "utf-8");
        return { path: relative(WORKING_DIR, resolved), bytesWritten: content.length };
    },
    retry,
});

const listFiles = agent.tool({
    description: "List files and directories at a given path. Returns names, types, and sizes.",
    parameters: {
        type: "object",
        properties: {
            path: { type: "string", description: "Directory path to list (default: current directory)" },
        },
    },
    execute: (args: string) => {
        const { path = "." } = JSON.parse(args);
        const resolved = resolve(WORKING_DIR, path);
        if (!existsSync(resolved)) {
            return { error: `Directory not found: ${path}` };
        }
        const entries = readdirSync(resolved).map((name) => {
            const fullPath = resolve(resolved, name);
            try {
                const stat = statSync(fullPath);
                return {
                    name,
                    type: stat.isDirectory() ? "directory" : "file",
                    size: stat.isFile() ? stat.size : undefined,
                };
            } catch {
                return { name, type: "unknown" };
            }
        });
        return { path: relative(WORKING_DIR, resolved) || ".", entries };
    },
    retry,
});

const runCommand = agent.tool({
    description: "Execute a shell command and return its output. Use for running tests, builds, git commands, etc.",
    parameters: {
        type: "object",
        properties: {
            command: { type: "string", description: "The shell command to execute" },
        },
        required: ["command"],
    },
    execute: (args: string) => {
        const { command } = JSON.parse(args);
        try {
            const output = execSync(command, {
                cwd: WORKING_DIR,
                encoding: "utf-8",
                timeout: 30_000,
                maxBuffer: 1024 * 1024,
            });
            return { command, exitCode: 0, output: output.slice(0, 4096) };
        } catch (err: unknown) {
            const e = err as { status?: number; stdout?: string; stderr?: string };
            return {
                command,
                exitCode: e.status ?? 1,
                output: (e.stdout ?? "").slice(0, 2048),
                error: (e.stderr ?? "").slice(0, 2048),
            };
        }
    },
    retry,
});

// ── System Prompt ────────────────────────────────────────────────────────────

const SYSTEM_PROMPT = `You are an expert coding assistant with access to the local filesystem and shell.
You can read files, write files, list directories, and run shell commands.

Working directory: ${WORKING_DIR}

Guidelines:
- Read files before modifying them to understand existing code.
- When writing files, include the complete content — do not use placeholders.
- Run tests or builds after making changes to verify correctness.
- Keep explanations concise. Focus on what you changed and why.
- If a task is ambiguous, ask clarifying questions.`;

// ── Agent Handler ────────────────────────────────────────────────────────────

export const codingAgent = agent({ id: "coding-agent" })
    .use(agent.logging("coding"))
    .use(agent.state())
    .use(agent.messageHistory())
    .use(agent.systemMessage(() => SYSTEM_PROMPT))
    .use(agent.tools(() => ({ readFile, writeFile, listFiles, runCommand })))
    .use(
        agent.llmLoop(() => ({
            request: {
                model: "anthropic/claude-sonnet-4",
            },
            llm_client: "openrouter",
            retry,
        })),
    );
