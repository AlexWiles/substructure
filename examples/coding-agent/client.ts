import {
    TUI,
    Text,
    Markdown,
    Editor,
    Loader,
    ProcessTerminal,
} from "@mariozechner/pi-tui";
import type { Event } from "@substructure.ai/sdk";

// ── Config ───────────────────────────────────────────────────────────────────

const SERVER_URL = process.env.SERVER_URL ?? "http://localhost:3001";
const sessionId = crypto.randomUUID();

// ── ANSI helpers ─────────────────────────────────────────────────────────────

const dim = (s: string) => `\x1b[2m${s}\x1b[22m`;
const bold = (s: string) => `\x1b[1m${s}\x1b[22m`;
const cyan = (s: string) => `\x1b[36m${s}\x1b[39m`;
const green = (s: string) => `\x1b[32m${s}\x1b[39m`;
const yellow = (s: string) => `\x1b[33m${s}\x1b[39m`;
const red = (s: string) => `\x1b[31m${s}\x1b[39m`;
const gray = (s: string) => `\x1b[90m${s}\x1b[39m`;
const white = (s: string) => `\x1b[37m${s}\x1b[39m`;
const italic = (s: string) => `\x1b[3m${s}\x1b[23m`;
const underline = (s: string) => `\x1b[4m${s}\x1b[24m`;
const strikethrough = (s: string) => `\x1b[9m${s}\x1b[29m`;

// ── Theme ────────────────────────────────────────────────────────────────────

const markdownTheme = {
    heading: bold,
    link: cyan,
    linkUrl: dim,
    code: yellow,
    codeBlock: white,
    codeBlockBorder: dim,
    quote: italic,
    quoteBorder: dim,
    hr: dim,
    listBullet: cyan,
    bold: bold,
    italic: italic,
    strikethrough: strikethrough,
    underline: underline,
};

const editorTheme = {
    borderColor: dim,
    selectList: {
        selectedPrefix: green,
        selectedText: green,
        description: dim,
        scrollInfo: dim,
        noMatch: dim,
    },
};

// ── TUI Setup ────────────────────────────────────────────────────────────────

const terminal = new ProcessTerminal();
const tui = new TUI(terminal);

const header = new Text(
    `${bold(cyan("coding-agent"))} ${dim("— session " + sessionId.slice(0, 8))}`,
    1,
    0,
);
tui.addChild(header);
tui.addChild(new Text(dim("─".repeat(60)), 1, 0));

// ── State ────────────────────────────────────────────────────────────────────

let isProcessing = false;
let loader: Loader | null = null;

// ── Helpers ──────────────────────────────────────────────────────────────────

function addUserMessage(text: string) {
    tui.addChild(new Text(`${bold(green("▶ you"))}`, 1, 0));
    tui.addChild(new Text(text, 2, 0));
    tui.addChild(new Text("", 0, 0));
}

function addAssistantMessage(content: string) {
    tui.addChild(new Text(`${bold(cyan("● assistant"))}`, 1, 0));
    tui.addChild(new Markdown(content, 2, 0, markdownTheme));
    tui.addChild(new Text("", 0, 0));
}

function addToolCall(name: string, args: string) {
    const argsShort = args.length > 120 ? args.slice(0, 120) + "…" : args;
    tui.addChild(
        new Text(`  ${yellow("⚡")} ${bold(yellow(name))} ${dim(argsShort)}`, 0, 0),
    );
}

function addToolResult(name: string, result: string) {
    const resultShort = result.length > 200 ? result.slice(0, 200) + "…" : result;
    tui.addChild(
        new Text(`  ${green("✓")} ${dim(name)} ${gray(resultShort)}`, 0, 0),
    );
}

function addToolError(name: string, error: string) {
    tui.addChild(
        new Text(`  ${red("✗")} ${dim(name)} ${red(error)}`, 0, 0),
    );
}

function addError(message: string) {
    tui.addChild(new Text(`${red("error:")} ${message}`, 1, 0));
}

function showLoader(message: string) {
    if (loader) return;
    loader = new Loader(tui, cyan, dim, message);
    tui.addChild(loader);
    loader.start();
}

function hideLoader() {
    if (loader) {
        loader.stop();
        tui.removeChild(loader);
        loader = null;
    }
}

// ── Server Communication ─────────────────────────────────────────────────────

async function submitMessage(message: string) {
    const res = await fetch(`${SERVER_URL}/submit`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message, sessionId }),
    });

    if (!res.ok) {
        throw new Error(`Server error: ${res.status} ${res.statusText}`);
    }

    const reader = res.body!.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop()!;

        for (const line of lines) {
            if (!line.trim()) continue;
            try {
                const event = JSON.parse(line) as Event;
                handleEvent(event);
            } catch {
                // skip malformed lines
            }
        }
    }
}

function handleEvent(event: Event) {
    const p = event.payload;

    switch (p.type) {
        case "tool.call.requested":
            addToolCall(p.name, p.arguments);
            break;

        case "tool.call.completed":
            addToolResult(p.name, p.result);
            break;

        case "tool.call.errored":
            addToolError(p.name, p.error);
            break;

        case "message.new":
            if (p.message.role === "assistant" && p.message.content) {
                hideLoader();
                const content =
                    typeof p.message.content === "string"
                        ? p.message.content
                        : p.message.content
                              .filter((part) => part.type === "text")
                              .map((part) => (part as { text: string }).text)
                              .join("\n");
                if (content) {
                    addAssistantMessage(content);
                }
            }
            break;

        case "llm.call.requested":
            showLoader("Thinking...");
            break;

        case "llm.call.errored":
            hideLoader();
            addError(`LLM error: ${p.error}`);
            break;

        case "turn.completed":
            hideLoader();
            break;
    }
}

// ── Input Handler ────────────────────────────────────────────────────────────

const editor = new Editor(tui, editorTheme);
editor.onSubmit = async (text: string) => {
    const trimmed = text.trim();
    if (!trimmed || isProcessing) return;

    if (trimmed === "/quit" || trimmed === "/exit") {
        tui.stop();
        process.exit(0);
    }

    isProcessing = true;
    editor.disableSubmit = true;
    editor.setText("");

    addUserMessage(trimmed);

    try {
        await submitMessage(trimmed);
    } catch (err) {
        hideLoader();
        addError(String(err));
    } finally {
        isProcessing = false;
        editor.disableSubmit = false;
    }
};

tui.addChild(
    new Text(
        dim("Type a message and press Enter to send. /quit to exit."),
        1,
        0,
    ),
);

tui.addChild(editor);
tui.setFocus(editor);

// ── Start ────────────────────────────────────────────────────────────────────

tui.addInputListener((data) => {
    if (data === "\x03") { // Ctrl+C
        tui.stop();
        process.exit(0);
    }
    return undefined;
});

tui.start();
