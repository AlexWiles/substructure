import Substructure from "@substructure.ai/sdk";

const sub = new Substructure();
const { agent } = sub;

// ── Tools ────────────────────────────────────────────────────────────────────

const retry = {
    timeout_secs: 60,
    max_retries: 3,
    backoff_base_secs: 1,
    backoff_max_secs: 10,
};

const delay = (ms: number) => new Promise((r) => setTimeout(r, ms));

const searchWeb = agent.tool({
    name: "search_web",
    description:
        "Search the web for information on a topic. Returns a list of relevant results with titles, URLs, and snippets.",
    parameters: {
        type: "object",
        properties: {
            query: { type: "string", description: "The search query" },
        },
        required: ["query"],
    },
    execute: async (args: string) => {
        const { query } = JSON.parse(args);
        await delay(400);
        return {
            query,
            results: [
                {
                    title: `Understanding ${query} — A Comprehensive Guide`,
                    url: `https://example.com/guide/${encodeURIComponent(query)}`,
                    snippet: `An in-depth exploration of ${query}, covering history, current developments, and future directions.`,
                },
                {
                    title: `${query}: Recent Advances and Breakthroughs`,
                    url: `https://example.com/advances/${encodeURIComponent(query)}`,
                    snippet: `A review of the most significant recent developments in ${query}, including key research findings.`,
                },
                {
                    title: `The Practical Applications of ${query}`,
                    url: `https://example.com/applications/${encodeURIComponent(query)}`,
                    snippet: `How ${query} is being applied in industry and research, with real-world case studies.`,
                },
            ],
        };
    },
    retry,
});

const readArticle = agent.tool({
    name: "read_article",
    description: "Read the full content of an article given its URL. Returns the article text.",
    parameters: {
        type: "object",
        properties: {
            url: { type: "string", description: "The article URL to read" },
        },
        required: ["url"],
    },
    execute: async (args: string) => {
        const { url } = JSON.parse(args);
        await delay(300);

        const topic = decodeURIComponent(url.split("/").pop() || "the topic");

        if (url.includes("/guide/")) {
            return {
                url,
                content: `${topic} has a rich history spanning several decades. Originally emerging from theoretical research, it has evolved into a field with profound practical implications. Key milestones include early foundational work in the 1990s, rapid expansion in the 2000s, and the current era of widespread adoption. The field draws on principles from multiple disciplines, creating a uniquely interdisciplinary area of study. Researchers continue to push boundaries, with new discoveries regularly reshaping our understanding.`,
            };
        }
        if (url.includes("/advances/")) {
            return {
                url,
                content: `Recent breakthroughs in ${topic} have been remarkable. In the past year alone, researchers have achieved significant improvements in efficiency and scalability. Notable advances include new algorithmic approaches that reduce computational costs by up to 40%, novel frameworks for real-time processing, and improved methods for handling edge cases. Several major institutions have published groundbreaking papers, and industry adoption has accelerated dramatically.`,
            };
        }
        return {
            url,
            content: `The practical applications of ${topic} span numerous industries. In healthcare, it enables faster diagnosis and personalized treatment plans. In finance, it powers risk assessment and fraud detection systems. Manufacturing uses it for quality control and predictive maintenance. Education benefits from adaptive learning platforms. Government agencies leverage it for policy analysis and resource allocation. Each sector continues to discover new ways to apply these technologies.`,
        };
    },
    retry,
});

const takeNotes = agent.tool({
    name: "take_notes",
    description: "Save research notes for later reference. Use this to record key findings from articles you've read.",
    parameters: {
        type: "object",
        properties: {
            notes: {
                type: "string",
                description: "The notes to save",
            },
        },
        required: ["notes"],
    },
    execute: async (args: string) => {
        const { notes } = JSON.parse(args);
        await delay(100);
        return {
            saved: true,
            length: notes.length,
            message: "Notes saved successfully.",
        };
    },
    retry,
});

const writeSummary = agent.tool({
    name: "write_summary",
    description:
        "Write a final research summary based on your notes. Call this once you have gathered enough information.",
    parameters: {
        type: "object",
        properties: {
            topic: { type: "string", description: "The research topic" },
            summary: {
                type: "string",
                description: "The full research summary",
            },
        },
        required: ["topic", "summary"],
    },
    execute: async (args: string) => {
        const { topic, summary } = JSON.parse(args);
        await delay(200);
        return {
            topic,
            summary,
            wordCount: summary.split(/\s+/).length,
            message: "Summary written successfully.",
        };
    },
    retry,
});

// ── System Prompt ────────────────────────────────────────────────────────────

const SYSTEM_PROMPT = `You are a research assistant. When the user gives you a topic, research it step by step:

1. Use searchWeb to find relevant articles about the topic.
2. Use readArticle to read 2-3 of the most relevant results.
3. Use takeNotes to record the key findings from each article.
4. Use writeSummary to produce a concise research summary.

Be thorough but concise. Always complete all four steps.`;

// ── Agent ────────────────────────────────────────────────────────────────────

export const researchAgent = agent({ id: "research-agent" })
    .use(agent.logging("research"))
    .use(agent.state())
    .use(agent.messageHistory())
    .use(agent.systemMessage(() => SYSTEM_PROMPT))
    .use(agent.tools(() => [searchWeb, readArticle, takeNotes, writeSummary]))
    .use(
        agent.llmLoop(() => ({
            request: {
                model: "anthropic/claude-sonnet-4",
            },
            llm_client: "openrouter",
            retry,
        })),
    );
