import "@assistant-ui/react-markdown/styles/dot.css";

import { MarkdownTextPrimitive } from "@assistant-ui/react-markdown";
import { memo } from "react";
import remarkGfm from "remark-gfm";

// Renders an assistant text part as markdown (GFM: tables, lists, code, etc.).
// Plugged into <Thread> as the assistant message's Text component.
export const MarkdownText = memo(() => (
    <MarkdownTextPrimitive remarkPlugins={[remarkGfm]} className="aui-md" />
));
