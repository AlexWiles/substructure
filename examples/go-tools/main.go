// A complete chat agent with a tool. No SDK, no dependencies — Go's net/http.
// The engine POSTs a decision request; this returns the next actions.
//
// The worker accepts every decision the engine has a default for (`proposed`)
// and authors only the two that are genuinely its own: the LLM request (the
// agent's identity) and the tool execution. Everything else — tool results,
// model replies, model failures, even broken or hallucinated tool calls —
// flows through the proposed-first line at the top.
//
// Point a local Substructure server at it:
//
//	subs serve --dev --provider anthropic --worker-url http://localhost:4444
package main

import (
	"encoding/json"
	"net/http"
	"time"
)

type tool struct {
	info map[string]any
	exec func(input any) any
}

var tools = []tool{
	{
		info: map[string]any{
			"name":        "get_current_time",
			"description": "Get the current UTC date and time. Call this whenever the user asks what time or date it is.",
		},
		exec: func(input any) any { return time.Now().UTC().Format(time.RFC3339) },
	},
}

func main() {
	http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		var req map[string]any
		json.NewDecoder(r.Body).Decode(&req)
		w.Header().Set("content-type", "application/json")
		json.NewEncoder(w).Encode(decide(req))
	})

	http.ListenAndServe(":4444", nil)
}

func decide(req map[string]any) map[string]any {
	// The engine proposes actions for a default agent tool loop
	if proposed, ok := req["proposed"].(map[string]any); ok {
		return proposed
	}

	t := req["trigger"].(map[string]any)

	switch t["type"] {
	// The client sent the conversation → record it, prompt the model.
	case "client.messages":
		infos := make([]any, len(tools))
		for i, tool := range tools {
			infos[i] = tool.info
		}
		return map[string]any{
			"messages": t["messages"],
			"actions": []any{map[string]any{
				"type": "llm.call", "stream": true,
				"request": map[string]any{"model": "claude-haiku-4-5-20251001", "messages": t["messages"], "tools": infos}}},
		}

	// A declared tool call with valid arguments → run it, answer.
	// Invalid tool calls will have a proposal from the engine already
	case "tool.execute":
		for _, tool := range tools {
			if tool.info["name"] == t["name"] {
				input := t["input"].(map[string]any)
				return map[string]any{"actions": []any{map[string]any{"type": "tool.result", "result": tool.exec(input["value"])}}}
			}
		}
	}

	return map[string]any{"actions": []any{}}
}
