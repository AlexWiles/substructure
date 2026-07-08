// A complete chat agent. No SDK, no dependencies — Go's net/http.
// The engine POSTs a decision request; this returns the next actions.
//
// The worker accepts every decision the engine has a default for (`proposed`)
// and authors only the one that is genuinely its own: the LLM request — the
// agent's identity. Everything else — model replies and model failures — flows
// through the proposed-first line at the top.
//
// Point a local Substructure server at it:
//
//	substructure serve --dev --provider anthropic --worker-url http://localhost:4444
package main

import (
	"encoding/json"
	"net/http"
)

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

	// The client sent the conversation → record it, prompt the model.
	if t["type"] == "client.messages" {
		return map[string]any{
			"messages": t["messages"],
			"actions": []any{map[string]any{
				"type": "llm.call", "stream": true,
				"request": map[string]any{"model": "claude-haiku-4-5-20251001", "messages": t["messages"]}}},
		}
	}

	return map[string]any{"actions": []any{}}
}
