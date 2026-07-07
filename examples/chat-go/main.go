// A complete chat agent. No SDK, no dependencies — Go's net/http.
// The engine POSTs a decision request; this returns the next actions.
//
// Point a local Substructure server at it:
//
//	substructure serve --dev --provider openrouter --worker-url http://localhost:4444
package main

import (
	"encoding/json"
	"net/http"
)

func main() {
	http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		var req map[string]any
		json.NewDecoder(r.Body).Decode(&req)
		t := req["trigger"].(map[string]any)
		out := map[string]any{"actions": []any{}}

		switch t["type"] {
		// The client sent the conversation → prompt the model.
		case "client.messages":
			out["messages"] = t["messages"]
			out["actions"] = []any{map[string]any{
				"type": "llm.call", "handler": "server", "stream": true,
				"request": map[string]any{"model": "anthropic/claude-sonnet-4-6", "messages": t["messages"]}}}

		// The model answered → record it, end the turn.
		case "llm.finished":
			msg := t["message"].(map[string]any)
			out["messages"] = append(req["messages"].([]any), msg)
			out["actions"] = []any{map[string]any{"type": "done", "data": msg["content"]}}
		}

		json.NewEncoder(w).Encode(out)
	})

	http.ListenAndServe(":4444", nil)
}
