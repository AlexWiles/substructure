// A complete chat agent served with net/http.
package main

import (
	"encoding/json"
	"net/http"
)

// DecisionRequest is what the engine POSTs; we read only what we act on.
type DecisionRequest struct {
	Trigger  Trigger         `json:"trigger"`
	Proposed json.RawMessage `json:"proposed"` // opaque; echoed back verbatim
}

type Trigger struct {
	Type string `json:"type"`
}

// Decision is what the worker returns; every field is optional.
type Decision struct {
	Agent *AgentConfig `json:"agent,omitempty"`
}

// AgentConfig is the agent's identity; the engine reads it to propose the LLM request.
type AgentConfig struct {
	Model  string `json:"model"`
	Stream bool   `json:"stream"`
}

func decide(req DecisionRequest) any {
	if req.Trigger.Type == "session.start" {
		// The engine will use this agent config to generate proposed actions.
		return Decision{Agent: &AgentConfig{Model: "claude-haiku-4-5-20251001", Stream: true}}
	}

	// Accept the engine's proposal for every other decision.
	return req.Proposed
}

func main() {
	http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		var req DecisionRequest
		json.NewDecoder(r.Body).Decode(&req)
		w.Header().Set("content-type", "application/json")
		json.NewEncoder(w).Encode(decide(req))
	})

	http.ListenAndServe(":4444", nil)
}
