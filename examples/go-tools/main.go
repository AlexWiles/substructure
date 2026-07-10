// A chat agent with a tool, served with net/http.
package main

import (
	"encoding/json"
	"net/http"
	"time"
)

// DecisionRequest is what the engine POSTs; we read only what we act on.
type DecisionRequest struct {
	Trigger  Trigger         `json:"trigger"`
	Proposed json.RawMessage `json:"proposed"` // opaque; echoed back verbatim
}

type Trigger struct {
	Type string `json:"type"`
	Name string `json:"name"` // on tool.execute
}

// Decision is what the worker returns; every field is optional.
type Decision struct {
	Agent   *AgentConfig `json:"agent,omitempty"`
	Actions []any        `json:"actions,omitempty"`
}

// AgentConfig is the agent's identity; the engine reads it to propose the LLM request.
type AgentConfig struct {
	Model  string `json:"model"`
	Stream bool   `json:"stream"`
	Tools  []Tool `json:"tools,omitempty"`
}

// Tool is a model-facing tool. Only name/description reach the model; exec is
// unexported, so it stays out of the agent config the engine reads.
type Tool struct {
	Name        string `json:"name"`
	Description string `json:"description"`
	exec        func() any
}

// ToolResult answers a tool.execute; the engine fills the id from the trigger.
type ToolResult struct {
	Type   string `json:"type"`
	Result any    `json:"result"`
}

var tools = []Tool{
	{
		Name:        "get_current_time",
		Description: "Get the current UTC date and time.",
		exec:        func() any { return time.Now().UTC().Format(time.RFC3339) },
	},
	{
		Name:        "get_current_time_zone",
		Description: "Get the user's current timezone",
		exec:        func() any { zone, _ := time.Now().Zone(); return zone },
	},
}

func decide(req DecisionRequest) any {
	if req.Trigger.Type == "session.start" {
		// The engine will use this agent config to generate proposed actions.
		return Decision{Agent: &AgentConfig{
			Model:  "claude-haiku-4-5-20251001",
			Stream: true,
			Tools:  tools,
		}}
	}

	// Run our tool when the model calls it.
	if req.Trigger.Type == "tool.execute" {
		for _, t := range tools {
			if t.Name == req.Trigger.Name {
				return Decision{Actions: []any{ToolResult{Type: "tool.result", Result: t.exec()}}}
			}
		}
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
