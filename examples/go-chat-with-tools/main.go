// A chat agent with two tools, served with net/http. Types are generated from
// schemas/protocol.schema.json (see README).
package main

import (
	"encoding/json"
	"log"
	"net/http"
	"time"
)

type tool struct {
	name        string
	description string
	exec        func() string
}

var tools = []tool{
	{
		name:        "get_current_time",
		description: "Get the current UTC date and time.",
		exec:        func() string { return time.Now().UTC().Format(time.RFC3339) },
	},
	{
		name:        "get_current_time_zone",
		description: "Get the user's current timezone",
		exec: func() string {
			zone, _ := time.Now().Zone()
			return zone
		},
	},
}

func decide(req DecisionRequest) *DecisionResponse {
	if req.Trigger.Type == SessionStart {
		// The engine will use this agent config to generate proposed actions.
		stream := true
		agentTools := make([]AgentTool, len(tools))
		for i, t := range tools {
			description := t.description
			agentTools[i] = AgentTool{Name: t.name, Description: &description}
		}
		return &DecisionResponse{
			Agent: &AgentConfig{
				Model:  "claude-haiku-4-5-20251001",
				Stream: &stream,
				Tools:  agentTools,
			},
		}
	}

	// Run our tool when the model calls it.
	if req.Trigger.Type == ToolExecute && req.Trigger.Name != nil {
		for _, t := range tools {
			if t.name == *req.Trigger.Name {
				return &DecisionResponse{
					Actions: []DecisionAction{{Type: FluffyToolResult, Result: t.exec()}},
				}
			}
		}
	}

	// Accept the engine's proposal for every other decision.
	return req.Proposed
}

func main() {
	http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		var req DecisionRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(decide(req))
	})

	log.Println("worker listening on http://localhost:4444")
	log.Fatal(http.ListenAndServe(":4444", nil))
}
