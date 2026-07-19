// A chat agent whose tools come from an MCP server, served with net/http. The
// protocol types in protocol.go are generated from the JSON Schema (see README).
package main

import (
	"context"
	"encoding/json"
	"log"
	"net/http"
	"os"
	"os/exec"
	"strings"

	"github.com/modelcontextprotocol/go-sdk/mcp"
)

var (
	ctx      = context.Background()
	session  *mcp.ClientSession
	mcpTools []*mcp.Tool
	mcpNames = map[string]bool{}
)

func decide(req DecisionRequest) DecisionResponse {
	if req.Trigger.Type == SessionStart {
		// Offer every MCP tool to the model as-is.
		stream := true
		tools := make([]AgentTool, len(mcpTools))
		for i, t := range mcpTools {
			description := t.Description
			tools[i] = AgentTool{Name: t.Name, Description: &description, Input: t.InputSchema}
		}
		return DecisionResponse{
			Agent: &AgentConfig{Model: "claude-haiku-4-5-20251001", Stream: &stream, Tools: tools},
		}
	}

	// Forward the call to the MCP server and settle with its output.
	if req.Trigger.Type == ToolExecute && req.Trigger.Name != nil && mcpNames[*req.Trigger.Name] {
		var args interface{}
		if req.Trigger.Input != nil {
			args = req.Trigger.Input.Value
		}
		res, err := session.CallTool(ctx, &mcp.CallToolParams{Name: *req.Trigger.Name, Arguments: args})
		if err != nil {
			msg := err.Error()
			return DecisionResponse{Actions: []DecisionAction{{Type: FluffyToolError, Error: &msg}}}
		}
		var text []string
		for _, c := range res.Content {
			if t, ok := c.(*mcp.TextContent); ok {
				text = append(text, t.Text)
			}
		}
		out := strings.Join(text, "\n")
		if res.IsError {
			return DecisionResponse{Actions: []DecisionAction{{Type: FluffyToolError, Error: &out}}}
		}
		return DecisionResponse{Actions: []DecisionAction{{Type: FluffyToolResult, Result: out}}}
	}

	// Accept the engine's proposal for every other decision.
	return req.Proposed
}

func main() {
	// Connect to an MCP server over stdio and pull in its tools. Here we run the
	// filesystem server scoped to this directory, but any MCP server works.
	cwd, _ := os.Getwd()
	client := mcp.NewClient(&mcp.Implementation{Name: "subs-mcp-example", Version: "1.0.0"}, nil)
	cmd := exec.Command("npx", "-y", "@modelcontextprotocol/server-filesystem", cwd)
	var err error
	session, err = client.Connect(ctx, &mcp.CommandTransport{Command: cmd}, nil)
	if err != nil {
		log.Fatal(err)
	}
	defer session.Close()

	list, err := session.ListTools(ctx, nil)
	if err != nil {
		log.Fatal(err)
	}
	mcpTools = list.Tools
	for _, t := range mcpTools {
		mcpNames[t.Name] = true
	}

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
