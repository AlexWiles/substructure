// A complete chat agent in Go, served with net/http. Types are generated from
// schemas/protocol.schema.json (see README).
package main

import (
	"encoding/json"
	"log"
	"net/http"
)

func decide(req DecisionRequest) DecisionResponse {
	// The whole worker: accept the engine's proposal for every decision. The
	// agent is declared in subs.toml, and arrives as the
	// `session.start` proposal, so there is nothing to author here.
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
