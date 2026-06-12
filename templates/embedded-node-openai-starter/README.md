# embedded-node-openai-starter

The smallest possible agent: no server, no worker, no control plane. The
Substructure runtime runs in-process and persists the session and event log to a
local SQLite file (`agent.db`), so a turn interrupted partway through resumes by
replaying the log on restart.

The model is the OpenAI generator (`openaiGenerate`), called directly, and
Substructure owns the tool loop. The example wires one tool and runs a single
turn — add your own tools and turns from there.

## Run

```sh
export OPENAI_API_KEY=sk-...
npm install
npm start
```
