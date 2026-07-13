### Rust crates
```
crates/core              → Rust agent-loop engine, server, CLI
crates/napi              → N-API bindings exposing core to Node
```

### TypeScript packages
```
@substructure.ai/cli     → npm wrapper shipping the Rust CLI binary
@substructure.ai/runtime → In-process Substructure runtime for Node (napi build)
```

### Coding guidelines:
Keep comments to an absolute minimum and use as few words as possible.

### CHANGELOG.md:
Keep CHANGELOG.md up to date. There should be a maximum of two sentences per entry. First sentence states the problem in as few words as possible. Second sentence states the solution in as few words as possible.

