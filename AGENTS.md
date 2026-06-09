### Rust crates
```
crates/core              → Rust agent-loop engine, server, CLI
crates/napi              → N-API bindings exposing core to Node
```

### TypeScript packages
```
@substructure.ai/sdk     → TypeScript SDK: clients, worker, framework adapters
@substructure.ai/cli     → npm wrapper shipping the Rust CLI binary
@substructure.ai/runtime → In-process Substructure runtime for Node (napi build)
```

### Coding guidelines:
Only use comments to explain code that is unclear. No narrative comments, no comments that restate code behavior that is clear in the code.

### Changes:
Keep CHANGELOG.md up to date with a brief description of changes as they relate to the previous release.

