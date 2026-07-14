---
title: Typed bindings
group: Reference
---

The worker protocol is published as machine-readable schemas. Point a code
generator at them for typed request and response bindings in your language,
instead of hand-writing the wire types.

## Artifacts

Both live in [`schemas/`](../schemas) and are generated from the engine's Rust
types.

| File | Covers |
| --- | --- |
| `protocol.schema.json` | Every wire type, one JSON Schema. Top-level type `Protocol`. |
| `worker.openapi.json` | The worker endpoint: `POST /`, `DecisionRequest` to `DecisionResponse`. |

## Generating

Point a generator at `protocol.schema.json`. The examples commit the output as
`protocol.ts`, `protocol.go`, or `protocol.py`.

TypeScript and Go, with [quicktype](https://github.com/glideapps/quicktype):

```sh
npx quicktype --src-lang schema --lang typescript \
    --src schemas/protocol.schema.json --top-level Protocol \
    --just-types --prefer-unions -o protocol.ts
```

Swap `--lang go --package main -o protocol.go` for Go.

Python, with [datamodel-code-generator](https://github.com/koxudaxi/datamodel-code-generator):

```sh
datamodel-codegen \
    --input schemas/protocol.schema.json --input-file-type jsonschema \
    --output-model-type pydantic_v2.BaseModel --output protocol.py
```

For OpenAPI tooling, generate from `worker.openapi.json` instead.

## Next

- [Protocol](./150-protocol.md): the types the schemas describe.
- [Core concepts](./20-concepts.md): what the worker exchanges.
