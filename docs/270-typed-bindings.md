---
title: Typed bindings
group: Reference
---

The worker protocol comes with machine-readable schemas. Point a code generator
at them to get typed request and response types in your language.

You do not write the wire types yourself.

## Files

Both files are in [`schemas/`](../schemas). They are generated from the engine's
Rust types.

| File | Holds |
| --- | --- |
| `protocol.schema.json` | Every wire type, in one JSON Schema. The top-level type is `Protocol`. |
| `worker.openapi.json` | The worker endpoint: `POST /`, from `DecisionRequest` to `DecisionResponse`. |

## Generate the types

Point a generator at `protocol.schema.json`. The examples keep the output as
`protocol.ts`, `protocol.go`, or `protocol.py`.

TypeScript and Go, with [quicktype](https://github.com/glideapps/quicktype):

```sh
npx quicktype --src-lang schema --lang typescript \
    --src schemas/protocol.schema.json --top-level Protocol \
    --just-types --prefer-unions -o protocol.ts
```

Go takes neither `--just-types` nor `--prefer-unions`:

```sh
npx quicktype --src-lang schema --lang go --package main \
    --src schemas/protocol.schema.json --top-level Protocol -o protocol.go
```

Python, with
[datamodel-code-generator](https://github.com/koxudaxi/datamodel-code-generator):

```sh
datamodel-codegen \
    --input schemas/protocol.schema.json --input-file-type jsonschema \
    --output-model-type pydantic_v2.BaseModel --disable-timestamp --output protocol.py
```

For OpenAPI tools, generate from `worker.openapi.json` instead.

## Next steps

- [Protocol](./230-protocol.md): the types these schemas describe.
- [Workers](./50-workers.md): the code that uses them.
