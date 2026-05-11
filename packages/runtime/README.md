# @substructure.ai/runtime

The in-process runtime for [Substructure](https://github.com/substructureai/substructure), a durable runtime for AI agents. NAPI bindings to the Rust core.

Use this if you want to run the agent runtime embedded in your Node.js process instead of running a separate server. For the standalone server, use [`@substructure.ai/cli`](https://www.npmjs.com/package/@substructure.ai/cli).

## Install

```sh
npm i @substructure.ai/runtime
```

This pulls a platform-specific native binding for your OS and architecture.

Supported platforms: macOS (arm64, x64), Linux (arm64, x64).

## Usage

Typically used through [`@substructure.ai/sdk`](https://www.npmjs.com/package/@substructure.ai/sdk):

```js
import { Substructure } from "@substructure.ai/sdk";
import { EmbeddedRuntime } from "@substructure.ai/runtime";

const runtime = new EmbeddedRuntime({ db: "agent.db" });
const sub = new Substructure({ runtime });
```

See the [main repository](https://github.com/substructureai/substructure) for full documentation and examples.

## License

[FSL-1.1-ALv2](./LICENSE). Converts to Apache 2.0 two years after each release.
