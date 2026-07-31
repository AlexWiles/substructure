# @substructure.ai/cli

The CLI for [Substructure](https://github.com/substructureai/substructure), a durable runtime for AI agents. Runs the orchestration server.

## Install

```sh
npm i -g @substructure.ai/cli
```

This pulls a platform-specific binary for your OS and architecture.

Supported platforms: macOS (arm64, x64), Linux (arm64, x64).

## Usage

```sh
subs serve --no-auth --provider openrouter --worker-url http://localhost:4444
```

Run `subs --help` for the full command list. Cloud management commands (`apps`, `keys`, `webhook`, `sessions`, …) are top-level subcommands.

## Links

- [Repository and documentation](https://github.com/substructureai/substructure)
- [SDK (`@substructure.ai/sdk`)](https://www.npmjs.com/package/@substructure.ai/sdk)
- [Embedded runtime (`@substructure.ai/runtime`)](https://www.npmjs.com/package/@substructure.ai/runtime)

## License

[FSL-1.1-ALv2](./LICENSE). Converts to Apache 2.0 two years after each release.
