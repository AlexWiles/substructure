# @substructure.ai/cli

The CLI for [Substructure](https://substructure.ai), a durable runtime for AI agents. Runs the orchestration server.

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

- [substructure.ai](https://substructure.ai)
- [Repository and documentation](https://github.com/substructureai/substructure)
- [SDK (`@substructure.ai/sdk`)](https://www.npmjs.com/package/@substructure.ai/sdk)

## License

[MIT](./LICENSE)
