# elixir-plug-chat-with-tools

A chattable agent with two tools in Elixir, served with [Plug](https://hexdocs.pm/plug)
and [Bandit](https://hexdocs.pm/bandit). Decision requests are pattern-matched
as plain maps, so no type generation is needed.

## Run

Install the CLI:

```sh
npm i -g @substructure.ai/cli
```

Two terminals.

**1. Start the worker**:

```sh
mix deps.get
mix run --no-halt
```

**2. Send a message with the CLI**

```sh
export ANTHROPIC_API_KEY=sk-ant-...
subs run -c substructure.toml --agent my-agent "what time is it in my timezone?"
```
