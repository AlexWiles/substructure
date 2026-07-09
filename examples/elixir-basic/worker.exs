# A complete chat agent. No SDK — one Plug handler.
# The engine POSTs a decision request; this returns the next actions.
#
# The worker accepts every decision the engine has a default for (`proposed`)
# and authors only the one that is genuinely its own: the LLM request — the
# agent's identity. Everything else — model replies and model failures — flows
# through the proposed-first clause at the top.
#
# Point a local Substructure server at it:
#   subs serve --dev --provider anthropic --worker-url http://localhost:4444
Mix.install([:bandit, :plug, :jason])

defmodule Worker do
  use Plug.Router

  plug Plug.Parsers, parsers: [:json], json_decoder: Jason
  plug :match
  plug :dispatch

  post "/" do
    send_resp(conn, 200, Jason.encode!(decide(conn.body_params)))
  end

  # The engine proposes actions for a default agent tool loop
  defp decide(%{"proposed" => proposed}) when is_map(proposed), do: proposed
  defp decide(%{"trigger" => t}), do: trigger(t)

  # The client sent the conversation → record it, prompt the model.
  defp trigger(%{"type" => "client.messages", "messages" => messages}) do
    %{"messages" => messages, "actions" => [%{
      "type" => "llm.call", "stream" => true,
      "request" => %{"model" => "claude-haiku-4-5-20251001", "messages" => messages}}]}
  end

  defp trigger(_), do: %{"actions" => []}
end

require Logger
Logger.info("worker listening on http://localhost:4444")
{:ok, _} = Bandit.start_link(plug: Worker, port: 4444)
Process.sleep(:infinity)
