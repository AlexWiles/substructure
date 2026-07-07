# A complete chat agent. No SDK — one Plug handler.
# The engine POSTs a decision request; this returns the next actions.
#
# Point a local Substructure server at it:
#   substructure serve --dev --provider openrouter --worker-url http://localhost:4444
Mix.install([:bandit, :plug, :jason])

defmodule Worker do
  use Plug.Router

  plug Plug.Parsers, parsers: [:json], json_decoder: Jason
  plug :match
  plug :dispatch

  post "/" do
    t = conn.body_params["trigger"]

    decision =
      case t["type"] do
        # The client sent the conversation → prompt the model.
        "client.messages" ->
          %{"messages" => t["messages"], "actions" => [%{
            "type" => "llm.call", "handler" => "server", "stream" => true,
            "request" => %{"model" => "anthropic/claude-sonnet-4-6", "messages" => t["messages"]}}]}

        # The model answered → record it, end the turn.
        "llm.finished" ->
          %{"messages" => conn.body_params["messages"] ++ [t["message"]],
            "actions" => [%{"type" => "done", "data" => t["message"]["content"]}]}

        _ -> %{"actions" => []}
      end

    send_resp(conn, 200, Jason.encode!(decision))
  end
end

require Logger
Logger.info("worker listening on http://localhost:4444")
{:ok, _} = Bandit.start_link(plug: Worker, port: 4444)
Process.sleep(:infinity)
