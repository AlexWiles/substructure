# A chat agent with two tools, served with Plug + Bandit. JSON is handled
# dynamically, so no type generation is needed.
defmodule Worker.Router do
  use Plug.Router

  plug Plug.Parsers, parsers: [:json], pass: ["application/json"], json_decoder: Jason
  plug :match
  plug :dispatch

  defp tools do
    [
      %{
        name: "get_current_time",
        description: "Get the current UTC date and time.",
        exec: fn -> DateTime.utc_now() |> DateTime.to_iso8601() end
      }
    ]
  end

  # The engine will use this agent config to generate proposed actions.
  # The declared config arrives as the proposal; keep its "llm" and "model" and
  # add only what this worker knows.
  defp decide(%{"trigger" => %{"type" => "session.start"}, "proposed" => proposed}) do
    agent =
      Map.merge(proposed["agent"] || %{}, %{
        "stream" => true,
        "tools" => Enum.map(tools(), &%{name: &1.name, description: &1.description})
      })

    %{agent: agent}
  end

  # Run our tool when the model calls it.
  defp decide(%{"trigger" => %{"type" => "tool.execute", "name" => name}}) do
    tool = Enum.find(tools(), &(&1.name == name))
    %{actions: [%{type: "tool.result", result: tool.exec.()}]}
  end

  # Accept the engine's proposal for every other decision.
  defp decide(%{"proposed" => proposed}), do: proposed

  post "/" do
    conn
    |> put_resp_content_type("application/json")
    |> send_resp(200, Jason.encode!(decide(conn.body_params)))
  end

  match _ do
    send_resp(conn, 404, "not found")
  end
end
