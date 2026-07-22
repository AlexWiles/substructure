defmodule Worker.Application do
  use Application
  require Logger

  def start(_type, _args) do
    Logger.info("worker listening on http://localhost:4444")
    children = [{Bandit, plug: Worker.Router, port: 4444}]
    Supervisor.start_link(children, strategy: :one_for_one)
  end
end
