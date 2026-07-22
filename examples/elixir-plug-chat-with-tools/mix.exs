defmodule Worker.MixProject do
  use Mix.Project

  def project do
    [app: :worker, version: "0.1.0", elixir: "~> 1.16", deps: deps()]
  end

  def application do
    [mod: {Worker.Application, []}, extra_applications: [:logger]]
  end

  defp deps do
    [
      {:bandit, "~> 1.5"},
      {:plug, "~> 1.16"},
      {:jason, "~> 1.4"}
    ]
  end
end
