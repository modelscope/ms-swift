# Agent deployment examples

`server.sh` starts a tool-capable model with `swift deploy`; `client.py` demonstrates
the request format with a weather tool response.

## Parallel Search MCP

`parallel_search.py` gives a deployed model web search and page fetching through
[Parallel Search MCP](https://docs.parallel.ai/integrations/mcp/search-mcp). It
discovers the tool schemas over Streamable HTTP, executes the model's tool calls,
and sends results back until the model answers, with at most six model calls by
default. This is a separate, non-streaming client example.

Search uses Fast mode. No Parallel account or API key is needed. The anonymous
service has lower rate limits and is intended for exploration and light use.
The example does not read Parallel credentials. Model deployment still needs
the usual SWIFT dependencies and hardware.

From the repository root, install the client dependencies (Python 3.10+):

```shell
python -m venv .venv-parallel
source .venv-parallel/bin/activate
pip install -r examples/deploy/agent/requirements-parallel.txt
```

Check search and fetch without a model server:

```shell
python examples/deploy/agent/parallel_search.py --search "ms-swift deployment agent templates"
python examples/deploy/agent/parallel_search.py --fetch https://swift.readthedocs.io/en/latest/
```

In a separate environment with [SWIFT installed](../../../docs/source_en/GetStarted/SWIFT-installation.md),
start the existing agent server:

```shell
bash examples/deploy/agent/server.sh
```

Then run the client:

```shell
python examples/deploy/agent/parallel_search.py \
    --question "Search for ms-swift's deployment documentation and explain how to serve a model. Cite URLs."
```

Use `--base-url` for another SWIFT server and `--model` for its served model ID.
The model must support tool calling, with an appropriate `--agent_template` on
the server. Whether it searches, fetches, or answers directly depends on the model.
Tool errors and timeouts stop the example; it does not retry rate-limited calls.
Only `web_search` and `web_fetch` are dispatched. Web content is sent to the model
as tool results, so do not include private information in search questions.

Run the client tests after installing the dependencies:

```shell
python -m unittest discover -s examples/deploy/agent -p 'test_parallel_search.py'
```
