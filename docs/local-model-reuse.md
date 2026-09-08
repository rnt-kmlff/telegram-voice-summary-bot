# Reusing the local Qwen model

The same Qwen3.5 model served by Ollama can support this Telegram bot and other local text-processing tools. Each tool supplies its own input and instructions; sharing the model does not share the bot's messages or give another tool Telegram access.

## Start with the summary command

This project's reusable `OllamaSummarizer` is separate from the Telegram message handler. The command below summarizes a UTF-8 text file through the configured Ollama model:

```sh
uv run voice-summary summarize transcript.txt --language auto
```

Use `--language en` or `--language ru` to request the output language. The text command does not transcribe audio or send a Telegram message. Ollama must be running and the configured model available.

Codex can use this command for a bounded task: select the relevant text, obtain a draft summary, and check the result against the source. Future helpers could extract stated dates or action items, classify messages, or draft short text. Those additional task modes are not implemented here.

## Three integration routes

| Route | Meaning | Status |
| --- | --- | --- |
| Local command | Codex calls the summary command and reviews its output. | Provided by this project; use within the active task's permissions. |
| Local MCP tool | A small adapter exposes the summarizer as a named tool. Codex remains the coordinating model. | Possible future extension; no MCP server or Codex configuration is installed by this project. |
| Codex using a local provider | A separate Codex CLI session uses Ollama as its model provider. | Officially documented provider route; this project's Qwen3.5 compatibility has not been tested. |

Codex supports local MCP servers launched as processes, alongside HTTP servers. That makes an adapter a practical future route for narrowly scoped local operations. See [official MCP documentation](https://learn.chatgpt.com/docs/extend/mcp).

For the provider route, official documentation describes `--oss`, `--local-provider ollama`, and custom model providers. This configures a Codex session's model connection; it does not by itself register Qwen in the desktop model picker. See [Advanced Configuration](https://learn.chatgpt.com/docs/config-file/config-advanced#oss-mode-local-providers).

## Native subagent support is unverified

Official documentation also describes custom subagents with model and session settings. Whether this particular Qwen3.5/Ollama setup can be selected and reliably used as a native subagent in the current desktop environment needs a separate compatibility test. This project makes no such claim and changes no Codex settings. See [official subagent documentation](https://learn.chatgpt.com/docs/agent-configuration/subagents#custom-agents).

## Keep delegated work small and reviewable

Use local helpers for clear transformations with supplied source text. Require missing information to remain missing, preserve names and numbers, and have Codex check the result before using it. The helper should return text or structured data; it should not independently send messages, publish changes, or perform other external actions.

Latency, output quality, and overall cost savings are not guaranteed. Model size, input length, other work on the computer, and Codex's review effort all matter. Benchmark representative English and Russian examples before expanding the helper's role. When the computer sleeps, local processing pauses.

Documentation checked on 2026-09-08.
