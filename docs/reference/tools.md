# Tools and content

::: orkexa.Tool

## Content sources

A tool can declare where the content it returns comes from: `@Tool(content_source="folder")`. The label is short,
lower case (`[a-z][a-z0-9_]{0,31}`). The agent reports it on every call as `ToolCall.content_source`, in
`before_tool` and `after_tool`, so a host can record what entered the model's context, for example to stop
outbound calls once content of a granted source is in a conversation. Orkexa itself does not act on it.

| Tool | `content_source` |
|---|---|
| your `@Tool` | what you declare (default `None`) |
| the tool of a `FileStore` (`files=`) | `files` (`orkexa.files.FILES_CONTENT_SOURCE`) |
| the search tool of an `Index` (`index=`) | `index` (`orkexa.retrieval.INDEX_CONTENT_SOURCE`) |
| the memory tool (`memory=`) | the backend's `content_source`, `memory` unless a subclass or the instance sets `None` |
| an unknown tool, an MCP tool | `None` |

The declaration is information for the host, not a proof: an undeclared tool can still return such content.

::: orkexa.image

::: orkexa.document

::: orkexa.uploaded_file

::: orkexa.ToolContent
