# ADR 0017 – Egress guard, web search and page fetching

Status: accepted (2026-11-14, NAL-A2; implementation approved by the owner on 2026-11-13)

## Context

Agents need two web tools that do not depend on one model vendor: a search that returns raw hits from a provider
the operator chooses, and a fetch of one page as text. The URL of a fetch comes from the model, so it is untrusted
input: without protection the tool is a server-side request forgery (SSRF) primitive that reaches cloud metadata
services, the local network and the host itself. Nodestro already has a reviewed guard for MCP connections
(nodestro-runtime `mcp/netguard.py` and `net_address.py`, master 27dd8d5); other egress (search APIs, later
connectors) needs the same protection, and Orkexa must not import Nodestro (K-33).

## Decision

### 1. `orkexa.net.guard`: the guard core (standard library only)

* **Address rules.** `address_problem(address, *, allow_loopback=False, allow_private=False)` returns `None` only
  for global unicast addresses. It refuses metadata addresses (`METADATA_ADDRESSES`) under every option, scoped
  IPv6, loopback, unspecified (`0.0.0.0`, `::`), multicast, link-local (`169.254/16`, `fe80::/10`), site-local
  `fec0::/10`, private (RFC 1918, `fc00::/7`, `0.0.0.0/8` and the other special ranges the standard library
  marks private), shared `100.64/10`, reserved and anything not global. IPv6 addresses that carry an IPv4
  address (mapped, NAT64 `64:ff9b::/96` and `64:ff9b:1::/48`, 6to4, IPv4-compatible) are judged by that IPv4
  address; Teredo `2001::/32` is refused explicitly. The reasons and the behavior without the new
  `allow_private` keyword are those of the runtime's guard, unchanged (its vector list is the first block of the
  catalog).
* **`allow_private`** opens exactly `PRIVATE_NETWORKS` (loopback, RFC 1918, `100.64/10`, `fc00::/7`) for a
  self-hosted service such as SearXNG; metadata, link-local, `0.0.0.0/8`, multicast and reserved ranges stay
  refused.
* **`EgressPolicy`** (frozen): `schemes` (default `http`, `https`), `allow_private`, `allow_hosts` (`None` = any
  public host; `*.example.com` for a domain), `deny_hosts` (added to `METADATA_HOSTS`, which every policy refuses),
  `ports` (default 80, 443; `None` = any), `max_redirects` (5), `max_bytes` (5 MiB), `timeout` (30 s for the
  whole exchange). `check_url` refuses user info, other schemes, ports, hosts with characters a DNS name cannot
  have (for example `%`), names of this computer (`localhost`, `*.localhost`), metadata names, literal addresses
  that `address_problem` refuses, and every **numeric host form** whose last label is a number (`2130706433`,
  `0x7f000001`, `127.1`, `0177.0.0.1`): `getaddrinfo` and browsers read them as IPv4 addresses. An allow list never
  allows a refused address.
* **`GuardedBackend`**: an `httpcore` network backend. It resolves the name once, checks every answer (one refused
  answer refuses the name), connects to the checked address and returns the stream; TLS uses the URL's host name
  for SNI and certificate verification. DNS rebinding therefore cannot slip in between check and connect, and each
  new connection (redirect, pool refill) resolves and checks again. Unix sockets are refused.

### 2. `orkexa.net.http`: the `httpx` client (extra `web`)

`guarded_client(policy)` returns an `httpx.AsyncClient` with `trust_env=False` (no proxy variables, no `.netrc`),
`follow_redirects=False` and `GuardedTransport` as its only transport. The transport checks each request URL with
the policy (and refuses a URL whose host two parsers read differently), asks for `Accept-Encoding: identity`,
refuses compressed answers (so the bytes counted are the bytes parsed: no decompression bombs) and cuts the body
off once more than `max_bytes` arrived, counted on the wire. `send_following_redirects` follows at most
`max_redirects` hops by hand; every hop passes the policy and the backend again. `deadline(policy)` bounds an
exchange including name resolution, connects, redirects and the body.

### 3. `orkexa.web`: search and fetch (extras `web` and `web-ddgs`)

* Models `SearchCapabilities`, `SearchRequest`, `SearchHit`, `SearchResponse` (with `notes` and `billed_units`),
  the `SearchProvider` protocol (`name`, `capabilities`, `async search()`, `async check()`) and the errors
  `SearchUnavailable`, `SearchRateLimited` (with `retry_after`), `SearchAuthFailed`, `SearchBadRequest` (all
  `SearchError`). Unsupported parameters are ignored with a note, never an error, so switching providers never
  breaks a caller.
* Adapters: `TestSearchProvider` (deterministic fixtures or generated hits, no network, no cost), `BraveSearch`,
  `TavilySearch`, `MojeekSearch`, `SearXNGSearch` and `DDGSSearch`. The HTTP adapters go through
  `guarded_client`; their default policy allows only their API host over https on 443 (SearXNG: the configured
  host and port, with `allow_private`). Error messages never contain API keys (Mojeek's key travels in the query).
  **Nothing selects a provider**: there is no registry and no default; the host constructs the one its
  configuration names.
* DDGS uses the fixed backends `duckduckgo,brave,mojeek,wikipedia` (news: `duckduckgo`, its only news backend),
  never `books`, never `extract()`, runs in a worker thread, and reports an empty answer with the note
  `provider returned no results; it may be blocking requests` because DDGS cannot tell a block from an empty
  result. Its requests go to fixed public hosts through its own HTTP stack (not the guard); no caller-chosen URL
  reaches it.
* `fetch_page(url, policy, max_bytes=5 MiB, accept=...)` returns `FetchedPage(final_url, status, content_type,
  charset, title, text, truncated, size, sha256, fetched_at, redirects)`. HTML becomes Markdown of the main
  content (trafilatura) or plain text when none is found; text stays text; PDFs need the `%PDF-` header and are
  read by `pdf_pages_isolated` (separate process with time, memory and page budgets). A longer HTML or text body
  is cut (`truncated=True`), a longer PDF is refused. Character set: byte-order mark, valid UTF-8, header,
  `<meta>`, detection (charset-normalizer), UTF-8 with replacement.

## Threat model (summary)

| Threat | Measure |
|---|---|
| SSRF to metadata, loopback, LAN | address rules on every resolved address; metadata names and addresses refused by every policy |
| Numeric and odd host spellings | numeric last label refused; DNS-name syntax only; two-parser host comparison |
| DNS rebinding | resolve once per connection, check all answers, connect to the checked address |
| Redirect to an internal target | no automatic redirects; every hop checked again; hop limit |
| Large or compressed answers | identity encoding only; streamed byte limit; PDF size and page budgets in a separate process |
| Slow servers and resolvers | one deadline over the whole exchange |
| Proxy or credential leakage from the environment | `trust_env=False`; user info in URLs refused; keys never in messages |
| Unexpected file types | media-type allow list; PDF magic bytes; no archives |

Prompt injection in fetched content, source ledgers, per-run call limits and offline refusal are the host's job
(nodestro-runtime tools, NAL-A2 steps 6-8). An independent review of `orkexa.net.guard` is due before wide use
in a server profile (D1).

## Reuse by nodestro-runtime

The runtime can replace its copies with imports and keep its behavior: `METADATA_ADDRESSES`, `METADATA_HOSTS`,
`IPAddress`, `address_problem`, `url_secret_problem`, `redact_url` and `GuardedBackend` keep the runtime's names,
reasons, messages and fields (`allow_loopback`, `resolver`, `connected`, `statuses`). Two hooks make the backend
fit: `httpcore_module="httpcore2"` for the runtime's `httpx2` stack, and an overridden `error(code, message,
**details)` that returns `NodestroError(f"mcp.{code}", ...)`. Differences: `redact_url` brackets IPv6 hosts and
returns `(invalid address)` for an invalid port instead of raising.

## Platforms

All new dependencies are pure Python or ship wheels for the six desktop targets (Linux x86-64 and arm64, macOS
arm64 and x86-64, Windows x64 and arm64): httpx, httpcore, anyio, trafilatura and ddgs are pure Python; lxml
(through trafilatura), charset-normalizer and primp (through ddgs) publish `win_arm64`, `macosx` (universal2 or
per-architecture) and `manylinux` x86-64/aarch64 wheels. The Windows ARM64 CI job installs both extras.

## Out of scope

Image and video search, crawling, JavaScript rendering, logins, persistent result caches, model-vendor search
tools, the agent tools themselves (their specs, untrusted-content markers, source ledger, costs and offline
checks live in nodestro-runtime), compressed transfer encodings, HTTP/2, and live tests against paid providers
(only with owner approval, R-3). `httpx` has had no release since 12/2024; its fork `httpx2` is watched
(docs/TODO.md).

## Consequences

* One guard for every outbound request Orkexa or a host makes; new egress (connectors, database endpoints) reuses
  `EgressPolicy` and `guarded_client` instead of growing its own checks.
* The SSRF catalog (`tests/net`, more than 40 vectors, including rebinding with changing answers, a 302 to
  `127.0.0.1`, blocked ports, size and deadline) runs on every platform without network: a loopback test server
  stands in for public addresses, and a test fixture fails any test that tries a non-loopback lookup or connect.
* The adapters are tested against handwritten answers from the providers' documentation, not recordings; the
  first live smoke test per provider may require small parser adjustments.

## Amendment: internal review (2026-11-14)

An internal review of the first implementation led to four changes; the independent external review (ORK-043)
is still open.

* HTML is turned into text in a separate process (`orkexa.web._html_worker`, started like the PDF worker), with
  CPU, file and (on Linux) memory limits and at most the rest of `policy.timeout` (at least one second). At most
  `MAX_HTML_CHARS` characters are parsed, and trafilatura gives up above `MAX_TREE_SIZE` elements. Before, a
  5 MB page could keep a thread of the caller busy for about 30 to 50 seconds after the request deadline.
* `SearXNGSearch(allow_private=False)` refuses private and loopback addresses for the instance; a multi-tenant
  host must pass it, because a host name that resolves to the operator's network would otherwise be reachable.
* `SearchResponse` drops hits whose URL is not `http`/`https`, longer than 2048 characters or contains whitespace
  or control characters (`usable_hit_url`): scraped results could otherwise carry text into later prompts.
* Deeply nested JSON from a provider is a `SearchUnavailable`, not an escaping `RecursionError`.
