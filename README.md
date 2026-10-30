*Simulated systems for the IT Helpdesk Agent brief of the Agentic AI Bootcamp, organized by Jerusalem High-Tech Foundry (JHF) · in partnership with COMCEC.*

# IT Helpdesk kit

Everything your agent needs around it, so your week goes into the agent:

- the **dataset** of Nakhla Systems, a fictional software company: 20 knowledge-base documents
  (2 of them PDF-only), 40 historical tickets, 25 employees, a status page;
- the company's **helpdesk API** (simulated system of record, FastAPI, port **8102**);
- an **MCP server** in front of it (stdio, 6 tools);
- a **UI starter**, the **eval runner** and 10 **sample cases**.

You build the agent: retrieval over the knowledge base, the conversation, the approval gate,
your API contract on port 8000. Do not change the simulated systems: the hidden tests run
against the original.

## Quickstart (Windows PowerShell)

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
copy .env.example .env                                  # then fill in OPENROUTER_API_KEY

uvicorn services.helpdesk_api:app --port 8102           # terminal 1: the helpdesk API
python check_kit.py                                     # terminal 2: must end with "Kit ready."
python dataset/validate_dataset.py                      # optional: dataset consistency
```

macOS/Linux: `python3.12 -m venv .venv`, `source .venv/bin/activate`, `cp .env.example .env`.
Docker instead of uvicorn: `docker compose up --build` (same API on port 8102).

One venv is enough for the kit **and** your agent: `requirements.txt` also pins the course's agent
framework (`langgraph==1.2.10`, `openai==2.52.0`) and `mcp==1.28.1`, the MCP version CrewAI 1.15 needs.
`requirements-dev.txt` adds pytest. For a runtime-only install, use `pip install -r requirements.txt`.

## Layout

| Path | What |
|---|---|
| `dataset/` | knowledge base, tickets, schemas, manifest, precedence rules: start with `dataset/README.md` |
| `services/helpdesk_api.py` | the simulated helpdesk API (system of record) |
| `services/helpdesk_mcp.py` | the MCP server your agent uses (`python -m services.helpdesk_mcp`) |
| `services/data/` | seed data: `employees.json` (directory), `status.json` (status page) |
| `services/Dockerfile`, `docker-compose.yml` | container for the API; add your app to the compose file |
| `ui/app_ui.py` | Gradio starter: chat, sources panel, ticket-draft card with Approve / Reject. Talks only to your API. |
| `evals/` | `cases.sample.jsonl` (10 cases) and `run_evals.py` (the shared runner) |
| `check_kit.py` | smoke test: API health, a keyed REST call, MCP `list_tools` + one read call |

## The helpdesk API (port 8102)

Business endpoints need the header `X-API-Key` (value of `HELPDESK_API_KEY`). Errors are JSON:
`{"error": {"code": "...", "message": "...", "fields": [...]}}` with real HTTP codes.
Interactive docs: http://localhost:8102/docs

| Method | Path | What it does | Errors |
|---|---|---|---|
| GET | `/health` | `{"status": "ok", "service": "helpdesk"}` (no key) | |
| GET | `/api/directory/{email}` | employee: department, office, VIP, contractor, MFA, VPN client and gateway, account status | 401, 404 `employee_not_found` |
| GET | `/api/status` | status page: services and incidents (`active_incidents` lists the ids) | 401 |
| GET | `/api/tickets?requester_email=&q=&status=&limit=` | search the archive and new tickets; `q` = keywords in subject and description; `status` = open, resolved, closed; `limit` 1-50 (default 10) | 401, 422 |
| GET | `/api/tickets/{ticket_id}` | one ticket with notes | 401, 404 `ticket_not_found` |
| POST | `/api/tickets` | create a ticket; body = `dataset/schemas/ticket.schema.json`; header `Idempotency-Key`; returns 201 and the ticket with `ticket_id` IT-2001, IT-2002, ... | 401, 400, 409 `idempotency_conflict`, 422 `validation_error` with `fields` |
| POST | `/api/tickets/{ticket_id}/notes` | add a note `{"note": "..."}`; header `Idempotency-Key` | 401, 404, 409, 422 |

Ticket body rules: `category` from the 8 categories, `priority` P1-P4, `requester_email` must be in
the directory, `department` must match the directory, `attempted_steps` needs at least one step,
every `kb_refs` id must exist (`[]` is allowed when the KB has no answer), optional `affected_users`.
The API sets `assigned_team` from the category and adds `warnings` if you cite a deprecated or draft document.
Same `Idempotency-Key` + same body = the same ticket, never a duplicate.

### Test-only admin endpoints (no key)

They exist for the eval runner and the hidden tests. **Your app must not call them.**

| Method | Path | What it does |
|---|---|---|
| POST | `/admin/reset` | restore seed data, clear faults and logs. **Persisted tickets, notes and ids are kept** (see "Permanent ticket storage") |
| POST | `/admin/faults` | `{"target": "create_ticket", "mode": "error_503", "count": 1}`: the next `count` calls of that operation fail |
| GET / DELETE | `/admin/faults` | show / clear armed faults |
| GET | `/admin/log` | recent requests: method, path, status, ms, operation |
| GET | `/admin/tickets` | tickets created through the API, persisted in SQLite (`?all=1`: archive too) |
| GET | `/admin/notes` | notes added through the API, persisted in SQLite |

Fault targets: `create_ticket`, `directory`, `status`, `search_tickets`, `get_ticket`, `add_note`.
Modes: `error_500`, `error_503`, `timeout` (waits 15 s, then 504), `slow` (waits 3 s, then succeeds).

```powershell
curl.exe -X POST http://localhost:8102/admin/faults -H "Content-Type: application/json" `
  -d '{\"target\": \"create_ticket\", \"mode\": \"error_503\", \"count\": 1}'
curl.exe -X POST http://localhost:8102/admin/reset
```

## The MCP server (stdio)

Your app starts it as a subprocess and talks MCP over stdin/stdout. It calls the REST API, so
its state is the state the tests see. Tool errors come back as `{"ok": false, "error": {...}}`.

| Tool | Use it to |
|---|---|
| `lookup_employee(email)` | get department, VIP, contractor, MFA, VPN client version and gateway **before** applying department rules or choosing a priority |
| `get_service_status()` | check for an active incident **before** troubleshooting or drafting a ticket |
| `search_tickets(query=None, requester_email=None, status=None, limit=5)` | the requester's ticket history (your long-term memory) and similar past cases |
| `create_ticket(subject, description, category, priority, requester_email, department, attempted_steps, kb_refs, idempotency_key, affected_users=1)` | **write**: only from your approval step, after the employee approved this exact draft |
| `get_ticket(ticket_id)` | one ticket with its notes |
| `add_ticket_note(ticket_id, note)` | **write**: add information to the employee's own ticket |

```python
import os, sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

params = StdioServerParameters(command=sys.executable, args=["-m", "services.helpdesk_mcp"],
                               env=dict(os.environ))   # pass env on, or the server will not see your .env values
async with stdio_client(params) as (read, write):
    async with ClientSession(read, write) as session:
        await session.initialize()
        result = await session.call_tool("lookup_employee", {"email": "tariq.hamdan@nakhla.example"})
```

The knowledge base is **not** an MCP tool: ingestion, chunking, embeddings and retrieval over
`dataset/knowledge_base/` (Markdown and PDF) are yours to build, for example OpenRouter
`openai/text-embedding-3-small` with a numpy cosine store, or Chroma.

## The approval gate

Keep `create_ticket` (and `add_ticket_note`) **out of the model's tool list**. The application
prepares a ticket draft from the conversation and verified directory/status results; `/api/chat`
returns it as `pending_action` (`type: "create_ticket"`). Only `POST /api/approve` calls
`create_ticket` over MCP with exactly that payload and `idempotency_key = action_id`. Show only
the `ticket_id` the API returned. A rejected draft writes nothing.

Ticket intake starts only after an explicit request to prepare/open a ticket; ordinary
troubleshooting remains a normal chat turn. The app asks follow-up questions for missing facts,
checks a supplied Nakhla work email against the employee directory, and checks service status
before registering a draft. A matching active incident is directed to its existing workaround
instead of producing an individual ticket. The generated payload is checked against
`dataset/schemas/ticket.schema.json` before it is returned or stored as a pending action.

The current UI and `/api/chat` contract accept an email address from the user but do not provide
an authenticated identity. Directory lookup therefore confirms that the employee record and
department exist and match the supplied address; it does not prove that the caller owns that
address. Deployments that need caller identity assurance must add trusted authentication rather
than treating the chat email field as proof of identity.

## Local ticket metadata and JSON export

The app uses Python's built-in SQLite support to retain a local reference after the external
helpdesk API confirms ticket creation. By default the database is
`services/data/helpdesk_metadata.sqlite3`; set `HELPDESK_METADATA_DB_PATH` to an absolute path
or a path relative to this project directory to change it. The app creates the database and
tables on startup. SQLite stores only the approval action's idempotency key, the external ticket
ID, local creation time, status, and a SHA-256 digest of the validated ticket payload. It does
not store another editable copy of ticket bodies, credentials, or API keys.

The helpdesk API remains the authority for actual ticket records. The SQLite record is local
tracking metadata and cannot make a database transaction atomic with the remote API. If the
remote ticket is created while local persistence is unavailable, the response identifies the
ticket and reports the local persistence failure; retrying the same approved action does not
submit a second ticket.
The simulated helpdesk API now persists its own tickets; see "Permanent ticket storage" below.
This metadata database does not hold ticket bodies.

`dataset/schemas/ticket.schema.json` remains the JSON Schema used to validate ticket payloads;
it is not ticket data. JSON ticket data is not maintained as a second live store. To create a
point-in-time JSON export of the local ticket-reference metadata, run this from the project
directory:

```powershell
python -m services.export_ticket_metadata exports/ticket-references.json
```

The export uses a versioned metadata format and contains references and payload digests, not
the ticket body. SQLite remains the local source for this metadata; the JSON file is a
regenerable export. The generated database and its SQLite sidecar files are ignored by Git.

Run the app after starting the helpdesk API and configuring `.env`:

```powershell
uvicorn app:app --port 8000
```

Run the test suite with:

```powershell
python -m pytest -v
```

SQLite is appropriate for this single-instance training app. A multi-instance production
deployment should use a shared transactional database and an authenticated identity provider;
the remote helpdesk system should remain authoritative for ticket content. A production
deployment also needs an operational backup/retention policy for local metadata and a recovery
strategy for remote-success/local-database-failure cases.

## Permanent ticket storage (simulated helpdesk API)

The simulated helpdesk API is the system of record for tickets in this kit. It stores complete
ticket records, notes and idempotency records in its own SQLite database,
`services/data/helpdesk_tickets.sqlite3` by default (`HELPDESK_TICKET_DB_PATH`, absolute or relative
to this folder). It must not be the same file as the app metadata database. Tables:
`tickets`, `ticket_notes`, `idempotency_records`, `id_counter`. Writes use one transaction each;
a storage failure returns `503 storage_unavailable` and changes nothing.

- `dataset/tickets/historical_tickets.json` stays a read-only seed archive (IT-1001...). Search and
  retrieval combine it with database tickets. Notes added to archived tickets are stored in the
  database and overlaid on read; the archive file is never rewritten.
- New ids start at IT-2001 and only move forward: they survive restarts and are never reused.
- Starting the API never deletes or resets data. `POST /admin/reset` reloads the seed directory,
  status page, faults and log but **keeps tickets, notes and ids**. It only wipes the ticket database
  when `HELPDESK_ALLOW_DESTRUCTIVE_RESET=1` **and** `HELPDESK_TICKET_DB_PATH` is set explicitly, so
  evals/tests that need a clean slate must use a temporary database path.
- Tickets created by the earlier in-memory API were never saved; their bodies cannot be rebuilt
  (the metadata database holds only references and digests).
- If a real external helpdesk is configured instead of this simulator, that system stays
  authoritative; do not use this database as a second editable ticket store.

Approval durability: pending (non test-only) ticket drafts, the user's decision and the write status
are stored in an additive `approval_actions` table of the app metadata database, so an approval can
complete after an app restart. The session id is kept only as a SHA-256 hash (a different session
still cannot approve), the draft is checked against its stored digest on load, the draft body is
removed once the ticket is created and recorded, and test-only drafts are never stored. A write that
was interrupted is retried with the same idempotency key, so no duplicate ticket is created.
`/api/chat` still never creates a ticket.

Three different JSON things:

| File | Purpose |
|---|---|
| `dataset/schemas/ticket.schema.json` | JSON Schema contract (not data) |
| `python -m services.export_ticket_history out.json [--include-archive]` | **full** ticket-history export, format `helpdesk-ticket-history-export-v1`, every record validated against `$defs/ticket_record` |
| `python -m services.export_ticket_metadata out.json` | metadata-only export, format `helpdesk-ticket-reference-export-v1` (no bodies) |

Fields such as category, status, department, requester, resolution, `resolved_at` and `created_at` are
stored as indexed columns for later search, trend and knowledge-learning work (not implemented here).
Limitations: remote and local writes are not atomic; intake follow-up conversation state is still in
memory; in Docker, mount a volume for `services/data` or the ticket database is lost with the container.

## Ticket intelligence API (read-only, token-gated)

`services/ticket_intelligence.py` analyses existing tickets with deterministic local code: no LLM, no
embeddings, no MCP and no network calls. `app.py` exposes it through three routes that are **disabled
(404) unless `HELPDESK_INTELLIGENCE_TOKEN` is set** and are never called from `/api/chat`.

| Route (POST, JSON body) | Returns |
|---|---|
| `/api/intelligence/duplicates` | candidate duplicate tickets: ticket ID, status, matching reasons and evidence (shared error codes, text similarity, same requester, days apart) |
| `/api/intelligence/trends` | observed counts per period and group, time range, sample size, data source, limitations |
| `/api/intelligence/knowledge-proposals` | draft, **unverified** KB-improvement proposals with source ticket IDs, KB reference checks and uncertainty |

Local-only usage (put the token in your own, gitignored `.env`; never commit it):

```powershell
$env:HELPDESK_INTELLIGENCE_TOKEN = "choose-a-long-random-value"
python -m uvicorn app:app --host 127.0.0.1 --port 8000
curl.exe -X POST http://127.0.0.1:8000/api/intelligence/trends -H "Authorization: Bearer $env:HELPDESK_INTELLIGENCE_TOKEN" -H "Content-Type: application/json" -d "{}"
```

Guarantees: read-only (tickets, archive, Knowledge Base and approval state are never written); results keep
`automatic_action: none`, limitations and uncertainty; proposals are always `verified: false`,
`publishable: false`, `requires_human_review: true` and never include resolution text; duplicate results
contain no descriptions or requester emails; trend and proposal results never include predictions or a
significance claim. Data comes from the historical archive plus `HELPDESK_TICKET_DB_PATH` (an absent ticket
database is read as empty and is never created). Data failures return a generic 503.

Security limitations: a single shared bearer token is the only control. It identifies no individual user,
is not a role system, and gives no audit trail; `session_id` and self-asserted emails are not authorization.
Use it on localhost or behind a trusted network or reverse proxy with TLS, and rotate it by changing the
variable and restarting. There is no rate limiting. Duplicate detection is a conservative signal and misses
duplicates that share no error code and use different wording; conflict flags on proposals are a word-overlap
heuristic; database-created tickets carry no resolution yet, so proposals currently come from archived tickets.

## Chat memory (short-term, long-term, history)

Implemented in `services/chat_memory.py`, stored in its own SQLite file
`services/data/chat_memory.sqlite3` (override with `CHAT_MEMORY_DB_PATH`; gitignored). It is
separate from the ticket and approval databases.

- **Conversation history (short-term):** every `/api/chat` turn stores the user message and the
  final *verified* assistant reply with a timestamp. Only the last 6 messages are sent to the
  model; at most 20 messages per session and 500 sessions are kept (oldest removed first). A
  request retried within 10 seconds with the same message and reply is not stored twice.
  History survives restarts, so a clarifying question and the user's answer stay connected.
- **Long-term memory:**
  - *User facts*: operating system, preferred language and answer style (concise/detailed), taken
    from fixed vocabularies only. Each fact records whether it was `explicit` (stated by the user)
    or `inferred` (uncertain, e.g. PowerShell implies Windows), plus created/updated times. A newer
    explicit statement replaces an older value; an inference never overwrites an explicit fact.
    Style and language apply to every answer; environment facts are offered only for technical
    requests.
  - *Resolution notes*: a verified, cited answer is staged, and saved only when the user then says
    it worked ("that worked", "problem solved"). Abstentions, partially verified or uncited answers
    and questions are never saved. A note is offered back only when it overlaps the new message.
- **Safety:** passwords, tokens, API keys and MFA codes are redacted from history and block
  facts and notes. Quoted/pasted text is ignored. Memory reaches the model only as an untrusted
  context message: it is not a knowledge-base source, never reaches the answer verifier as
  evidence, and cannot override KB precedence, system rules or ticket approval. Memory never
  creates or approves tickets.
- **Delete:** `POST /api/memory/forget` with `{"session_id": ...}` removes that session's history,
  facts and notes.
- **Identity limitation:** the app has no authentication (`user.email` in `/api/chat` is not
  verified), so memory is scoped to the SHA-256 hash of `session_id`. A new session starts empty,
  and anyone who knows a session ID could read or delete that session's memory, as with its
  pending approvals. There is no cross-session personalization.
- **Tests:** `python -m pytest -q` (offline; uses temporary databases only).

## Guardrails AI

`services/guardrails_service.py` uses the real `guardrails-ai` framework (`Guard` plus custom
`Validator`s, pinned to 0.10.0; never 0.10.1, which was a malicious release). It adds a layer; the
Answer Verifier, KB precedence, citation checks, memory, approval gate and tool-boundary
authorization are unchanged and still run.

| Guard | Runs | Validators |
|---|---|---|
| input | `/api/chat` before ticket intake or tools, and again at the start of `run_agent` | `PromptInjection`, `SecretRequest`, `PrivilegeClaim` (incl. approval bypass), `OtherUserData` |
| context | on every retrieved chunk before it can reach the model | `PromptInjection` (same rules as `answer_verifier.looks_like_injection`) |
| output | after the Answer Verifier, before the reply is returned | `SecretLeak`, `PersonalData`, `GroundedIdentifiers`, `Citations`, `EvidenceAndPolicy` |

- **All checks are deterministic** (regex rules, the manifest and the existing verifier). No model-based or
  Guardrails Hub validator is used, so no extra API key, hosted service or network call is needed.
  Framework telemetry is switched off in code.
- **Failure handling:** a blocked input returns a fixed refusal (`state: "blocked_by_guardrails"`) and
  nothing else runs. A failed output is replaced by a fixed fallback that never repeats the blocked text
  (a secret or contact leak gets "I can't share that information"; unsupported content gets the usual
  honest fallback). A fabricated `[ID]` marker is removed if the rest still passes. An exception,
  timeout (`GUARDRAILS_TIMEOUT_SECONDS`, default 5) or malformed value fails closed; the message is not
  logged.
- **Authorization is not decided by text.** The input guard only refuses unmistakable bypass attempts.
  Access to tickets is still enforced at `/api/tickets`/MCP and the approval gate. There is still no
  authentication, so a claimed identity is never trusted.
- **Limits:** regex rules can be evaded by rephrasing and may block unusual legitimate wording; the app's
  own ticket-draft replies are not passed through the output guard; the guard cannot prove an answer
  true, only that it is consistent with the retrieved evidence.
- **Install/test:** `pip install -r requirements-dev.txt`, then `python -m pytest tests/test_guardrails.py -q`.
  Tests use deterministic checks and test doubles where framework invocation needs isolation; no API key
  or Guardrails Hub service is needed. The compatible pins use Guardrails AI 0.10.0, Gradio 5.50.0,
  Pydantic 2.12.3, and Hugging Face Hub 0.36.0. Gradio 6.30 conflicts with Guardrails 0.10.0's
  declared `click<=8.2.0` and `typer<0.20` constraints; these pins were resolved and checked together.

## Environment variables

| Variable | Default | Used by |
|---|---|---|
| `HELPDESK_API_KEY` | `dev-key-change-me` | API, MCP server, check_kit (must be the same everywhere) |
| `HELPDESK_API_URL` | `http://localhost:8102` | MCP server, check_kit, your app |
| `HELPDESK_TODAY` | real today | API: date of new tickets and of the status page (`YYYY-MM-DD`) |
| `HELPDESK_TIMEOUT_S` | `20` | MCP server: seconds before a call to the API gives up |
| `HELPDESK_MCP_LOG_LEVEL` | `WARNING` | MCP server log level on stderr |
| `HELPDESK_METADATA_DB_PATH` | `services/data/helpdesk_metadata.sqlite3` | app: local SQLite ticket-reference metadata and approval state |
| `HELPDESK_TICKET_DB_PATH` | `services/data/helpdesk_tickets.sqlite3` | API: persistent ticket database |
| `HELPDESK_ALLOW_DESTRUCTIVE_RESET` | unset | API: `1` lets `/admin/reset` wipe an explicitly configured ticket DB (tests only) |
| `HELPDESK_INTELLIGENCE_TOKEN` | unset (routes disabled) | app: bearer token enabling the read-only `/api/intelligence/*` routes |
| `HELPDESK_DATA_DIR`, `HELPDESK_DATASET_DIR` | `services/data`, `dataset` | API seed files |
| `OPENROUTER_API_KEY`, `OPENROUTER_BASE_URL`, `MODEL`, `EMBEDDING_MODEL` | see `.env.example` | your agent |
| `APP_URL` | `http://localhost:8000` | UI starter, eval runner |

The API and the MCP server also read the kit's `.env` (without overriding variables you already set).

## UI starter and evals

```powershell
python ui/app_ui.py                          # http://localhost:7860, talks to APP_URL
python evals/run_evals.py                    # the 10 sample cases against your app on :8000
python evals/run_evals.py --only S2-05 -v    # one case, every turn
# Live and optional (real model calls): same cases on throw-away servers (8110/8112)
# with temporary databases, so your real tickets are never read or reset.
python evals/run_isolated_evals.py           # writes evals/isolated_scorecard.md/.json
```

The case format and the API contract are described in the mini-project's `common/CONTRACT.md`.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `check_kit.py`: API not answering | start `uvicorn services.helpdesk_api:app --port 8102` from this folder |
| 401 `unauthorized` | the same `HELPDESK_API_KEY` for the API, the MCP server and your app |
| MCP tool returns `unreachable` | the API is not running, or `HELPDESK_API_URL` is wrong (in Docker use the service name) |
| Everything fails with 503 | a fault is still armed: `curl.exe -X DELETE http://localhost:8102/admin/faults` |
| Ticket ids keep growing | expected: ids persist and are never reused. To start clean, point `HELPDESK_TICKET_DB_PATH` at a new file |
