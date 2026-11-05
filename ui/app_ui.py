"""Gradio UI starter for the IT Helpdesk Agent - a small client of YOUR API contract.

It never talks to the helpdesk API or the LLM directly. It only calls your app:
    POST {APP_URL}/api/chat      {"session_id", "message", "user": {"email"}}
    POST {APP_URL}/api/identify  {"email"}   (read-only directory check, shows the employee's name)
    POST {APP_URL}/api/approve   {"session_id", "action_id", "approved"}
Run your app first (port 8000), then:   python ui/app_ui.py   ->  http://localhost:7860
Replace it with your own UI if you like; keep the contract.
"""
from __future__ import annotations

import os
import uuid

import gradio as gr
import httpx

APP_URL = os.getenv("APP_URL", "http://localhost:8200").rstrip("/")
NO_CARD = "_No ticket draft waiting for approval._"


def call_app(path: str, body: dict) -> dict:
    try:
        r = httpx.post(f"{APP_URL}{path}", json=body, timeout=180)
    except httpx.HTTPError as exc:
        return {"reply": f"Cannot reach your app at {APP_URL} ({exc.__class__.__name__}). Is it running?",
                "_error": True}
    try:
        data = r.json()
    except ValueError:
        data = {"reply": r.text[:500]}
    if r.status_code >= 400:
        data = {"reply": f"Your app answered HTTP {r.status_code}: {str(data)[:400]}", "_error": True}
    return data if isinstance(data, dict) else {"reply": str(data)}


def side_panels(data: dict) -> tuple[str, str, str]:
    cites = data.get("citations") or []
    cite_md = "### Sources\n" + ("\n".join(
        f"- **{c.get('id', '?')}** {c.get('title', '')}" if isinstance(c, dict) else f"- {c}" for c in cites)
        or "_No knowledge-base article cited in the last answer._")
    calls = data.get("tool_calls") or []
    tools_md = "### Tool calls (last turn)\n" + ("\n".join(
        f"- `{t.get('tool', '?')}` {'ok' if t.get('ok') else 'FAILED'}"
        + (f" · {t['ms']} ms" if t.get("ms") is not None else "")
        for t in calls if isinstance(t, dict)) or "_none_")
    result = data.get("result") or {}
    state = data.get("state")
    result_md = ""
    if result:
        result_md = "### Result\n" + "\n".join(f"- **{k}**: `{v}`" for k, v in result.items())
    if state:
        result_md += f"\n\n_Workflow state: {state}_"
    return cite_md, tools_md, result_md


def ticket_card(pending: dict | None) -> str:
    if not pending:
        return NO_CARD
    payload = pending.get("payload") or {}
    lines = ["### Ticket draft - waiting for your approval", f"**{pending.get('summary', '')}**", ""]
    for key in ("subject", "category", "priority", "department", "requester_email"):
        if key in payload:
            lines.append(f"- **{key}**: {payload[key]}")
    if payload.get("attempted_steps"):
        lines.append("- **attempted steps**: " + "; ".join(map(str, payload["attempted_steps"])))
    if "kb_refs" in payload:
        lines.append("- **KB refs**: " + (", ".join(payload["kb_refs"]) or "none"))
    if payload.get("description"):
        lines += ["", f"> {payload['description']}"]
    return "\n".join(lines)


def show(data: dict, history: list, previous: dict | None = None, keep_undecided: bool = False):
    history = history + [{"role": "assistant", "content": data.get("reply") or "(empty reply)"}]
    pending = data.get("pending_action") or None
    # Only the turn that creates a draft returns it; the backend keeps it until it is decided.
    # An ordinary chat reply must not make the card disappear while the draft is still pending.
    if pending is None and keep_undecided and previous:
        pending = previous
    # The backend still holds the draft after a transport/HTTP error, and allows retrying the
    # same approved action after a failed creation, so keep the card and buttons in those cases.
    if pending is None and previous and (data.get("_error") or data.get("state") == "ticket_creation_failed"):
        pending = previous
    cite_md, tools_md, result_md = side_panels(data)
    buttons = gr.update(interactive=bool(pending))
    return history, pending, ticket_card(pending), buttons, buttons, cite_md, tools_md, result_md


def send(message: str, history: list, session_id: str, email: str, pending: dict | None = None):
    if not message.strip():
        return ("", history, *[gr.skip()] * 7)
    body = {"session_id": session_id, "message": message}
    if email.strip():
        body["user"] = {"email": email.strip().lower()}
    history = history + [{"role": "user", "content": message}]
    return ("", *show(call_app("/api/chat", body), history, pending, keep_undecided=True))


def decide(approved: bool, history: list, session_id: str, pending: dict | None):
    if not pending:
        return show({"reply": "There is no draft to approve or reject."}, history)
    history = history + [{"role": "user", "content": "Approve" if approved else "Reject"}]
    data = call_app("/api/approve", {"session_id": session_id, "action_id": pending.get("action_id"),
                                     "approved": approved})
    return show(data, history, pending)


def new_session_id() -> str:
    return f"ui-{uuid.uuid4().hex[:10]}"


def sign_in(email: str):
    """Identify the employee through the directory and start a fresh conversation.

    Returns (profile markdown, signed-in email, session id, chat, pending, card,
    approve button, reject button). Only a directory-verified email is kept;
    anything else leaves the chat anonymous. This is identification, not a login.
    """
    address = (email or "").strip().lower()
    signed_out = ("", new_session_id(), [], None, NO_CARD, gr.update(interactive=False),
                  gr.update(interactive=False))
    if not address:
        return ("_Enter your work email to sign in._", *signed_out)
    data = call_app("/api/identify", {"email": address})
    if data.get("verified") is not True:
        reason = data.get("message") or data.get("reply") or "Email not recognised."
        return (f"⚠️ **Not signed in.** {reason}", "", *signed_out[1:])
    name = data.get("name") or address
    details = " · ".join(str(v) for v in (data.get("title"), data.get("department")) if v)
    greeting = f"Hi {name}, how can I help you today?"
    return (f"✅ **Signed in as {name}**\n\n{details}\n\n`{data['email']}`", data["email"],
            new_session_id(), [{"role": "assistant", "content": greeting}], None, NO_CARD,
            gr.update(interactive=False), gr.update(interactive=False))


def new_conversation(signed_email: str):
    """Start a new conversation; the signed-in employee's long-term memory still applies."""
    return (new_session_id(), [], None, NO_CARD, gr.update(interactive=False),
            gr.update(interactive=False))


with gr.Blocks(title="Nakhla IT Helpdesk") as demo:
    session = gr.State(new_session_id)
    signed_email = gr.State("")
    pending_state = gr.State(None)
    gr.Markdown("## Nakhla IT Helpdesk\nIT support assistant for Nakhla Systems (a fictional company). "
                f"The helpdesk system behind it is simulated. Talking to `{APP_URL}`.")
    with gr.Row():
        with gr.Column(scale=3):
            with gr.Row():
                email = gr.Textbox(label="Your work email", placeholder="name.surname@nakhla.example",
                                   scale=4)
                sign_in_btn = gr.Button("Sign in", variant="primary", scale=1)
                new_chat_btn = gr.Button("New conversation", scale=1)
            profile = gr.Markdown("_Enter your work email to sign in._")
            chat = gr.Chatbot(label="Chat", height=460)
            msg = gr.Textbox(label="Message", placeholder="Describe your IT problem, e.g. NakhlaConnect shows NX-781",
                             submit_btn=True)
        with gr.Column(scale=2):
            with gr.Group():
                card = gr.Markdown(NO_CARD, padding=True)
                with gr.Row():
                    approve_btn = gr.Button("Approve", variant="primary", interactive=False)
                    reject_btn = gr.Button("Reject", variant="stop", interactive=False)
            result = gr.Markdown("")
            cites = gr.Markdown("### Sources\n_No answer yet._")
            tools = gr.Markdown("### Tool calls (last turn)\n_none_")
    outputs = [chat, pending_state, card, approve_btn, reject_btn, cites, tools, result]
    reset_outputs = [session, chat, pending_state, card, approve_btn, reject_btn]
    sign_in_outputs = [profile, signed_email, *reset_outputs]
    email.submit(sign_in, [email], sign_in_outputs)
    sign_in_btn.click(sign_in, [email], sign_in_outputs)
    new_chat_btn.click(new_conversation, [signed_email], reset_outputs)
    msg.submit(send, [msg, chat, session, signed_email, pending_state], [msg, *outputs])
    approve_btn.click(lambda h, s, p: decide(True, h, s, p), [chat, session, pending_state], outputs)
    reject_btn.click(lambda h, s, p: decide(False, h, s, p), [chat, session, pending_state], outputs)

if __name__ == "__main__":
    demo.launch(server_port=int(os.getenv("UI_PORT", "7860")))
