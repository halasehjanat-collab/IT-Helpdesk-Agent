"""A pending draft must stay visible and actionable after later chat turns (offline)."""
from unittest.mock import AsyncMock

import pytest

import app as app_module
from tests.test_ui import ui  # noqa: F401  (fixture)


def _chat_reply(monkeypatch, text="Sure, anything else?"):
    monkeypatch.setattr(
        app_module,
        "run_agent",
        AsyncMock(return_value={"reply": text, "citations": [], "tool_calls": []}),
    )


def test_ordinary_chat_reply_keeps_the_pending_draft(ui, monkeypatch, seed_ticket_action):
    _chat_reply(monkeypatch)
    action = seed_ticket_action("keep-1", "act-keep")

    out = ui.send("thanks, one more question about wifi", [], "keep-1", "", action)
    _, pending, card, approve_btn, reject_btn = out[1:6]

    assert pending == action
    assert card != ui.NO_CARD and "waiting for your approval" in card
    assert approve_btn["interactive"] is True and reject_btn["interactive"] is True


def test_kept_draft_can_still_be_approved_exactly_once(ui, monkeypatch, fake_mcp, seed_ticket_action):
    _chat_reply(monkeypatch)
    action = seed_ticket_action("keep-2", "act-keep-2")

    out = ui.send("what is the wifi password policy?", [], "keep-2", "", action)
    history, pending = out[1], out[2]
    result = ui.decide(True, history, "keep-2", pending)

    assert result[1] is None  # decided: card cleared
    assert fake_mcp.create_approved_ticket.await_count == 1


def test_decision_clears_the_draft(ui, fake_mcp, seed_ticket_action):
    action = seed_ticket_action("keep-3", "act-keep-3")
    _, pending, card, *_ = ui.decide(False, [], "keep-3", action)
    assert pending is None and card == ui.NO_CARD
    fake_mcp.create_approved_ticket.assert_not_awaited()


def test_no_draft_means_no_card(ui, monkeypatch):
    _chat_reply(monkeypatch)
    out = ui.send("hello there", [], "keep-4", "", None)
    assert out[2] is None and out[3] == ui.NO_CARD
