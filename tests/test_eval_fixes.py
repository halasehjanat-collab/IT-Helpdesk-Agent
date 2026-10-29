import asyncio
from pathlib import Path

import agent as agent_module
import app as app_module
from evals import run_isolated_evals as iso
from services import answer_verifier


def test_vpn_clarification_only_without_an_error_code_or_ticket_request():
    needs = agent_module._needs_vpn_clarification
    assert needs("My VPN is not working. Can you fix it?", [])
    assert not needs("My VPN is not working, NX-112", [])
    assert not needs("My VPN is not working. Please open a ticket", [])
    earlier = [{"role": "user", "content": "NakhlaConnect shows NX-112"}]
    assert not needs("My VPN is not working", earlier)


def test_vague_vpn_complaint_asks_for_the_code_without_model_or_retrieval(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("must not be called")

    monkeypatch.setattr(agent_module, "retrieve_chunks", boom)
    monkeypatch.setattr(agent_module, "generate_reply_with_status", boom)
    result = asyncio.run(agent_module.run_agent("My VPN is not working. Can you fix it?"))
    assert "error code" in result["reply"].lower()
    assert result["citations"] == [] and result["pending_action"] is None


def test_unconfirmed_reply_offers_escalation(monkeypatch):
    monkeypatch.setattr(agent_module, "retrieve_chunks", lambda *a, **k: [])
    monkeypatch.setattr(
        agent_module,
        "generate_reply_with_status",
        lambda *a, **k: ("I cannot confirm this from the knowledge base.", None, True),
    )
    result = asyncio.run(agent_module.run_agent("What does ZZ-999 mean?"))
    assert "ticket" in result["reply"].lower()


def test_ticket_trigger_prepare_it_for_me():
    assert app_module._requests_ticket_creation("Please prepare it for me.")


def test_attempted_steps_understands_i_did_all_of_that():
    steps = app_module._attempted_steps(
        ["I did all of that: restarted the client, and reinstalled it"]
    )
    assert len(steps) == 2


def test_software_center_step_does_not_make_a_hardware_issue_software():
    report = "My dock firmware failed after I updated it from Software Center"
    assert app_module._ticket_category(report, "Engineering") == "hardware"


def test_priority_p2_for_cannot_work_and_finance_vpn():
    prio = app_module._ticket_priority
    assert prio("I cannot work, no workaround", "Engineering", None) == "P2"
    assert prio("VPN fails", "Finance", None) == "P2"
    assert prio("I cannot work, no workaround", "Engineering", 10) == "P1"


def test_kb_direct_step_is_only_returned_for_documented_cases():
    step = app_module._kb_direct_escalation_step("VPN down", "Finance", "vpn")
    assert step and step[1] == "KB-016"
    assert app_module._kb_direct_escalation_step("printer jam", "Sales", None) is None


def test_verifier_ignores_filler_words_but_still_checks_codes():
    assert {"error", "code", "means"} <= answer_verifier._STOPWORDS


def test_isolated_eval_env_never_points_at_project_databases(tmp_path):
    env = iso.isolated_env(tmp_path)
    project_data = (Path(iso.ROOT) / "services" / "data").resolve()
    for key in ("HELPDESK_TICKET_DB_PATH", "HELPDESK_METADATA_DB_PATH", "CHAT_MEMORY_DB_PATH"):
        path = Path(env[key]).resolve()
        assert tmp_path.resolve() in path.parents
        assert project_data not in path.parents
    assert env["HELPDESK_API_URL"].endswith(str(iso.HELPDESK_PORT))
