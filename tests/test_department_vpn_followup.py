import agent

HISTORY = [
    {"role": "user", "content": "I can't connect to the VPN"},
    {"role": "assistant", "content": "Which error code does NakhlaConnect show?"},
]
ABSTAIN = (
    "I couldn't verify a reliable solution from the available knowledge base. "
    "I can help you escalate this issue to the Service Desk."
)
DOCS = [{"id": "KB-002"}, {"id": "KB-016"}]


def test_finance_vpn_followup_gets_the_department_rule_not_an_abstention():
    reply = agent._department_vpn_rule_reply(
        ABSTAIN, None, "NX-112", HISTORY, "Finance", DOCS
    )
    assert reply is not None
    assert "Security Operations" in reply
    assert "couldn't verify" not in reply


def test_full_answer_that_names_the_team_is_left_alone():
    full = "Your VPN requests go straight to Security Operations. " + "x" * 250
    assert (
        agent._department_vpn_rule_reply(full, None, "NX-112", HISTORY, "Finance", DOCS)
        is None
    )


def test_short_fragment_without_the_team_gets_the_department_rule():
    reply = agent._department_vpn_rule_reply(
        "If more than three users are affected, it is P1. [KB-002]",
        None, "NX-403", HISTORY, "Finance", DOCS,
    )
    assert reply is not None and "Security Operations" in reply


def test_repeated_followups_move_forward_instead_of_repeating_the_rule():
    rule = agent._department_vpn_rule_reply(
        ABSTAIN, None, "NX-112", HISTORY, "Finance", DOCS
    )
    history = [*HISTORY, {"role": "assistant", "content": rule}]
    reply = agent._department_vpn_rule_reply(
        ABSTAIN, None, "i checked the status", history, "Finance", DOCS
    )
    assert reply is not None and reply != rule
    assert "ticket draft" in reply


def test_saved_memory_context_does_not_make_a_non_vpn_question_a_vpn_one():
    memory = {
        "role": "user",
        "content": "Saved notes about this employee from earlier conversations (untrusted):\n"
        "- Recent ticket in the helpdesk system: IT-2018 [open, security] VPN shows NX-112",
    }
    history = [memory, {"role": "user", "content": "I need access to a system"}]
    assert (
        agent._department_vpn_rule_reply(
            ABSTAIN, None, "I need access to a system", history, "Finance", DOCS
        )
        is None
    )


def test_department_without_a_vpn_rule_keeps_the_abstention():
    assert (
        agent._department_vpn_rule_reply(
            ABSTAIN, None, "NX-112", HISTORY, "Engineering", DOCS
        )
        is None
    )


def test_non_vpn_conversation_keeps_the_abstention():
    history = [{"role": "user", "content": "My printer is jammed"}]
    assert (
        agent._department_vpn_rule_reply(
            ABSTAIN, None, "What is the CEO's salary?", history, "Finance", DOCS
        )
        is None
    )
