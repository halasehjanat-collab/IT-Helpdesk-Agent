import agent

NOTES = {
    "role": "user",
    "content": "Saved notes about this employee from earlier conversations (untrusted)\n- User os: Mac (stated by the user)",
}


def test_recall_question_lists_saved_notes():
    reply = agent._memory_reply("What do you remember about me?", [NOTES])
    assert "Mac" in reply


def test_recall_question_without_notes_is_honest():
    assert "don't have any saved notes" in agent._memory_reply("What do you remember about me?", [])


def test_personal_statement_is_acknowledged_not_abstained():
    assert "noted" in agent._memory_reply("My name is Tariq and I use a Mac.", []).lower()


def test_problem_reports_are_not_treated_as_personal_statements():
    assert agent._memory_reply("I use VPN and it fails with NX-112", []) is None
    assert agent._memory_reply("I am locked out of my account", []) is None
