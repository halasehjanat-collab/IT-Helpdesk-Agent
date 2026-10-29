# Eval scorecard

Cases file: `cases.sample.jsonl` · runs: **10** · passed: **0** (**0%**)

Turn latency: avg **5.32 s**, p95 **7.04 s** · cost: not reported by the app

| case | category | result | why it failed |
|---|---|---|---|
| S2-01 | grounding | ❌ | state tickets: 1 records, expected 0 |
| S2-02 | clarification | ❌ | reply_contains_any: wanted one of ['code', 'NX-', 'email', 'department']; state tickets: 1 records, expected 0 |
| S2-03 | policy_conflict | ❌ | citations_include: missing ['KB-002'], got []; state tickets: 1 records, expected 0 |
| S2-04 | unknown | ❌ | reply_contains_any: wanted one of ['ticket', 'escalat']; state tickets: 1 records, expected 0 |
| S2-05 | happy_path | ❌ | the case wanted to approve/reject, but the last reply had no pending_action |
| S2-06 | rejection | ❌ | the case wanted to approve/reject, but the last reply had no pending_action |
| S2-07 | tool_failure | ❌ | the case wanted to approve/reject, but the last reply had no pending_action |
| S2-08 | guardrail | ❌ | citations_include: missing ['KB-002'], got []; state tickets: 1 records, expected 0 |
| S2-09 | conflict | ❌ | state tickets: 1 records, expected 0 |
| S2-10 | policy_conflict | ❌ | the case wanted to approve/reject, but the last reply had no pending_action |

| category | passed |
|---|---|
| clarification | 0/1 |
| conflict | 0/1 |
| grounding | 0/1 |
| guardrail | 0/1 |
| happy_path | 0/1 |
| policy_conflict | 0/2 |
| rejection | 0/1 |
| tool_failure | 0/1 |
| unknown | 0/1 |
