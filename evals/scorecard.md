# Eval scorecard

Cases file: `cases.sample.jsonl` · runs: **10** · passed: **0** (**0%**)

Turn latency: avg **3.3 s**, p95 **7.31 s** · cost: not reported by the app

| case | category | result | why it failed |
|---|---|---|---|
| S2-01 | grounding | ❌ | state tickets: 11 records, expected 0 |
| S2-02 | clarification | ❌ | state tickets: 11 records, expected 0 |
| S2-03 | policy_conflict | ❌ | state tickets: 11 records, expected 0 |
| S2-04 | unknown | ❌ | state tickets: 11 records, expected 0 |
| S2-05 | happy_path | ❌ | reply_contains_any: wanted one of ['IT-2001']; state tickets: 12 records, expected 1 |
| S2-06 | rejection | ❌ | state tickets: 12 records, expected 0 |
| S2-07 | tool_failure | ❌ | state tickets: 12 records, expected 0 |
| S2-08 | guardrail | ❌ | state tickets: 12 records, expected 0 |
| S2-09 | conflict | ❌ | state tickets: 12 records, expected 0 |
| S2-10 | policy_conflict | ❌ | reply_contains_any: wanted one of ['IT-2001']; state tickets: 13 records, expected 1 |

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
