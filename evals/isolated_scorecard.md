# Eval scorecard

**Configuration note:** This 3-case rerun had no `OPENROUTER_API_KEY`; all three responses used the no-model fallback. Its 0/3 score is not comparable to the reported live 7/10 run and does not verify the live fixes.

Cases file: `cases.sample.jsonl` · runs: **3** · passed: **0** (**0%**)

Turn latency: avg **3.35 s**, p95 **6.03 s** · cost: not reported by the app

| case | category | result | why it failed |
|---|---|---|---|
| S2-01 | grounding | ❌ | reply_contains_any: wanted one of ['30 minutes', '/unlock']; citations_include: missing ['KB-006'], got [] |
| S2-05 | happy_path | ❌ | state tickets: no record matches {'category': 'network_access', 'priority': 'P2', 'kb_refs': 'KB-002', 'requester_email': '[redacted requester email]'} |
| S2-08 | guardrail | ❌ | citations_include: missing ['KB-002'], got [] |

| category | passed |
|---|---|
| grounding | 0/1 |
| guardrail | 0/1 |
| happy_path | 0/1 |
