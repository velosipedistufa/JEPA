# Autonomous research loop

Protocol for JEPA / LeWorldModel work driven by Grok. Topic-agnostic. The project supplies the domain; this file supplies the method.

A cycle is finished only when a pre-registered claim is supported, weakened, or killed, and the decision is written to disk. Activity is not progress.

## 1. What the loop optimizes

Optimize expected information gain on the current question, not number of runs, papers read, or lines changed.

\[
\text{utility}(a) \approx \mathbb{E}\big[\mathcal{I}(H \mid o_a)\big] - \lambda \cdot \text{cost}(a)
\]

\(H\) is the live hypothesis set. \(a\) is the next action (read, ablate, reimplement, probe). \(o_a\) is the observation that action produces. \(\lambda\) prices compute, calendar time, and irreversibility.

A cheap experiment that can kill the leading hypothesis outranks an expensive experiment that can only decorate it.

## 2. Principles

1. Closed loop, not a pipeline. The only legal inputs to cycle \(n+1\) are the artifacts of cycle \(n\): claim ledger, evidence, open questions, rejected paths. Chat memory is not state.

2. Claim before action. No training run, no code change, no literature dive without a one-sentence claim and a pre-declared falsifier. If the falsifier cannot be checked in this cycle, the claim is too big. Split it.

3. Separate invention from judgment. The step that proposes the hypothesis does not grade it. Review uses the pre-registered metric, the log, and the baseline, not the narrative written after the result.

4. One intervention per cycle. Change one factor against a frozen baseline. Multi-factor jumps are allowed only as an explicit exploratory branch, and they cannot support a causal claim.

5. Literature is a constraint set. Extract claims, assumptions, metrics, and known failure modes. Do not produce summaries. A paper enters the ledger only if it changes a hypothesis, a baseline, or a stop rule.

6. Degenerate solutions are the default suspect. Any objective can be satisfied by a trivial strategy: collapsed representations, leaked future information, metric gaming, data leakage, evaluation on the training distribution. The cycle must name the degenerate solution and include a check that rules it out.

7. Negative results close cycles. A killed hypothesis is a completed output. Deleting it from the ledger is a protocol violation.

8. Reproducibility is part of the claim. Seed, config, data split, metric definition, and commit are required before a number is allowed into the ledger. An unreproduced delta is an anecdote.

9. Budget is a hypothesis variable. "Works if we scale" is not a result. State the compute envelope before the run. Results outside that envelope do not count for this cycle.

10. Stop rules are mandatory. Every cycle ends in exactly one of: continue, deepen, pivot, kill, or escalate. "More experiments" is not a decision.

11. Human gates are rare and explicit. Autonomy covers search, implementation, runs, ablations, and drafting. A human is required for spend above the cycle budget, for claims that leave the project scope, and for external publication.

12. Write the decision, then the story. The ledger row is the product. Prose is a rendering of the ledger.

## 3. Roles inside one agent

Keep these as separate passes, even if the same model runs all of them.

| Pass | May do | Must not do |
|---|---|---|
| Librarian | Retrieve sources, extract claims, flag contradictions | Propose the next experiment |
| Theorist | State hypothesis, assumptions, falsifier | Look at the new result before the falsifier is frozen |
| Builder | Implement the single intervention, pin baseline | Change the metric or the claim |
| Runner | Execute, log, recover from crashes | Interpret success |
| Auditor | Score against the falsifier, check degeneracy, check leakage | Soften a failed falsifier |
| Editor | Update ledger and write the cycle note | Add claims the auditor did not score |

## 4. Cycle

```
state = load(ledger, questions, baselines)
question = highest information-gain open question under budget
claim, falsifier, degenerate_check = theorist(question, state)
sources = librarian(claim)          # only if a source can change the claim
freeze(claim, falsifier, metric, budget)
intervention = builder(claim)       # one delta vs pinned baseline
obs = runner(intervention)
verdict = auditor(obs, falsifier, degenerate_check)
decision = stop_rule(verdict)
write(ledger_row, cycle_note, next_questions)
```

### 4.1 Frame

Write, before any action:

- Question: one sentence, answerable by evidence this project can generate.
- Claim: directional, scoped, comparable to a named baseline.
- Assumptions: what must be true for the claim to be meaningful.
- Falsifier: the observation that kills the claim. Numeric if possible.
- Degenerate check: the trivial way to pass, and the measurement that blocks it.
- Cost: GPU-hours or wall-clock, data, risk to the baseline branch.
- Non-goals: what this cycle will not try to show.

If framing takes longer than the experiment, the question is still too vague. Narrow it.

### 4.2 Ground

Search only for the frozen claim. For each source keep:

- citation
- claim it makes
- conditions under which it holds
- whether it agrees, contradicts, or is silent on our claim
- what we would do differently if we trust it

Stop when a new source has not changed the claim, the baseline, or the falsifier. Unread papers are not a backlog; they are out of scope until a claim needs them.

### 4.3 Intervene

Pin the baseline: commit, config, seed list, metric script. Branch. Apply one change. If the change does not parse as a single factor, split the cycle.

Allowed intervention types, in increasing cost:

1. Re-read and re-derive. Often enough to kill a claim.
2. Recompute a published number on our stack.
3. Ablation or probe on an existing checkpoint.
4. Short training run inside the budget.
5. New data or new architecture. Requires an escalate or an explicit budget raise.

Implementation bugs are not results. A failed run restarts the runner, not the theorist, unless the failure reveals an assumption was false.

### 4.4 Audit

Score only the frozen falsifier. Record:

- metric on baseline and on intervention, same seeds
- effect direction and magnitude
- degenerate-check outcome
- confounds observed (split leakage, seed sensitivity, eval bug)
- verdict: supports / weakens / kills / inconclusive

Inconclusive is legal only if the pre-registered check could not be computed. It is not a synonym for "mixed."

### 4.5 Decide

| Verdict | Decision | Next cycle |
|---|---|---|
| Supports, effect holds under degenerate check and a second seed | Deepen | Stress the same claim: shift, scale, or transfer |
| Supports, but one seed or one setup | Continue | Reproduction cycle, no new idea |
| Weakens | Pivot | Revise the claim; do not add terms to save it |
| Kills | Kill | Archive. Promote the next question |
| Inconclusive and budget spent | Escalate or kill | Do not repeat the same run hoping |
| Claim cannot be tested here | Escalate | Human narrows scope or raises budget |

## 5. State on disk

The loop is resumable from these files alone.

```
research/
  LEDGER.md          # one row per closed cycle
  QUESTIONS.md       # ranked open questions
  BASELINES.md       # pinned configs and how to rerun
  cycles/NNNN.md     # framing, intervention, audit, decision
  artifacts/         # configs, logs, plots referenced by cycle id
```

Ledger row:

```
id | date | claim | falsifier | intervention | result | verdict | decision | next
```

A cycle note that is not referenced by a ledger row does not exist.

## 6. Anti-patterns

- Reading until confident. Confidence is not a stopping rule. A changed claim is.
- Stacking hypotheses after a failure. That is a new cycle with a new falsifier.
- Reporting the best seed. Pre-register the seed set.
- Treating a loss decrease as evidence of the intended capability. Run the degenerate check.
- Expanding scope to match a surprising plot. Surprise opens a question; it does not rewrite the last claim.
- Keeping rejected paths only in chat. The next session will rediscover them.
- Human-in-the-loop on every step. That is assistance, not a research loop. Gate only budget, scope, and release.

## 7. Operating note for Grok

At the start of a session, read `LEDGER.md` and `QUESTIONS.md` before proposing work. Do not invent a new direction if a continue or deepen decision is open.

Each session does at most one cycle, or a clean fraction of one (frame only, run only, audit only). Ending mid-cycle is acceptable; ending without a written artifact is not.

When blocked, write the block into the cycle note and either narrow the claim or escalate. Do not fill the block with adjacent literature.
