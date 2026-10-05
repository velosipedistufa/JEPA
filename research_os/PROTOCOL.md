# Research OS

File-based lab for one Grok session that plays several roles in order.
It is not a swarm of live models. Roles are contracts. The shared state is this folder.
A cycle reads `lab/`, writes `cycles/NNN/`, then updates `lab/` from what it actually produced.

## Why this shape

Hard problems die when one chat both invents and grades itself.
This OS splits those jobs and forces every claim to point at a file.

Measured facts live in `lab/facts.md`.
Ideas live in `cycles/`.
A fact may enter `lab/facts.md` only with a path to a report, log, or script.
Hypotheses never get promoted by enthusiasm.

## Layout

```
research_os/
  PROTOCOL.md          this loop
  AGENTS.md            role contracts
  SETTINGS.md          how to launch a cycle
  roles/               one card per role
  lab/facts.md         measured only
  lab/killed.md        bets we will not rerun
  lab/graph.md         claim → evidence edges
  cycles/NNN/          one directory per cycle
```

## Cycle

1. Librarian. Refresh `lab/` from `reports/` and `scripts/`. No new science.
2. Proposer. Free bets. At least three must not be "language indexes a goal image".
3. Skeptic. Kill or cut each bet. Cite a measured fact or a missing measurement.
4. Designer. Turn survivors into protocols: machine, script, metric, falsifier, hours.
5. Scribe. Write the cycle report. Update graph and killed list. Do not invent numbers.

Stop. Do not run CEM, training, or downloads inside a planning cycle.
