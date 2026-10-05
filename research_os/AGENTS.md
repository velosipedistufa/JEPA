# Roles

Each role reads the previous role's file in this cycle and the lab.
Each role writes one markdown file. Do not blend roles in one section.

## Librarian

Input: `reports/`, `scripts/`, `lab/`.
Output: `cycles/NNN/01_librarian.md` and an updated `lab/facts.md`, `lab/graph.md`.
May not propose experiments. May not smooth a number. If two reports disagree, record both and the file paths.

## Proposer

Input: librarian file + facts + killed list.
Output: `cycles/NNN/02_proposals.md`.
Eight to twelve bets. Wild bets are allowed if they name a measurement.
At least three bets must not be "map a sentence onto a goal image the current CEM already chases".
Each bet: claim, why now, what would change our mind, rough compute.

## Skeptic

Input: proposals + facts.
Output: `cycles/NNN/03_skeptic.md`.
For each bet: keep, cut, or kill.
Kill if it repeats a row in `lab/killed.md`, needs an action scale other than the measured z-score, or assumes in-place 180° is a language bug.
Cut if the claim is real but the protocol cannot be run by three students.

## Designer

Input: skeptic file. Only keep/cut bets.
Output: `cycles/NNN/04_protocols.md`.
For each survivor: steps, script to write or reuse, machine (Xeon or a timed 4090), success metric, falsifier, stop rule.
No new datasets unless the protocol says exactly which columns and why the live env is not enough.

## Scribe

Input: all four files.
Output: `cycles/NNN/REPORT.md`, update `lab/killed.md` and `lab/graph.md`.
Plain language first, then the table. The team should be able to read the report without the role files.
