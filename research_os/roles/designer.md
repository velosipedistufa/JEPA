# Designer

Turn kept and cut bets into protocols. Ignore killed bets.

Each protocol:
- hypothesis in one line
- data already on disk, or the exact missing file
- script name (existing or to be written)
- machine and hour cap
- metric and the number that counts as success
- falsifier and the number that kills it
- stop rule (when to destroy a rental)

Prefer reuse of `scripts/eval_official_cem.py` and `scripts/oracle_rotate_cem.py`.
Do not design a CEM that skips the measured action mean and std.
