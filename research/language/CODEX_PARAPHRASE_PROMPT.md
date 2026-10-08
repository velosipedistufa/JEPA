# Prompt for Codex: held-out command paraphrases

Paste everything below the line into Codex, opened in the JEPA repo on branch `5090-run-demo-breadcrumbs`.

---

You are writing a held-out test set for a small language parser. A different model will be scored on it, so write the sentences yourself. Do not open or read any file in this repo except the one you create. In particular, do not read anything under `scripts/` or `research/language/` other than this prompt.

## The setting

A top-down 2D game. A grey T-shaped block lies on a table seen from above. A small round pusher can push it. A green T outline on the table marks the target pose. The table is 512 px square and the T is about 100 px long. A person gives the robot one short instruction.

Each instruction means exactly one of three actions:

- `target`: put the T on the green target outline (its pose, not just near it).
- `turn`: rotate the T by some angle, staying roughly where it is. `deg` is signed. Positive means clockwise as seen on the screen, negative means counter-clockwise. The absolute value is between 30 and 180.
- `push`: slide the T in one screen direction, angle free. `dir` is one of `left`, `right`, `up`, `down`, where up is toward the top of the screen. `px` is between 30 and 200.

When the sentence gives no amount: a turn is 90 and a push is 80.
When it only says "a little", "slightly", "a bit" or "nudge": a turn is 45 and a push is 40.
When it says "a lot" or "far": a push is 160.
A quarter turn is 90, half a turn is 180 and an eighth of a turn is 45.
"One T length" is 100 px.

## What to write

Create the file `research/language/paraphrases.jsonl` with exactly 120 lines. Each line is one JSON object:

```
{"id": 0, "text": "...", "action": "turn", "deg": -90, "dir": null, "px": null, "style": "casual"}
```

- `deg` is set only for `turn`. `dir` and `px` are set only for `push`. Use `null` otherwise.
- Write 30 `target` lines, 45 `turn` lines and 45 `push` lines, shuffled.
- Turns: mix the signs evenly, and use a range of amounts (quarter turns, explicit degrees like 30 or 135, half turns, "a bit"). Pushes: all four directions, with a range of amounts (explicit pixels, T lengths, vague words).
- `style` is one of:
  - `plain`: a clear, direct instruction
  - `casual`: everyday speech
  - `indirect`: the goal is described rather than ordered, e.g. "I'd like the T sitting a bit further to the left"
  - `odd`: typos, slang, unusual word order, non-native phrasing, or other names for the block ("the tee", "that piece", "the letter")

  Make at least 30 lines `odd` and at least 20 `indirect`.
- Every line must have one clear correct label under the rules above. If you are unsure about a line, rewrite it.
- Do not use these seven sentences or trivial edits of them: "put the T on the target", "turn the T 90 degrees clockwise", "turn the T 90 degrees counter-clockwise", "push the T left", "push the T right", "push the T up", "push the T down".

Write only that one file. Do not commit. Print the counts per action and per style when done.
