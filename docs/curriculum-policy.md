# Multi-Topic Curriculum Policy

## Item bank

The canonical bank is `data/items/item-bank-v1.json` (186 items, rebuilt by
`make items`, described in [`data-card.md`](data-card.md)). Every item names one
concept in `cse-prerequisite-graph-v1`, and the concept's own
prerequisites are copied onto the item so a selection policy never has to walk
the graph to find out what an item depends on.

Difficulty comes from `artifacts/datasets/item-parameters-v1.csv`, not from the
question files: `difficulty_bin` is the 10-point scale the policies are meant
to consume. The research pipeline reads it from Phase 8 onwards; the backend
still reads `difficultyScore` off the question JSON and is repointed at the
bank in Phase 10. Until Phase 3 supplies real responses, `b` is a z-scored
author prior and the bin is exactly the author's own 1–10 rating — the plumbing
is real, the numbers are not yet estimated.

24 of 36 concepts hold fewer than 5 items, so the roadmap can select a
concept it cannot then drill. `validate_bank` prints which ones. Authoring more
items is the fix; weighting around the shortfall would hide it.

## Selection

The backend selects a question from three inputs after every response:

1. The history-aware rule policy determines a bounded next difficulty.
2. The concept roadmap identifies the lowest-mastery eligible concept.
3. The topic policy dynamically weights concepts using topic accuracy, roadmap priority, recent exposure, fatigue, and cognitive load.

The policy avoids repeating recently asked topics, prioritizes topics with weak performance, and awards a small co-teaching bonus for compatible topic pairs. Current pairs include recursion with binary trees, algorithms with data structures, operating systems with networks, and DBMS with object-oriented programming.

`backend/src/data/questions/multitopic_cse.json` holds 126 curated CSE items,
each mapped to a graph concept by `question_to_concept` in
`backend/src/data/concept_graph.json`. It still feeds the backend seed; the
research pipeline reads the consolidated bank instead. Items should be reviewed
by subject-matter experts before use with students.

Create a session from the bank without an LLM by posting `{ "topic": "multi-topic cse" }` to `POST /question`.
