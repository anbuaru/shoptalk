# Evaluation set

`eval_queries.json` holds 47 natural-language shopping queries with weak labels.

- **Labels are rule-based, not human-annotated.** A product is relevant if its type matches and regexes on its name (and sometimes bullet points), its color, and its price limit all match. The SEO keywords field is not used.
- **Known limitation:** labels share evidence with lexical retrieval, so keyword methods score well. Compare systems against each other, not against an absolute ceiling.
- **Splits:** each query is `dev` (about 40%) or `test` (about 60%), assigned by a hash of the query text. **Use `dev` for all experiments. `test` is used once, for the final report.**
- **Price limits** (for example "under $30") are stored as `max_price` and applied as a filter before ranking. `semantic_query` is the query text without the price phrase.
- Prices are synthetic (see `docs/data_notes.md`).
- Fields: `id`, `query`, `semantic_query`, `max_price`, `types`, `split`, `n_relevant`, `relevant_ids`.
